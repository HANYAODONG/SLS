import argparse
import hashlib
import json
import os
import subprocess
import time
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
import torchaudio

import eval_metrics_DF as df_metrics
from analysis.score_stats import read_metadata, read_scores, row_label, row_utt_id
from model import Model


ROOT = Path(__file__).resolve().parent
DEFAULT_THRESHOLD_SCORE = ROOT / "scores" / "scores_DF_20000.txt"
DEFAULT_THRESHOLD_METADATA = ROOT / "keys" / "DF_20000" / "CM" / "trial_metadata.txt"

_MODEL_CACHE = {}
_THRESHOLD_CACHE = {}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_ffmpeg_convert(input_path, output_path):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-ac",
        "1",
        "-ar",
        "16000",
        "-sample_fmt",
        "s16",
        str(output_path),
    ]
    subprocess.run(command, check=True)
    return output_path


def audio_info(path):
    info = {
        "path": str(path),
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "sha256": file_sha256(path) if path.exists() else None,
    }
    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
            info.update(
                {
                    "sample_rate": rate,
                    "channels": handle.getnchannels(),
                    "sample_width_bytes": handle.getsampwidth(),
                    "num_frames": frames,
                    "duration_seconds": frames / float(rate) if rate else None,
                }
            )
    except wave.Error:
        pass
    return info


def load_processed_audio(path, sr=16000):
    try:
        waveform, sample_rate = torchaudio.load(str(path))
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != sr:
            waveform = torchaudio.transforms.Resample(sample_rate, sr)(waveform)
        return waveform.squeeze(0).numpy(), sr
    except Exception:
        with wave.open(str(path), "rb") as handle:
            sample_rate = handle.getframerate()
            channels = handle.getnchannels()
            sample_width = handle.getsampwidth()
            frames = handle.readframes(handle.getnframes())
        if sample_width != 2:
            raise ValueError("Only 16-bit PCM wav fallback is supported")
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if channels > 1:
            audio = audio.reshape(-1, channels).mean(axis=1)
        if sample_rate != sr:
            raise ValueError("Fallback wav loader expects {} Hz audio".format(sr))
        return audio, sr


def pad_audio(x, max_len=64600):
    x_len = x.shape[0]
    if x_len == 0:
        raise ValueError("Cannot pad an empty audio signal")
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    return np.tile(x, (1, num_repeats))[:, :max_len][0]


def threshold_from_scores(score_path=DEFAULT_THRESHOLD_SCORE, metadata_path=DEFAULT_THRESHOLD_METADATA, phase="eval"):
    cache_key = (str(score_path), str(metadata_path), phase)
    if cache_key in _THRESHOLD_CACHE:
        return _THRESHOLD_CACHE[cache_key]

    score_path = Path(score_path)
    metadata_path = Path(metadata_path)
    if not score_path.is_file() or not metadata_path.is_file():
        payload = {
            "threshold": None,
            "eer_percent": None,
            "source_score": str(score_path),
            "source_metadata": str(metadata_path),
            "phase": phase,
            "available": False,
        }
        _THRESHOLD_CACHE[cache_key] = payload
        return payload

    scores = read_scores(score_path)
    metadata = read_metadata(metadata_path)
    bonafide_scores = []
    spoof_scores = []
    has_phase_column = any(len(row) > 7 for row in metadata)
    for row in metadata:
        if has_phase_column and len(row) > 7 and row[7] != phase:
            continue
        utt_id = row_utt_id(row)
        label = row_label(row)
        if utt_id not in scores:
            continue
        if label == "bonafide":
            bonafide_scores.append(scores[utt_id])
        elif label == "spoof":
            spoof_scores.append(scores[utt_id])

    if not bonafide_scores or not spoof_scores:
        payload = {
            "threshold": None,
            "eer_percent": None,
            "source_score": str(score_path),
            "source_metadata": str(metadata_path),
            "phase": phase,
            "available": False,
        }
        _THRESHOLD_CACHE[cache_key] = payload
        return payload

    eer, threshold = df_metrics.compute_eer(
        np.asarray(bonafide_scores),
        np.asarray(spoof_scores),
    )
    payload = {
        "threshold": float(threshold),
        "eer_percent": float(100 * eer),
        "source_score": str(score_path),
        "source_metadata": str(metadata_path),
        "phase": phase,
        "available": True,
    }
    _THRESHOLD_CACHE[cache_key] = payload
    return payload


def score_to_risk(score, threshold_payload):
    threshold = threshold_payload.get("threshold")
    if threshold is None:
        label = "bonafide-like" if score >= 0 else "spoof-like"
        return {
            "label": label,
            "risk_level": "low" if label == "bonafide-like" else "high",
            "margin": None,
            "threshold": None,
        }

    margin = score - threshold
    if margin >= 1.0:
        risk = "low"
    elif margin >= 0:
        risk = "medium"
    elif margin >= -1.0:
        risk = "medium-high"
    else:
        risk = "high"
    return {
        "label": "bonafide-like" if score >= threshold else "spoof-like",
        "risk_level": risk,
        "margin": float(margin),
        "threshold": float(threshold),
    }


def load_model(model_path, xlsr_checkpoint, device):
    cache_key = (str(model_path), str(xlsr_checkpoint), device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    args = SimpleNamespace(xlsr_checkpoint=str(xlsr_checkpoint))
    model = Model(args, device)
    model = nn.DataParallel(model).to(device)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    del state_dict
    if device == "cuda":
        torch.cuda.empty_cache()
    model.eval()
    _MODEL_CACHE[cache_key] = model
    return model


def infer_audio(
    audio_path,
    model_path="MMpaper_model.pth",
    xlsr_checkpoint="xlsr2_300m.pt",
    converted_path=None,
    threshold_score=DEFAULT_THRESHOLD_SCORE,
    threshold_metadata=DEFAULT_THRESHOLD_METADATA,
    disable_cudnn=True,
):
    started = time.time()
    audio_path = Path(audio_path)
    model_path = Path(model_path)
    xlsr_checkpoint = Path(xlsr_checkpoint)
    if disable_cudnn:
        torch.backends.cudnn.enabled = False

    if converted_path is None:
        converted_path = ROOT / "samples" / "processed" / (audio_path.stem + "_16k.wav")
    converted_path = Path(converted_path)
    run_ffmpeg_convert(audio_path, converted_path)

    x, sr = load_processed_audio(converted_path, sr=16000)
    x_pad = pad_audio(x, 64600)
    batch_x = torch.Tensor(x_pad).unsqueeze(0)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(model_path, xlsr_checkpoint, device)
    with torch.no_grad():
        batch_out = model(batch_x.to(device))
        score = float(batch_out[:, 1].data.cpu().numpy().ravel()[0])
        log_probs = batch_out.data.cpu().numpy().ravel().tolist()
        probs = torch.exp(batch_out).data.cpu().numpy().ravel().tolist()

    threshold_payload = threshold_from_scores(threshold_score, threshold_metadata)
    decision = score_to_risk(score, threshold_payload)

    return {
        "input_audio": audio_info(audio_path),
        "processed_audio": audio_info(converted_path),
        "model": {
            "model_path": str(model_path),
            "xlsr_checkpoint": str(xlsr_checkpoint),
            "device": device,
            "score_name": "log_probability_class_1",
        },
        "score": score,
        "log_probabilities": log_probs,
        "probabilities": {
            "spoof": float(probs[0]),
            "bonafide": float(probs[1]),
            "fake_probability": float(probs[0]),
            "bonafide_probability": float(probs[1]),
        },
        "decision": decision,
        "threshold_reference": threshold_payload,
        "processing": {
            "converted_to": "16kHz mono PCM wav",
            "padded_or_cut_samples": 64600,
            "elapsed_seconds": time.time() - started,
        },
        "note": "单条语音结果是模型判别倾向，不等同于数据集级 EER/t-DCF。",
    }


def main():
    parser = argparse.ArgumentParser(description="Run single-audio SLS inference.")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model_path", default="MMpaper_model.pth")
    parser.add_argument("--xlsr_checkpoint", default="xlsr2_300m.pt")
    parser.add_argument("--converted_path", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = infer_audio(
        args.audio,
        model_path=args.model_path,
        xlsr_checkpoint=args.xlsr_checkpoint,
        converted_path=args.converted_path,
    )
    payload = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
