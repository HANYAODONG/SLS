import numpy as np
import torch
import torchaudio

from explainability.audio_utils import ensure_length


def add_noise_snr(audio, snr_db, rng):
    audio = np.asarray(audio, dtype=np.float32)
    signal_power = np.mean(audio ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=audio.shape[0]).astype(np.float32)
    return ensure_length(audio + noise, len(audio))


def resample_roundtrip(audio, sample_rate):
    tensor = torch.tensor(audio, dtype=torch.float32).view(1, -1)
    down = torchaudio.transforms.Resample(sample_rate, 8000)(tensor)
    up = torchaudio.transforms.Resample(8000, sample_rate)(down)
    return ensure_length(up.view(-1).numpy(), len(audio))


def trim_and_pad(audio, sample_rate):
    trim = min(int(0.05 * sample_rate), max(1, len(audio) // 20))
    if trim <= 0:
        return audio
    trimmed = audio[trim:-trim] if len(audio) > trim * 2 else audio
    return ensure_length(trimmed, len(audio))


def build_variants(audio, config):
    rng = np.random.RandomState(1234)
    sample_rate = int(config.get("sample_rate", 16000))
    variants = [{"name": "original", "audio": audio}]
    tests = set(config.get("stability_tests", []))
    if "volume" in tests:
        variants.append({"name": "volume_0.9", "audio": ensure_length(audio * 0.9, len(audio))})
        variants.append({"name": "volume_1.1", "audio": ensure_length(audio * 1.1, len(audio))})
    if "noise" in tests:
        snr = float(config.get("stability_noise_snr_db", 30))
        variants.append({"name": "noise_{}_a".format(int(snr)), "audio": add_noise_snr(audio, snr, rng)})
        variants.append({"name": "noise_{}_b".format(int(snr)), "audio": add_noise_snr(audio, snr, rng)})
    if "resample" in tests:
        variants.append({"name": "resample_roundtrip", "audio": resample_roundtrip(audio, sample_rate)})
    if "trim" in tests:
        variants.append({"name": "trim_pad", "audio": trim_and_pad(audio, sample_rate)})
    return variants


def stability_analysis(adapter, audio, config):
    rows = []
    for item in build_variants(audio, config):
        pred = adapter.predict_temp_wav(item["audio"])
        rows.append({"name": item["name"], "fake_probability": pred["fake_probability"]})
    scores = [row["fake_probability"] for row in rows]
    score_range = max(scores) - min(scores) if scores else 0.0
    if score_range <= 0.08:
        level = "high"
    elif score_range <= 0.20:
        level = "medium"
    else:
        level = "low"
    return {
        "scores": rows,
        "mean": float(np.mean(scores)) if scores else None,
        "std": float(np.std(scores)) if scores else None,
        "range": float(score_range),
        "level": level,
    }
