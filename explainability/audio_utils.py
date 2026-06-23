import tempfile
from pathlib import Path

import numpy as np
import torch
import torchaudio

from single_audio_infer import load_processed_audio, run_ffmpeg_convert


def load_audio_16k(audio_path, sample_rate=16000):
    audio_path = Path(audio_path)
    with tempfile.TemporaryDirectory() as tmpdir:
        converted = Path(tmpdir) / "audio_16k.wav"
        run_ffmpeg_convert(audio_path, converted)
        audio, _ = load_processed_audio(converted, sr=sample_rate)
    return np.asarray(audio, dtype=np.float32)


def save_audio_16k(audio, output_path, sample_rate=16000):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tensor = torch.tensor(audio, dtype=torch.float32).view(1, -1)
    torchaudio.save(str(output_path), tensor, sample_rate)
    return output_path


def ensure_length(audio, length):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.shape[0] == length:
        return audio
    if audio.shape[0] > length:
        return audio[:length]
    return np.pad(audio, (0, length - audio.shape[0]), mode="constant")
