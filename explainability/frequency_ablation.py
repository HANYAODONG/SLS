import numpy as np

from explainability.audio_utils import ensure_length


def smooth_frequency_attenuation(audio, sample_rate, low_hz, high_hz, attenuation_db=-24.0):
    audio = np.asarray(audio, dtype=np.float32)
    spectrum = np.fft.rfft(audio)
    freqs = np.fft.rfftfreq(audio.shape[0], d=1.0 / sample_rate)
    mask = np.ones_like(freqs, dtype=np.float32)

    transition = max(100.0, (high_hz - low_hz) * 0.15)
    attenuation = float(10 ** (attenuation_db / 20.0))
    core = (freqs >= low_hz) & (freqs <= high_hz)
    mask[core] = attenuation

    left = (freqs >= max(0.0, low_hz - transition)) & (freqs < low_hz)
    if np.any(left):
        ratio = (freqs[left] - (low_hz - transition)) / transition
        mask[left] = 1.0 + (attenuation - 1.0) * ratio

    right = (freqs > high_hz) & (freqs <= high_hz + transition)
    if np.any(right):
        ratio = 1.0 - (freqs[right] - high_hz) / transition
        mask[right] = 1.0 + (attenuation - 1.0) * ratio

    ablated = np.fft.irfft(spectrum * mask, n=audio.shape[0]).astype(np.float32)
    return ensure_length(ablated, len(audio))


def frequency_ablation(adapter, audio, base_fake_probability, config):
    sample_rate = int(config.get("sample_rate", 16000))
    attenuation_db = float(config.get("frequency_attenuation_db", -24.0))
    rows = []
    for low_hz, high_hz in config.get("frequency_bands", []):
        variant = smooth_frequency_attenuation(
            audio,
            sample_rate,
            float(low_hz),
            float(high_hz),
            attenuation_db=attenuation_db,
        )
        pred = adapter.predict_temp_wav(variant)
        prob = pred["fake_probability"]
        rows.append(
            {
                "low_hz": float(low_hz),
                "high_hz": float(high_hz),
                "attenuation_db": attenuation_db,
                "ablated_probability": prob,
                "probability_drop": base_fake_probability - prob,
            }
        )
    rows.sort(key=lambda item: item["probability_drop"], reverse=True)
    return rows
