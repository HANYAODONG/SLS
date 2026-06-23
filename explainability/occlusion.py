import numpy as np

from explainability.audio_utils import ensure_length


def make_time_windows(num_samples, sample_rate, window_seconds, hop_seconds):
    window = max(1, int(window_seconds * sample_rate))
    hop = max(1, int(hop_seconds * sample_rate))
    if num_samples <= window:
        return [(0, num_samples)]
    windows = []
    for start in range(0, num_samples - window + 1, hop):
        windows.append((start, start + window))
    if windows[-1][1] < num_samples:
        windows.append((num_samples - window, num_samples))
    return windows


def apply_time_mask(audio, start, end, mode, rng):
    masked = np.asarray(audio, dtype=np.float32).copy()
    if mode == "zero":
        masked[start:end] = 0.0
    elif mode == "noise":
        segment = masked[start:end]
        scale = max(float(np.std(segment)), 1e-4) * 0.05
        masked[start:end] = rng.normal(0.0, scale, size=end - start).astype(np.float32)
    else:
        raise ValueError("Unsupported time occlusion mode: {}".format(mode))
    return ensure_length(masked, len(audio))


def time_occlusion(adapter, audio, base_fake_probability, config):
    sample_rate = int(config.get("sample_rate", 16000))
    modes = config.get("time_occlusion_modes", ["zero", "noise"])
    windows = make_time_windows(
        len(audio),
        sample_rate,
        float(config.get("time_occlusion_window", 0.5)),
        float(config.get("time_occlusion_hop", 0.25)),
    )
    rng = np.random.RandomState(1234)
    rows = []
    for start, end in windows:
        mode_rows = []
        drops = []
        for mode in modes:
            masked = apply_time_mask(audio, start, end, mode, rng)
            pred = adapter.predict_temp_wav(masked)
            masked_prob = pred["fake_probability"]
            drop = base_fake_probability - masked_prob
            drops.append(drop)
            mode_rows.append(
                {
                    "mode": mode,
                    "masked_probability": masked_prob,
                    "probability_drop": drop,
                }
            )
        rows.append(
            {
                "start": start / float(sample_rate),
                "end": end / float(sample_rate),
                "average_probability_drop": float(np.mean(drops)),
                "modes": mode_rows,
            }
        )
    rows.sort(key=lambda item: item["average_probability_drop"], reverse=True)
    return rows
