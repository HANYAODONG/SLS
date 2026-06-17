import json
from pathlib import Path

import numpy as np


def load_wespeaker_model(model_name="chinese"):
    try:
        import wespeaker
    except ImportError as exc:
        raise RuntimeError(
            "WeSpeaker is not installed. Install optional dependency `wespeaker` "
            "before using speaker enrollment or matching."
        ) from exc
    return wespeaker.load_model(model_name)


def extract_embedding(audio_path, model_name="chinese"):
    model = load_wespeaker_model(model_name)
    embedding = model.extract_embedding(str(audio_path))
    return np.asarray(embedding, dtype=np.float32).reshape(-1)


def load_enrollment(enrollment_path):
    enrollment_path = Path(enrollment_path)
    if not enrollment_path.is_file():
        return []
    rows = []
    with enrollment_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                item = json.loads(line)
                item["embedding"] = np.asarray(item["embedding"], dtype=np.float32)
                rows.append(item)
    return rows


def save_enrollment(rows, enrollment_path):
    enrollment_path = Path(enrollment_path)
    enrollment_path.parent.mkdir(parents=True, exist_ok=True)
    with enrollment_path.open("w", encoding="utf-8") as handle:
        for item in rows:
            payload = dict(item)
            payload["embedding"] = np.asarray(payload["embedding"]).tolist()
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def enroll_speaker(name, audio_path, enrollment_path, model_name="chinese"):
    rows = load_enrollment(enrollment_path)
    rows.append(
        {
            "name": name,
            "audio_path": str(audio_path),
            "embedding": extract_embedding(audio_path, model_name=model_name),
        }
    )
    save_enrollment(rows, enrollment_path)
    return {"name": name, "audio_path": str(audio_path), "enrollment_path": str(enrollment_path)}


def cosine_similarity(left, right):
    left = np.asarray(left, dtype=np.float32)
    right = np.asarray(right, dtype=np.float32)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0:
        return 0.0
    return float(np.dot(left, right) / denom)


def match_speaker(audio_path, enrollment_path, model_name="chinese"):
    rows = load_enrollment(enrollment_path)
    if not rows:
        raise RuntimeError("Enrollment database is empty: {}".format(enrollment_path))
    query = extract_embedding(audio_path, model_name=model_name)
    scored = [
        {
            "name": row["name"],
            "audio_path": row["audio_path"],
            "similarity": cosine_similarity(query, row["embedding"]),
        }
        for row in rows
    ]
    scored.sort(key=lambda item: item["similarity"], reverse=True)
    return scored[0], scored

