import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def create_timestamp_record(file_path, output_path=None, extra=None):
    file_path = Path(file_path)
    record = {
        "record_id": str(uuid.uuid4()),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "file_path": str(file_path),
        "sha256": sha256_file(file_path),
        "tsa_mode": "local_simulation",
        "note": "Local timestamp record for demo use; not a legal TSA certificate.",
        "extra": extra or {},
    }
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return record

