import json
import os
import subprocess


def run_speaker_command(speaker_python, args):
    env = dict(os.environ)
    env.pop("SPEAKER_PYTHON", None)
    try:
        completed = subprocess.run(
            [speaker_python, "extension_audit.py"] + args,
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
    except FileNotFoundError as exc:
        raise RuntimeError("SPEAKER_PYTHON not found: {}".format(speaker_python)) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RuntimeError("Speaker subprocess failed: {}".format(detail)) from exc
    return json.loads(completed.stdout)
