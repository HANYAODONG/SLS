#!/usr/bin/env bash
set -euo pipefail

SPEAKER_ENV="${SPEAKER_ENV:-venv_speaker}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

"${PYTHON_BIN}" -m venv "${SPEAKER_ENV}"
"${SPEAKER_ENV}/bin/python" -m pip install --upgrade pip setuptools wheel
"${SPEAKER_ENV}/bin/pip" install -r requirements-speaker.txt
"${SPEAKER_ENV}/bin/pip" install onnxruntime

"${SPEAKER_ENV}/bin/python" - <<'PY'
from pathlib import Path
import site

site_packages = Path(site.getsitepackages()[0])

(site_packages / "wespeaker_compat.py").write_text(
    '''"""Compatibility patches loaded by wespeaker_compat.pth."""\n\n'''
    '''try:\n'''
    '''    import torchaudio\n'''
    '''    if not hasattr(torchaudio, "set_audio_backend"):\n'''
    '''        def _set_audio_backend(_backend):\n'''
    '''            return None\n'''
    '''        torchaudio.set_audio_backend = _set_audio_backend\n'''
    '''except Exception:\n'''
    '''    pass\n''',
    encoding="utf-8",
)
(site_packages / "wespeaker_compat.pth").write_text("import wespeaker_compat\n", encoding="utf-8")

frontend_init = site_packages / "wespeaker" / "frontend" / "__init__.py"
if frontend_init.is_file():
    frontend_init.write_text(
        '''# Patched for the isolated SLS speaker environment. Optional frontends\n'''
        '''# are imported lazily so fbank models can run when s3prl/torchaudio\n'''
        '''# compatibility paths are unavailable.\n\n'''
        '''try:\n'''
        '''    from .s3prl import S3prlFrontend\n'''
        '''except Exception:\n'''
        '''    S3prlFrontend = None\n\n'''
        '''try:\n'''
        '''    from .whisper_encoder import whisper_encoder\n'''
        '''except Exception:\n'''
        '''    whisper_encoder = None\n\n'''
        '''try:\n'''
        '''    from .w2vbert import W2VBertFrontend\n'''
        '''except Exception:\n'''
        '''    W2VBertFrontend = None\n\n'''
        '''frontend_class_dict = {\n'''
        '''    "fbank": None,\n'''
        '''    "s3prl": S3prlFrontend,\n'''
        '''    "whisper_encoder": whisper_encoder,\n'''
        '''    "w2vbert": W2VBertFrontend,\n'''
        '''}\n''',
        encoding="utf-8",
    )
PY

echo "Speaker environment ready: ${SPEAKER_ENV}"
echo "Use it with:"
echo "  SPEAKER_PYTHON=${SPEAKER_ENV}/bin/python venv/bin/python extension_audit.py enroll-speaker ..."
