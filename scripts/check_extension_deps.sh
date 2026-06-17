#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

check_cmd() {
  if command -v "$1" >/dev/null 2>&1; then
    echo "[ok] command: $1"
  else
    echo "[missing] command: $1"
  fi
}

check_py() {
  if "${PYTHON_BIN}" -c "import $1" >/dev/null 2>&1; then
    echo "[ok] python module: $1"
  else
    echo "[missing] python module: $1"
  fi
}

check_cmd ffmpeg
check_cmd ffprobe
if "${PYTHON_BIN}" -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())" >/dev/null 2>&1; then
  echo "[ok] python fallback: imageio_ffmpeg"
else
  echo "[missing] python fallback: imageio_ffmpeg"
fi
check_py wespeaker
if [[ -n "${SPEAKER_PYTHON:-}" ]] && "${SPEAKER_PYTHON}" -c "import wespeaker" >/dev/null 2>&1; then
  echo "[ok] speaker env module: wespeaker (${SPEAKER_PYTHON})"
elif [[ -x "venv_speaker/bin/python" ]] && venv_speaker/bin/python -c "import wespeaker" >/dev/null 2>&1; then
  echo "[ok] speaker env module: wespeaker (venv_speaker/bin/python)"
else
  echo "[missing] speaker env module: wespeaker"
fi
check_py whisper
check_py pymilvus
