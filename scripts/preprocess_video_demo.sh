#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
VIDEO_PATH="${1:?Usage: bash scripts/preprocess_video_demo.sh path/to/video.mp4}"
CASE_ID="${CASE_ID:-demo_case}"

"${PYTHON_BIN}" extension_audit.py preprocess-video \
  --video "${VIDEO_PATH}" \
  --audio-output "artifacts/audit/${CASE_ID}/audio.wav" \
  --frames-dir "artifacts/audit/${CASE_ID}/frames" \
  --frame-fps "${FRAME_FPS:-1}" \
  --probe \
  --output "artifacts/audit/${CASE_ID}/preprocess.json"
