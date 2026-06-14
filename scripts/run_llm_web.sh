#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-7860}"

"${PYTHON_BIN}" web_app.py --host "${HOST}" --port "${PORT}"
