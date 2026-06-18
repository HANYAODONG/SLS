#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

"${PYTHON_BIN}" -m py_compile model_hybrid.py main_hybrid.py test_hybrid_modules.py
"${PYTHON_BIN}" test_hybrid_modules.py
