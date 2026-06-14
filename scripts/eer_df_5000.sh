#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

"${PYTHON_BIN}" evaluate_2021_DF.py scores/scores_DF_5000.txt keys/DF_5000 eval
