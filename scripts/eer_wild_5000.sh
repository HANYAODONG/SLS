#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

"${PYTHON_BIN}" evaluate_in_the_wild.py scores/scores_Wild_5000.txt keys/Wild_5000 eval
