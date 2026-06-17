#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
TOP_N="${TOP_N:-50}"

"${PYTHON_BIN}" analysis/risk_assessment.py \
  --score scores/scores_Wild_20000.txt \
  --metadata keys/Wild_20000/trial_metadata.txt \
  --top-n "${TOP_N}" \
  --output reports/wild_20000_risk_top${TOP_N}.csv
