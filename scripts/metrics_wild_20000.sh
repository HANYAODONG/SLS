#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

"${PYTHON_BIN}" analysis/score_stats.py \
  --score scores/scores_Wild_20000.txt \
  --metadata keys/Wild_20000/trial_metadata.txt \
  --phase eval \
  --experiment wild_20000 \
  --output reports/metrics_Wild_20000.json

"${PYTHON_BIN}" analysis/roc_auc.py \
  --score scores/scores_Wild_20000.txt \
  --metadata keys/Wild_20000/trial_metadata.txt \
  --phase eval \
  --positive-label bonafide \
  --output-json reports/roc_auc_Wild_20000.json \
  --output-csv reports/roc_curve_Wild_20000.csv \
  --output-svg reports/roc_curve_Wild_20000.svg
