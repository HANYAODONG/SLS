#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"

"${PYTHON_BIN}" analysis/score_stats.py \
  --score scores/scores_DF_20000.txt \
  --metadata keys/DF_20000/CM/trial_metadata.txt \
  --phase eval \
  --experiment df_20000 \
  --output reports/metrics_DF_20000.json

"${PYTHON_BIN}" analysis/roc_auc.py \
  --score scores/scores_DF_20000.txt \
  --metadata keys/DF_20000/CM/trial_metadata.txt \
  --phase eval \
  --positive-label bonafide \
  --output-json reports/roc_auc_DF_20000.json \
  --output-csv reports/roc_curve_DF_20000.csv \
  --output-svg reports/roc_curve_DF_20000.svg
