#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
MODEL_PATH="${MODEL_PATH:-MMpaper_model.pth}"
XLSR_CHECKPOINT="${XLSR_CHECKPOINT:-xlsr2_300m.pt}"

"${PYTHON_BIN}" main.py \
  --track In-the-Wild \
  --database_path release_in_the_wild \
  --protocols_path database/ASVspoof_DF_cm_protocols/in_the_wild.full.eval.txt \
  --model_path "${MODEL_PATH}" \
  --xlsr_checkpoint "${XLSR_CHECKPOINT}" \
  --eval_output scores/scores_Wild_full.txt \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --disable_cudnn
