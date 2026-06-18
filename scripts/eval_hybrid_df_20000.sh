#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a trained hybrid checkpoint, for example models/.../best.pth}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
XLSR_CHECKPOINT="${XLSR_CHECKPOINT:-xlsr2_300m.pt}"
DATABASE_PATH="${DATABASE_PATH:-data/ASVspoof2021_DF_eval}"
OUTPUT_PATH="${OUTPUT_PATH:-scores/scores_Hybrid_DF_20000.txt}"

"${PYTHON_BIN}" main_hybrid.py \
  --track DF \
  --eval \
  --protocols_path database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.first20000.trl.txt \
  --database_path "${DATABASE_PATH}" \
  --xlsr_checkpoint "${XLSR_CHECKPOINT}" \
  --model_path "${MODEL_PATH}" \
  --eval_output "${OUTPUT_PATH}" \
  --eval_batch_size "${EVAL_BATCH_SIZE}" \
  --use_stat_sls "${USE_STAT_SLS:-1}" \
  --use_swiglu "${USE_SWIGLU:-1}" \
  --pooling_type "${POOLING_TYPE:-cgta}" \
  --disable_cudnn
