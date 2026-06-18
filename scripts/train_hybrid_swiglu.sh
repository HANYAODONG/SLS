#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-venv/bin/python}"
EPOCHS="${EPOCHS:-10}"
BATCH_SIZE="${BATCH_SIZE:-1}"
DATABASE_PATH="${DATABASE_PATH:-data}"
XLSR_CHECKPOINT="${XLSR_CHECKPOINT:-xlsr2_300m.pt}"

"${PYTHON_BIN}" main_hybrid.py \
  --track DF \
  --database_path "${DATABASE_PATH}" \
  --protocols_path database \
  --xlsr_checkpoint "${XLSR_CHECKPOINT}" \
  --batch_size "${BATCH_SIZE}" \
  --num_epochs "${EPOCHS}" \
  --num_workers "${NUM_WORKERS:-2}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE:-3}" \
  --use_stat_sls 0 \
  --use_swiglu 1 \
  --pooling_type maxpool \
  --comment swiglu \
  --disable_cudnn
