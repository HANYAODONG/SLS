#!/usr/bin/env bash
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python main.py \
  --track DF \
  --eval \
  --protocols_path database/ASVspoof_DF_cm_protocols/ASVspoof2021.DF.cm.eval.first5000.trl.txt \
  --database_path data/ASVspoof2021_DF_eval \
  --xlsr_checkpoint xlsr2_300m.pt \
  --model_path MMpaper_model.pth \
  --eval_output scores/scores_DF_5000.txt \
  --eval_batch_size "${EVAL_BATCH_SIZE:-1}" \
  --disable_cudnn
