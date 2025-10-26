#!/usr/bin/env bash
set -euo pipefail

python -u src/train_pag_apo.py \
  --model Qwen/Qwen2.5-1.5B-Instruct \
  --train_ratio 0.5 \
  --epochs 1 \
  --lr 2e-5 \
  --lambda_kl 0.01 \
  --vstar_path outputs/stage1_vstar.jsonl \
  --save_dir outputs/checkpoints
