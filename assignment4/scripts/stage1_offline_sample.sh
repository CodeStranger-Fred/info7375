#!/usr/bin/env bash
set -euo pipefail

python -u src/train_stage1_offline.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --train_ratio 0.005 \
  --k 1 \
  --out_samples outputs/stage1_samples.jsonl \
  --out_vstar outputs/stage1_vstar.jsonl
