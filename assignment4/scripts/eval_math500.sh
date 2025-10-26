#!/usr/bin/env bash
set -euo pipefail

CKPT="outputs/checkpoints"  # or a specific subfolder
python -u src/eval_math500.py --model_or_ckpt "$CKPT"
