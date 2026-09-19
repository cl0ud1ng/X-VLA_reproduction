#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/mnt/data/zxw/cross-embodiment_generalization/X-VLA_reproduction
ACCELERATE=/mnt/mnt/data/zxw/cross-embodiment_generalization/model_test/RoboTwin/.venv/bin/accelerate
SEED=${SEED:-0}
PORT=${PORT:-29593}

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$ACCELERATE" launch \
  --multi_gpu \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  train.py \
  --models models/X-VLA-Pt \
  --train_metas_path outputs/robotwin_ft/manifests/official_clean_50/total.json \
  --output_dir "outputs/robotwin_ft/domain_balanced/seed${SEED}" \
  --sampler_mode domain_balanced \
  --batch_size 8 \
  --num_workers 4 \
  --learning_rate 1e-5 \
  --learning_coef 0.1 \
  --weight_decay 0.0 \
  --betas 0.9 0.95 \
  --iters 30000 \
  --freeze_steps 1000 \
  --warmup_steps 2000 \
  --use_cosine_decay \
  --min_lr_ratio 0.1 \
  --max_grad_norm 1.0 \
  --save_interval 5000 \
  --log_interval 20 \
  --seed "$SEED" \
  --base_seed "$SEED" \
  --mixed_precision fp16
