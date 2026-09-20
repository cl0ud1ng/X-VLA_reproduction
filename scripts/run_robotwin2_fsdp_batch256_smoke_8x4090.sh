#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/mnt/mnt/data/zxw/cross-embodiment_generalization/X-VLA_reproduction
ACCELERATE=/mnt/mnt/data/zxw/cross-embodiment_generalization/model_test/RoboTwin/.venv/bin/accelerate
OUTPUT_DIR="$PROJECT_ROOT/outputs/robotwin_ft/fsdp_fullft_batch256_smoke"
PORT=${PORT:-29594}

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
  --output_dir "$OUTPUT_DIR" \
  --sampler_mode domain_balanced \
  --batch_size 32 \
  --global_batch_size 256 \
  --gradient_accumulation_steps 1 \
  --num_workers 0 \
  --learning_rate 1e-5 \
  --learning_coef 0.1 \
  --weight_decay 0.0 \
  --betas 0.9 0.95 \
  --iters 2 \
  --finetune_mode full \
  --freeze_steps 0 \
  --warmup_steps 0 \
  --max_grad_norm 1.0 \
  --log_interval 1 \
  --seed 0 \
  --base_seed 0 \
  --mixed_precision fp16 \
  --distributed_backend fsdp \
  --fsdp_auto_wrap_policy transformer_based_wrap \
  --fsdp_activation_checkpointing \
  --disable_checkpoint \
  --report_to none \
  --run_report_path "$OUTPUT_DIR/report.json"
