#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON" ]]; then
  echo "Missing project .venv: run scripts/bootstrap_robotwin2.sh" >&2
  exit 1
fi

SEED=${SEED:-1}
PORT=${PORT:-29595}
WANDB_MODE_VALUE=${WANDB_MODE_VALUE:-online}
# This is the 30k-step variant of the formal 8-GPU FSDP full-finetuning run.
# Keep the physical/global batch and learning rate fixed; scale warmup from
# 1k to 2k steps to preserve the original 1,000/15,000 schedule ratio.

cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 "$PYTHON" -m accelerate.commands.launch \
  --multi_gpu \
  --num_processes 8 \
  --num_machines 1 \
  --mixed_precision fp16 \
  --dynamo_backend no \
  --main_process_port "$PORT" \
  train.py \
  --models models/X-VLA-Pt \
  --train_metas_path outputs/robotwin_ft/manifests/official_clean_50/total.json \
  --output_dir "outputs/robotwin_ft/domain_balanced_30k/seed${SEED}" \
  --sampler_mode domain_balanced \
  --batch_size 32 \
  --global_batch_size 256 \
  --gradient_accumulation_steps 1 \
  --num_workers 4 \
  --learning_rate 5e-5 \
  --learning_coef 0.1 \
  --weight_decay 0.0 \
  --betas 0.9 0.95 \
  --iters 30000 \
  --finetune_mode full \
  --freeze_steps 0 \
  --warmup_steps 2000 \
  --use_cosine_decay \
  --min_lr_ratio 0.1 \
  --max_grad_norm 1.0 \
  --save_interval 10000 \
  --save_training_state \
  --log_interval 20 \
  --seed "$SEED" \
  --base_seed "$SEED" \
  --mixed_precision fp16 \
  --distributed_backend fsdp \
  --fsdp_auto_wrap_policy transformer_based_wrap \
  --fsdp_activation_checkpointing \
  --report_to wandb \
  --wandb_project xvla-robotwin2-ft \
  --wandb_run_name "robotwin2-fsdp-fullft-b256-30k-s${SEED}" \
  --wandb_mode "$WANDB_MODE_VALUE"
