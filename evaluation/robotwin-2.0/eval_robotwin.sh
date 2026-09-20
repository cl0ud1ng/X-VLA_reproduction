#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
export LD_LIBRARY_PATH="$PROJECT_ROOT/.venv/lib/python3.10/site-packages/torch/lib:$PROJECT_ROOT/.cache/cuda/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
PORT="${PORT:-8000}"

# Define your log directory here:
eval_log_dir="${EVAL_LOG_DIR:-$PROJECT_ROOT/outputs/robotwin_ft/eval}"
mkdir -p "$eval_log_dir"

# Start your RoboTwin client
cd "$PROJECT_ROOT/evaluation/robotwin-2.0"
"$PYTHON" client.py \
    --host 0.0.0.0 \
    --port "$PORT" \
    --eval_log_dir "$eval_log_dir" \
    --num_episodes 100 \
    --device 0 \
    --seed 0 \
    --task_name all \
    --output_path "$eval_log_dir" \
    --task_config demo_clean
