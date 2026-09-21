#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${XVLA_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Python interpreter not found or not executable: $PYTHON" >&2
  exit 2
fi

# Diagnostic comparison preset: execute 10 points from each prediction before
# requesting a new horizon. The formal protocol remains exec_points=1 in
# run_robotwin2_rollout_8GPU_exec1.sh.
GPUS="0,1,2,3,4,5,6,7"
TASKS="beat_block_hammer,stack_blocks_two,move_can_pot,open_microwave,place_dual_shoes"
DOMAINS="aloha-agilex,ARX-X5,piper-dual"
for argument in "$@"; do
  case "$argument" in
    --gpus|--gpus=*|--num-gpus|--num-gpus=*|--tasks|--tasks=*|--num-tasks|--num-tasks=*|--domains|--domains=*|--num-domains|--num-domains=*|--exec-points|--exec-points=*)
      echo "Use scripts/run_robotwin2_rollout.py to change GPU/task/domain counts or exec-points." >&2
      exit 2
      ;;
  esac
done

exec "$PYTHON" "$PROJECT_ROOT/scripts/run_robotwin2_rollout.py" \
  --project-root "$PROJECT_ROOT" \
  --python "$PYTHON" \
  --gpus "$GPUS" --num-gpus 8 \
  --tasks "$TASKS" --num-tasks 5 \
  --domains "$DOMAINS" --num-domains 3 \
  --exec-points 20 \
  --output-dir "${EVAL_LOG_DIR:-$PROJECT_ROOT/outputs/robotwin_ft/eval_exec20}" \
  --host "${MODEL_HOST:-127.0.0.1}" --port "${MODEL_PORT:-8000}" \
  --task-config "${TASK_CONFIG:-demo_clean}" \
  --num-episodes "${NUM_EPISODES:-1}" --seed "${ROLLOUT_SEED:-0}" \
  --max-steps "${MAX_STEPS:-0}" --save-video \
  "$@"
