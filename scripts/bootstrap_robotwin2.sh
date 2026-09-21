#!/usr/bin/env bash
set -euo pipefail

# Prepare every machine from a clean clone. Large model/data files and the
# RoboTwin checkout are deliberately ignored by Git and recreated here.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOTWIN_ROOT="$PROJECT_ROOT/third_party/RoboTwin"
ROBOTWIN_URL="${ROBOTWIN_URL:-https://github.com/RoboTwin-Platform/RoboTwin.git}"
ROBOTWIN_COMMIT="${ROBOTWIN_COMMIT:-96c1fea}"
XPOLICYLAB_COMMIT="${XPOLICYLAB_COMMIT:-c37109c500be67d0dea6b36bf7337bbd26e763cd}"
MODEL_ID="${MODEL_ID:-2toINF/X-VLA-Pt}"
# This is the revision represented by the currently validated local
# models/X-VLA-Pt snapshot.  Override it explicitly when changing the base
# checkpoint.
MODEL_REVISION="${MODEL_REVISION:-c1c4a64a7e03ac5b95c468bf1578f3d03651b53b}"
DATA_REVISION="${DATA_REVISION:-981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042}"
PYTHON="$PROJECT_ROOT/.venv/bin/python"
BOOTSTRAP_PYTHON="${BOOTSTRAP_PYTHON:-}"
BOOTSTRAP_PIP_INDEX_URL="${BOOTSTRAP_PIP_INDEX_URL:-https://pypi.org/simple}"
INSTALL_ENV=1
DOWNLOAD_MODEL=1
DOWNLOAD_DATA=1
PREPROCESS=1
WITH_ROLLOUT=0

usage() {
  cat <<'EOF'
Usage: scripts/bootstrap_robotwin2.sh [options]

Options:
  --skip-env        do not create/install the project-local .venv
  --skip-model      do not download the X-VLA foundation checkpoint
  --skip-data       do not download RoboTwin clean archives/assets
  --skip-preprocess do not generate normalized HDF5 and manifests
  --with-rollout    install RoboTwin runtime requirements in the local venv
  -h, --help        show this help

The RoboTwin checkout is placed at third_party/RoboTwin and is required at
ROBOTWIN_COMMIT for preprocessing preflight and simulation rollout.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-env) INSTALL_ENV=0; shift ;;
    --skip-model) DOWNLOAD_MODEL=0; shift ;;
    --skip-data) DOWNLOAD_DATA=0; shift ;;
    --skip-preprocess) PREPROCESS=0; shift ;;
    --with-rollout) WITH_ROLLOUT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

cd "$PROJECT_ROOT"
export UV_PYTHON_INSTALL_DIR="$PROJECT_ROOT/.python"
export UV_PYTHON_BIN_DIR="$PROJECT_ROOT/.python/bin"
export UV_CACHE_DIR="$PROJECT_ROOT/.cache/uv"
export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export TMPDIR="$PROJECT_ROOT/.cache/tmp"
export TORCH_EXTENSIONS_DIR="$PROJECT_ROOT/.cache/torch_extensions"
mkdir -p "$TMPDIR" third_party assets/robotwin data/raw data/processed/robotwin2_ft models

if [[ ! -d "$ROBOTWIN_ROOT/.git" ]]; then
  if [[ -e "$ROBOTWIN_ROOT" ]]; then
    echo "RoboTwin path exists but is not a Git checkout: $ROBOTWIN_ROOT" >&2
    exit 1
  fi
  git clone --recurse-submodules "$ROBOTWIN_URL" "$ROBOTWIN_ROOT"
  git -C "$ROBOTWIN_ROOT" checkout --detach "$ROBOTWIN_COMMIT"
fi
actual_robotwin="$(git -C "$ROBOTWIN_ROOT" rev-parse HEAD)"
if [[ "$actual_robotwin" != "$ROBOTWIN_COMMIT"* ]]; then
  echo "RoboTwin must be $ROBOTWIN_COMMIT, found $actual_robotwin" >&2
  echo "Use a clean checkout; this script will not reset local changes." >&2
  exit 1
fi
git -C "$ROBOTWIN_ROOT" submodule update --init --checkout
actual_xpolicylab="$(git -C "$ROBOTWIN_ROOT/XPolicyLab" rev-parse HEAD)"
if [[ "$actual_xpolicylab" != "$XPOLICYLAB_COMMIT" ]]; then
  echo "XPolicyLab must be $XPOLICYLAB_COMMIT, found $actual_xpolicylab" >&2
  exit 1
fi
if [[ "$WITH_ROLLOUT" -eq 1 ]]; then
  for required in objects files background_texture; do
    if [[ ! -d "$ROBOTWIN_ROOT/assets/$required" ]]; then
      echo "Missing RoboTwin scene asset: $ROBOTWIN_ROOT/assets/$required" >&2
      echo "Install the official RoboTwin scene assets before rollout." >&2
      exit 1
    fi
  done
fi

if [[ ! -x "$PYTHON" ]]; then
  if [[ "$INSTALL_ENV" -eq 0 ]]; then
    echo "Python not found: $PYTHON" >&2
    exit 1
  fi
  if [[ -z "$BOOTSTRAP_PYTHON" ]]; then
    command -v uv >/dev/null || { echo "Install uv or set BOOTSTRAP_PYTHON to Python 3.10" >&2; exit 1; }
    uv python install 3.10.19
    BOOTSTRAP_PYTHON="$UV_PYTHON_INSTALL_DIR/cpython-3.10.19-linux-x86_64-gnu/bin/python3.10"
  fi
  "$BOOTSTRAP_PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Python 3.10 required"'
  "$BOOTSTRAP_PYTHON" -m venv "$PROJECT_ROOT/.venv"
fi
"$PYTHON" -c 'import sys; assert sys.version_info[:2] == (3, 10), "Recreate .venv using Python 3.10"'

if [[ "$INSTALL_ENV" -eq 1 ]]; then
  "$PYTHON" -m pip install --index-url "$BOOTSTRAP_PIP_INDEX_URL" 'pip==25.3' 'setuptools==69.5.1' wheel
  "$PYTHON" -m pip install 'torch==2.4.1' 'torchvision==0.19.1' \
    --index-url "${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
  "$PYTHON" -m pip install --index-url "$BOOTSTRAP_PIP_INDEX_URL" \
    -c configs/robotwin2_ft/runtime_constraints.txt -r requirements.txt
  if [[ "$WITH_ROLLOUT" -eq 1 ]]; then
    "$PYTHON" -m pip install --index-url "$BOOTSTRAP_PIP_INDEX_URL" \
      -c configs/robotwin2_ft/runtime_constraints.txt -r configs/robotwin2_ft/rollout_requirements.txt
    "$PYTHON" -m pip install --index-url "$BOOTSTRAP_PIP_INDEX_URL" \
      -c configs/robotwin2_ft/runtime_constraints.txt -e "$ROBOTWIN_ROOT/XPolicyLab"
    # RoboTwin robot.py imports CuroboPlanner even for replay.
    if [[ "$WITH_ROLLOUT" -eq 1 ]]; then
      if [[ ! -d third_party/curobo ]]; then
        git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git third_party/curobo
      fi
      [[ "$(git -C third_party/curobo rev-parse HEAD)" == d64c4b005459db10c5dd867d8b30a87d5bda9bdb ]] || { echo 'Wrong cuRobo commit' >&2; exit 1; }
      : "${CUDA_HOME:?Set CUDA_HOME to a complete CUDA 12.1 toolkit (nvcc and headers required for cuRobo)}"
      CUDA_RUNTIME_INCLUDE="$($PYTHON -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent / "include")')"
      CUDA_PACKAGE_INCLUDE="$($PYTHON -c 'import pathlib,torch; print(pathlib.Path(torch.__file__).parent.parent / "nvidia/cuda_runtime/include")')"
      CUDA_LOCAL_LIB="$PROJECT_ROOT/.cache/cuda/lib"
      mkdir -p "$CUDA_LOCAL_LIB"
      if [[ ! -e "$CUDA_LOCAL_LIB/libcudart.so" ]]; then
        cudart=$(find "$CUDA_HOME" "$CUDA_RUNTIME_INCLUDE" -name 'libcudart.so.*' -type f -print -quit)
        [[ -n "$cudart" ]] && ln -s "$cudart" "$CUDA_LOCAL_LIB/libcudart.so"
      fi
      "$PYTHON" -m pip install --index-url "$BOOTSTRAP_PIP_INDEX_URL" \
        -c configs/robotwin2_ft/runtime_constraints.txt setuptools-scm ninja
      CPATH="$CUDA_PACKAGE_INCLUDE:$CUDA_RUNTIME_INCLUDE:$CUDA_HOME/include${CPATH:+:$CPATH}" \
      LIBRARY_PATH="$CUDA_LOCAL_LIB:$CUDA_HOME/lib${LIBRARY_PATH:+:$LIBRARY_PATH}" \
      LDFLAGS="-L$CUDA_LOCAL_LIB -L$CUDA_HOME/lib${LDFLAGS:+ $LDFLAGS}" \
      MAX_JOBS="${MAX_JOBS:-2}" "$PYTHON" -m pip install --index-url "$BOOTSTRAP_PIP_INDEX_URL" \
        -c configs/robotwin2_ft/runtime_constraints.txt --no-build-isolation -e third_party/curobo
    fi
  fi
  "$PYTHON" -m pip check
fi

if [[ "$DOWNLOAD_MODEL" -eq 1 ]]; then
  "$PYTHON" - "$MODEL_ID" "$MODEL_REVISION" "$PROJECT_ROOT/models/X-VLA-Pt" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_id, revision, target = sys.argv[1:]
Path(target).mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_id, revision=revision, local_dir=target)
print(f"Downloaded {repo_id}@{revision} to {target}")
PY
fi

if [[ "$DOWNLOAD_DATA" -eq 1 ]]; then
  "$PYTHON" scripts/download_robotwin2_assets_data.py \
    --output-root "$PROJECT_ROOT" --workers "${DOWNLOAD_WORKERS:-4}" \
    --revision "$DATA_REVISION"
  unzip -q -o assets/embodiments.zip -d assets/robotwin
fi

if [[ ! -e "$ROBOTWIN_ROOT/assets/embodiments" && -d "$PROJECT_ROOT/assets/robotwin/embodiments" ]]; then
  ln -s ../../../assets/robotwin/embodiments "$ROBOTWIN_ROOT/assets/embodiments"
fi

# The downloaded embodiment bundle stores cuRobo templates with a placeholder
# because the absolute checkout path differs between machines.  Materialize
# the runtime files in the project-owned asset bundle; RoboTwin sees them via
# the symlink above and the checkout itself remains otherwise untouched.
if [[ -d "$PROJECT_ROOT/assets/robotwin/embodiments" ]]; then
  for embodiment in aloha-agilex ARX-X5 piper; do
    source_dir="$PROJECT_ROOT/assets/robotwin/embodiments/$embodiment"
    [[ -d "$source_dir" ]] || continue
    for template in "$source_dir"/*_tmp.yml; do
      [[ -f "$template" ]] || continue
      target="${template%_tmp.yml}.yml"
      sed "s#\${ASSETS_PATH}#$ROBOTWIN_ROOT#g" "$template" > "$target"
    done
  done
fi

if [[ "$PREPROCESS" -eq 1 ]]; then
  "$PYTHON" scripts/audit_robotwin2_assets.py
  "$PYTHON" scripts/preprocess_robotwin2.py \
    --archive-root data/raw/archives \
    --output-root data/processed/robotwin2_ft \
    --manifest-root outputs/robotwin_ft/manifests/official_clean_50 \
    --base-config configs/robotwin2_ft/base_poses.json
  "$PYTHON" scripts/audit_robotwin2_preprocessed.py
fi

if [[ "$DOWNLOAD_MODEL" -eq 1 && "$PREPROCESS" -eq 1 ]]; then
  "$PYTHON" scripts/verify_robotwin2_bundle.py
fi

mkdir -p outputs/robotwin_ft
cat > outputs/robotwin_ft/bootstrap_manifest.json <<EOF
{
  "created_at_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "robotwin_root": "third_party/RoboTwin",
  "robowin_commit": "$actual_robotwin",
  "xpolicylab_commit": "$actual_xpolicylab",
  "model_id": "$MODEL_ID",
  "model_revision": "$MODEL_REVISION",
  "data_revision": "$DATA_REVISION"
}
EOF
echo "RobotWin2 bootstrap completed."
