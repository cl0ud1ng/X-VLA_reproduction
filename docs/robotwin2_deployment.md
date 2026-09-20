# RoboTwin 2.0 训练与 Rollout 部署流程

本项目通过 Git remote 同步代码，通过 bootstrap 脚本在每台机器上恢复模型、训练数据、Python 环境和 RoboTwin 运行时。大文件不进入 Git，也不使用 Git LFS。

## 1. 主机要求

- Linux、Git 和网络访问；
- `uv`（用于在项目内下载 Python 3.10.19），或显式提供 Python 3.10；
- NVIDIA 驱动、CUDA 和足够的 GPU 显存；
- 如果要运行仿真 rollout，还需要 RoboTwin 所需的图形和仿真运行库。

默认由 uv 将 Python 3.10.19 安装到 `.python/`，再创建 `.venv/`。如果使用 `BOOTSTRAP_PYTHON`，该解释器必须是 Python 3.10，且不能在创建环境后被删除。系统 NVIDIA 驱动、Vulkan/EGL 和 CUDA 编译工具仍是主机依赖，不通过 Git 分发。

## 2. 获取项目和 RoboTwin

```bash
git clone <project-remote> X-VLA_reproduction
cd X-VLA_reproduction

git clone --recurse-submodules \
  https://github.com/RoboTwin-Platform/RoboTwin.git \
  third_party/RoboTwin
git -C third_party/RoboTwin checkout --detach 96c1fea
git -C third_party/RoboTwin submodule update --init --checkout
git -C third_party/RoboTwin/XPolicyLab \
  checkout c37109c500be67d0dea6b36bf7337bbd26e763cd
```

`third_party/RoboTwin` 已加入 `.gitignore`，但必须存在于需要预处理或 rollout 的机器上。项目不再兼容旧的 `/model_test/RoboTwin` 路径。

## 3. 一键准备环境、模型和数据

完整准备命令如下：

```bash
bash scripts/bootstrap_robotwin2.sh
```

该命令会：

- 下载项目内 `.python/` 并创建 `.venv`；
- 安装训练依赖和 PyTorch；
- 下载 `2toINF/X-VLA-Pt` 基础模型到 `models/X-VLA-Pt/`；
- 下载官方 RoboTwin clean archive 和具身包；
- 生成 `data/processed/robotwin2_ft/`；
- 生成 `outputs/robotwin_ft/manifests/official_clean_50/`；
- 执行资产、数据和跨机器路径审计。

需要同时准备仿真 rollout 依赖时使用：

```bash
CUDA_HOME=/path/to/cuda-12.1 bash scripts/bootstrap_robotwin2.sh --with-rollout
```

如果模型、数据或 normalized HDF5 已经准备好，可以跳过对应步骤：

```bash
bash scripts/bootstrap_robotwin2.sh \
  --skip-model \
  --skip-data \
  --skip-preprocess \
  --with-rollout
```

上述命令需要先设置 `CUDA_HOME`。它会安装 `third_party/curobo` 的官方 v0.7.8，并编译 CUDA 扩展；工具链必须同时包含 nvcc、CUDA runtime headers 和开发链接库。

如果 PyPI 镜像不可用，可指定镜像：

```bash
BOOTSTRAP_PIP_INDEX_URL=https://pypi.org/simple \
BOOTSTRAP_PYTHON=/path/to/python3.10 \
CUDA_HOME=/path/to/cuda-12.1 bash scripts/bootstrap_robotwin2.sh --with-rollout
```

脚本会把项目内的 `assets/robotwin/embodiments/` 自动链接到
`third_party/RoboTwin/assets/embodiments`。RoboTwin 的场景物体、文件和背景纹理必须存在于 `third_party/RoboTwin/assets/`。

## 4. 训练

训练脚本默认使用项目内的 `.venv`，不读取其他机器的环境：

```bash
bash scripts/run_robotwin2_training_8GPU.sh
```

训练数据来自项目内的 normalized HDF5 和 manifest。manifest 使用仓库相对路径，因此项目根目录可以位于不同机器的不同绝对路径。

训练前可以单独执行：

```bash
.venv/bin/python scripts/verify_robotwin2_bundle.py
```

该检查确认 15 个 domain-task pair、窗口数量、模型必需文件和路径可用。当前冻结数据包含 750 episodes、149,265 个窗口。正式长跑前执行：

```bash
.venv/bin/python scripts/preflight_robotwin2_ft.py
bash scripts/run_robotwin2_fsdp_batch256_smoke_8GPU.sh
# 当前训练脚本默认上报 W&B，首次在线训练前登录
.venv/bin/wandb login
```

## 5. Rollout

本次改动准备的是仿真运行环境和路径。`evaluation/robotwin-2.0/client.py` 仍是旧版参考 client（包含旧观测/动作约定），不能把它的结果当作本实验的正式三具身评测。阶段 D 的 base/world 转换、scalar-first、三相机和 receding-horizon client 尚需按实验设计验收。Stage A 的官方轨迹 replay 不等于模型 rollout。

Stage A 预检可以先跑一个单元：

```bash
.venv/bin/python scripts/robotwin2_sim_preflight.py \
  --tasks beat_block_hammer \
  --domains aloha-agilex \
  --result-root outputs/robotwin_ft/local_environment_preflight
```

正式评测前应完成三个具身 × 五个任务的完整预检，并确认结果写入 `outputs/robotwin_ft/preflight_stage_a/summary.json`。

## 6. Git 同步边界

Git remote 只同步代码、配置、脚本和文档。以下内容由 bootstrap 在每台机器重新生成或下载：

```text
.venv/
.python/
third_party/curobo/
third_party/RoboTwin/
models/X-VLA-Pt/
assets/embodiments.zip
assets/robotwin/
data/raw/
data/processed/
outputs/robotwin_ft/
```

模型 revision 固定为 `c1c4a64a7e03ac5b95c468bf1578f3d03651b53b`，数据 revision 固定为 `981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042`。官方训练 archive/具身包的大小和 SHA-256 在 `configs/robotwin2_ft/assets_lock.json` 中校验；环境来源与版本另记录到 `outputs/robotwin_ft/`。不要把 `.venv`、RoboTwin 工作区、训练日志或大模型权重强行加入 Git。
