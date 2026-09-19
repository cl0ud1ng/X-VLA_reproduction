# RoboTwin 2.0 数据与具身资产准备记录

本目录记录按 [`experiment_design.md`](experiment_design.md) 完成的第一阶段准备结果。外部 RoboTwin 只作为只读运行时依赖；所有下载、解压、normalized HDF5、窗口索引和审计结果均在本项目内。

## 来源与产物

- RoboTwin 源码固定提交：`96c1fea`
- XPolicyLab 子模块固定提交：`c37109c`
- 数据仓库：`TianxingChen/RoboTwin2.0@main`
- 下载记录、大小和 SHA-256：[`data/raw/download_manifest.json`](../data/raw/download_manifest.json)
- 官方具身包：[`assets/embodiments.zip`](../assets/embodiments.zip)
- 解压后的具身资产：[`assets/robotwin/embodiments/`](../assets/robotwin/embodiments/)
- 具身审计：[`outputs/robotwin_ft/assets_preflight.json`](../outputs/robotwin_ft/assets_preflight.json)
- normalized 数据：[`data/processed/robotwin2_ft/`](../data/processed/robotwin2_ft/)
- 总 manifest：[`outputs/robotwin_ft/manifests/official_clean_50/total.json`](../outputs/robotwin_ft/manifests/official_clean_50/total.json)

固定下载矩阵是 5 个任务 × 3 个 clean archive：`beat_block_hammer`、`stack_blocks_two`、`move_can_pot`、`open_microwave`、`place_dual_shoes`；具身为 `aloha-agilex`、`ARX-X5`、`piper`。每个 archive 保留 50 个官方 episode。

## 可重复命令

依赖使用外部 RoboTwin 虚拟环境中已安装的 `h5py/numpy/scipy/Pillow/PyYAML`：

```bash
python3 scripts/download_robotwin2_assets_data.py --output-root . --workers 4
unzip -q -o assets/embodiments.zip -d assets/robotwin
/mnt/mnt/data/zxw/cross-embodiment_generalization/model_test/RoboTwin/.venv/bin/python scripts/audit_robotwin2_assets.py
/mnt/mnt/data/zxw/cross-embodiment_generalization/model_test/RoboTwin/.venv/bin/python scripts/preprocess_robotwin2.py \
  --archive-root data/raw/archives \
  --output-root data/processed/robotwin2_ft \
  --manifest-root outputs/robotwin_ft/manifests/official_clean_50 \
  --base-config configs/robotwin2_ft/base_poses.json
/mnt/mnt/data/zxw/cross-embodiment_generalization/model_test/RoboTwin/.venv/bin/python scripts/audit_robotwin2_preprocessed.py
```

下载脚本可安全重跑；已有文件按大小存在检查跳过。预处理脚本从 archive 的 native HDF5 提取必需字段，按官方 `envs/utils/pkl2hdf5.py` 对齐规则生成 XPolicyLab v1.0：`state=原轨迹[:-1]`、`action=原轨迹[1:]`、`vision=原图像[:-1]`。图像仍保存为 encoded bits，读取时只能调用官方 `decode_image_bit` 或项目内等价实现 [`scripts/robotwin2_decode.py`](../scripts/robotwin2_decode.py)，不得再做 BGR/RGB 交换。

## 当前审计结果

- 具身：三套资产 URDF/网格存在；三 domain 均为两臂 `[6,6]` + gripper `[1,1]`。
- 数据：15 个 pair manifest、750 个 normalized episode、149,265 个窗口。
- 每个窗口：`qdur=1.0 s`、30 点 absolute EEF6D、三视角、有效 language、terminal-hold 只在当前 episode 末帧 clamp。
- 图像：全量检查输出 `(240,320,3)`、`uint8`、RGB。
- 旋转：RoboTwin scalar-first `[w,x,y,z]`，base/world round-trip 误差 `<1e-5`。
- 采样：domain-balanced 和 tempered `T=2` 均通过 10,000 次抽样，最大绝对误差分别为 `0.00553` 和 `0.00455`。
- 每个 domain 的窗口数：`aloha-agilex=57,292`、`ARX-X5=51,371`、`piper-dual=40,602`。

Stage A simulator preflight 已完成：固定 RoboTwin `96c1fea` 的代码通过临时 overlay 运行，三种具身×五个任务共 15/15 通过 `setup_demo`、官方 clean archive 的 `episode0` seed 轨迹 replay、`play_once` 和 `check_success`。结果与日志见 [`outputs/robotwin_ft/preflight_stage_a/summary.json`](../outputs/robotwin_ft/preflight_stage_a/summary.json)；runner 为 [`scripts/robotwin2_sim_preflight.py`](../scripts/robotwin2_sim_preflight.py)。Aloha、ARX-X5、双 Piper 的观测相机均包含 `head_camera`、`left_camera`、`right_camera`，三者 arm dim 均为 `[6,6]`，control gripper dim 为 `[1,1]`。

阶段 B 的 transformed-sample 与 sampler preflight 已完成：`scripts/preflight_robotwin2_ft.py` 使用总 manifest 和真实 HDF5 图像/姿态解码，逐 pair 检查了 90 个样本（包含每个 pair 的 terminal-hold），全部满足 `[3,3,224,224]`、全 True mask、`[20]` proprio、`[30,20]` action、finite 和 `[0,1]` gripper；一个 6-sample batch 同时包含三个 domain（各 2 个）。domain-balanced 与 tempered(T=2) 各抽样 10,000 次，最大绝对误差分别为 `0.0055333` 和 `0.0045544`。结果写入被忽略的运行产物 `outputs/robotwin_ft/preflight_stage_b.json`。训练端通过 `datasets/domain_handler/robotwin2_ft.py` 和 `ManifestWindowDataset` 读取相同窗口索引。

阶段 C 的 8×RTX 4090 training smoke 已完成：使用 Accelerate/DDP、global batch 8、fp16 和正式 `freeze_steps=1000` 的前 100-step 阶段（VLM/transformer core LR 为 0，仅更新 soft prompt/action head）。每个 domain 至少 2 个真实样本，100 个 forward/backward/update step 无 shape、NaN、BCE target 或 device 错误；总 loss 从 `254.5387` 降到 `234.6687`。推理输出为 `[8,30,20]`，保存后重新加载确认 processor 完整且 domain embedding 有 30 行。逐步日志和验收报告位于被忽略的 `outputs/robotwin_ft/stage_c_smoke/{metrics.jsonl,smoke_report.json,state.json}`。完成重载验证后已自动删除 3.3 GB smoke checkpoint，只保留约 36 KB 审计记录。正式 8 卡训练入口为 `scripts/run_robotwin2_training_8x4090.sh`。

`configs/robotwin2_ft/base_poses.json` 记录了三个 domain 的 robot-base pose；ARX-X5 和双 Piper 使用 `[robot, robot, 0.60]` 的双臂任务语义。正式模型仿真 rollout 尚未开始；后续仍需在模型 client action base/world round-trip 和 receding-horizon 协议下执行阶段 D。
