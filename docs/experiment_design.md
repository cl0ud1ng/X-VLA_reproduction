# RoboTwin 2.0 三具身跨域混训微调实验设计

本文是本项目（`X-VLA_reproduction`）在 RoboTwin 2.0 上进行**微调阶段跨具身混训**的唯一实施规范。后续实现、数据转换、训练和仿真评测都必须以本文为准；如果代码现状与本文冲突，以本文的“冻结规范”和“验收标准”为准。

本文对应的代码和官方仓库快照：

- X-VLA fork：`/mnt/mnt/data/zxw/cross-embodiment_generalization/X-VLA_reproduction`，当前基线提交 `d1c031d`。
- RoboTwin 2.0：`/mnt/mnt/data/zxw/cross-embodiment_generalization/model_test/RoboTwin`，官方代码快照 `96c1fea`；XPolicyLab 官方子模块固定在 `c37109c`。
- 官方依据仅限上述提交中由 Git 跟踪的文件。RoboTwin 主仓库或 XPolicyLab 子模块工作区中的未跟踪、未提交和本地修改文件均不得作为设计或实现依据。
- 官方数据说明、HDF5 格式和图像解码说明来自已跟踪的 RoboTwin `README.md`、`data/decode_image_bit.py`、`envs/utils/pkl2hdf5.py`、`envs/`、明确列出的 `env_cfg` 官方文件、XPolicyLab 固定提交及官方数据仓库 `TianxingChen/RoboTwin2.0`。

## 1. 实验问题与不可改变的结论

目标是从 X-VLA 预训练 checkpoint 出发，在 RoboTwin 2.0 的三个**同形态、同自由度**双臂具身上，用相同的五个任务进行联合微调；训练数据只用于训练，最终能力只由 RoboTwin 仿真 rollout 衡量。

下列约束必须保持不变：

| 项目 | 冻结规范 |
|---|---|
| 具身 | 三个双臂、每臂 6-DoF + 1-DoF gripper：`aloha-agilex`、`ARX-X5`、双 Piper（左右臂均使用 `piper` 配置） |
| domain id | `aloha-agilex=0`、`ARX-X5=1`、`piper-dual=2`；训练和推理完全一致 |
| 任务 | `beat_block_hammer`、`stack_blocks_two`、`move_can_pot`、`open_microwave`、`place_dual_shoes` |
| 观测 | 1 个主相机 `head_camera` + 左腕 `left_camera` + 右腕 `right_camera`；RGB、当前 proprio、语言 |
| 图像 | 所有相机都使用官方 D435 配置，训练输入统一为 `224x224`，保持 X-VLA 当前 ImageNet 归一化和训练期 ColorJitter |
| 动作 | absolute EEF6D；左右臂各 10 维，拼成 20 维；位置和旋转均为 robot-base frame；gripper 统一为 `0=open, 1=closed` |
| 时间 | 每个样本的 future horizon 固定为 `qdur=1.0 s`、动作固定 30 点；训练数据限定为官方筛选的成功 clean demo，尾部不足 1 秒的窗口沿用 X-VLA 官方末帧 clamp 作为 terminal-hold supervision；不得跨 episode 填充 |
| 采样 | 以 action-observation window 数量统计 `N_d`；主结果跑 `domain-balanced`，并跑 `tempered(T=2)` 对照；五个任务在每个 domain 内等概率 |
| 评测 | RoboTwin 仿真，不用训练集 action loss 作为最终指标；每个具身 × 任务使用完全相同的评测协议和 seed 列表 |

### 1.1 关于第三个具身的硬性前置条件

官方数据仓库同时提供 `aloha-agilex`、`ARX-X5` 和 `piper` 的 clean archive；因此主实验第三个 domain 使用双 Piper。运行仿真前必须安装官方 Piper asset，并确认左右臂均为 6-DoF、任务配置确实创建双臂。

严禁用 `franka-panda` 代替第三个 domain：Franka 是 7-DoF，不满足本实验的同自由度要求。也严禁把单臂 Piper 当作第三个 domain。

双 Piper 的任务配置必须显式写成左右两个 Piper（形如 `[piper, piper, <官方验证的臂间距>]`），并为其增加对应的 `env_cfg` action profile（两臂均为 6+1）。`<官方验证的臂间距>` 必须从官方资产/任务配置或一次成功的仿真 preflight 得到，不允许凭经验猜测。若使用当前官方脚本的“单项 embodiment”语义，应确认它确实创建了双臂，而不是把同一模型误当单臂。

## 2. 对现有 X-VLA 代码的审查结论

### 2.1 可以复用的部分

- `models/configuration_xvla.py` 已将 `num_actions=30`、`action_mode="ee6d"`、`use_proprio=true` 固化在本地 `models/X-VLA-Pt/config.json` 中。
- `models/action_hub.py` 的 `EE6DActionSpace` 已提供 20 维布局、位置/旋转/gripper 分项 loss 和 gripper sigmoid 后处理。
- `models/modeling_xvla.py` 已将 `domain_id` 传给 soft prompt、domain-aware projection 和 action head；模型配置已有 `num_domains=30`，足够容纳本实验的 0/1/2。
- `datasets/dataset.py` 的 handler 接口、`action_slice` 的“首帧 proprio、后续帧 action”约定、`train.py` 的分组 optimizer 和冻结阶段可以作为实现起点。
- `datasets/dataset.py` 的图像尺寸、归一化参数和 `XVLAProcessor.encode_language` 的 50 token 上限沿用，不得为了 RoboTwin 单独改变。

### 2.2 必须修改/不能直接复用的部分

1. **现有 RoboTwin handler 不能直接消费官方 XPolicyLab-format episode。** `datasets/domain_handler/simulations.py:150` 虽然同样使用 1 秒 horizon，但假设旧的 `/endpose/...` 和 `/observation/...` 布局、固定约 30 Hz，且没有处理官方 `state/action/vision` 层次、文件内 frequency、encoded image bits 和 robot-base frame。应新增专用 `robotwin2_ft` handler，不要覆盖旧 handler 的语义。新 handler 必须沿用 X-VLA 官方的末帧 clamp 语义，但禁止跨 episode 填充，并在 manifest 中记录 `terminal_hold`。
2. **四元数顺序必须修正。** RoboTwin 使用 transforms3d 的 scalar-first `[w,x,y,z]`（官方 `robot.py` 调用 `t3d.quaternions.mat2quat/quat2mat`）；SciPy 默认是 scalar-last。读取 RoboTwin pose 时必须调用 `quat_to_rotate6d(q, scalar_first=True)`；推理输出转回 RoboTwin 时也必须使用 `rotate6d_to_quat(v6, scalar_first=True)`。当前 `datasets/utils.py` 和 `evaluation/robotwin-2.0/client.py` 的默认值为 `False`，不能用于本实验。
3. **官方图像只能用 `decode_image_bit`。** `datasets/utils.py:45` 的 `cv2.imdecode` 路径没有保证 RoboTwin 的 RGB 约定，且当前 handler 不识别嵌套 camera key。必须复用/导入 RoboTwin 官方 `XPolicyLab.utils.process_data.decode_image_bit`（或本地等价副本 `RoboTwin/data/decode_image_bit.py`），输出已经是 RGB，禁止再做 `COLOR_BGR2RGB`。
4. **不能夹带 delta action。** `action_slice` 只有在 handler 返回 `idx_for_delta` 时才做相减。RoboTwin 分支必须不返回该字段；`proprio` 和 30 个 action 都是同一 robot-base frame 下的绝对 EEF6D。
5. **当前采样权重不符合实验要求。** `datasets/dataset.py:111-122` 只按静态 `DATA_WEIGHTS` 选择 dataset，且 handler 递归生成器会重复遍历，既不是按 action-observation 数量计权，也没有 domain/task 统计。必须改成 manifest 驱动的可审计 sampler（见第 5 节）；terminal-hold 窗口计入 action-observation 数量，并单独记录数量与比例。
6. **当前 DataLoader/分布式路径不安全。** `datasets/__init__.py` 固定 4 worker，`train.py:225` 直接调用 `.cuda()`，且注释明确不 prepare iterable dataloader；多 GPU 时会重复采样或产生不一致频率。新的 sampler 必须 worker/rank aware，训练代码使用 accelerator device，不得硬编码 CUDA。
7. **当前 RoboTwin client 的 gripper 和坐标约定不一致。** `evaluation/robotwin-2.0/client.py` 将 gripper 映射为 `1-2*g`，并把 world-frame pose 直接送进模型；这与本实验的 `0/1` gripper 和 robot-base frame 不兼容。该 client 只能作为通信骨架，必须按第 7 节重写。
8. **当前 action-space 的 loss 假设目标 gripper 是 0/1。** `EE6DActionSpace` 对 gripper 使用 `BCEWithLogitsLoss`；任何 `-1/1` 或连续未归一化 gripper 都是错误数据，应在 handler 入口断言并转换。

### 2.3 X-VLA 的 absolute/relative 语义

X-VLA 模型本身不内置“把相对位移加到当前 proprio”或“把绝对 pose 转成 delta”的逻辑。`XVLA.forward()`/`generate_actions()` 只对输入 action space 做预处理、去噪和 loss/postprocess；是否为 absolute 或 relative 由数据 handler 是否在 `action_slice()` 前对指定维度减去当前 proprio 决定：

- handler 不返回 `idx_for_delta`：模型学习并输出 absolute EEF6D target；
- handler 返回 `idx_for_delta`：对应维度的训练 target 是相对当前 proprio 的 delta，部署端必须再加回当前状态。

官方 README 也将 EE6D action 描述为 “target delta/absolute pose”，说明这是数据/下游控制器约定，而不是模型架构的固定属性。[官方 X-VLA README](https://github.com/2toinf/X-VLA/blob/main/README.md) 的 LIBERO 说明则明确要求先将原始 relative EEF action replay 成 absolute EEF pose，再用于 absolute-action pipeline。[LIBERO absolute-action 预处理说明](https://github.com/2toinf/X-VLA/blob/main/evaluation/libero/preprocess.md)

本实验选择 absolute EEF6D：RoboTwin handler 不得设置 `idx_for_delta`，训练 label 和推理输出都定义为当前 robot-base frame 下的绝对目标 pose。评测 client 必须把该 base-frame target 变换回 RoboTwin world frame 后直接交给 `take_action(action_type='ee')`，不能再次加当前 proprio。

### 2.4 预训练阶段到底使用哪种 pose

对本项目当前 `X-VLA-Pt` 的预训练混合数据，结论是：**预训练的 EEF6D target 按 absolute pose 使用**，不是统一的 delta pose。判断依据不是 `action_mode="ee6d"` 这一项本身，而是预训练数据对应的 handler：`robomind-*`、`Droid-*` 和 `AGIBOT` 都直接把数据中的 EEF position/orientation 转成 `[xyz, Rotate6D, gripper]`，没有返回 `idx_for_delta`；因此 `action_slice()` 不会对这些维度减去当前 proprio。

仓库中仍存在 `LeRobotV21`、`X2Robot` 等带 `idx_for_delta` 的 handler。它们是**按 handler 选择的相对量例外**，不能反推为 X-VLA 的统一预训练语义；如果后续把这些数据加入训练混合，必须在 manifest 中显式记录 action representation，并单独做转换/审计。另一个独立问题是：上述 absolute pose 仍可能处于各数据集自己的坐标系，并不自动等于 RoboTwin 的 robot-base frame；本实验必须在 `robotwin2_ft` handler 中显式做坐标变换。

## 3. 具身、任务和数据来源

### 3.1 具身 domain 表

| domain id | RoboTwin 名称 | 双臂配置 | 每臂 arm dim | gripper dim | 当前状态 |
|---:|---|---|---:|---:|---|
| 0 | `aloha-agilex` | `[aloha-agilex]`（同一模型用于左右臂） | 6 | 1 | 官方 archive：`aloha-agilex_clean_50.zip` |
| 1 | `ARX-X5` | `[ARX-X5, ARX-X5, distance]` | 6 | 1 | 官方 archive：`arx-x5_clean_50.zip` |
| 2 | `piper-dual` | `[piper, piper, distance]` | 6 | 1 | 官方 archive：`piper_clean_50.zip` |

`domain_id` 是模型的 embodiment condition，不是 task id。任务名只能作为语言和 manifest 元数据，不能替代 domain id。

### 3.2 五个任务

任务集合取自官方数据仓库中同时存在以下三个 clean archive 的交集：

```text
dataset/<task>/aloha-agilex_clean_50.zip
dataset/<task>/arx-x5_clean_50.zip
dataset/<task>/piper_clean_50.zip
```

本实验固定使用以下五项：

1. `beat_block_hammer`
2. `stack_blocks_two`
3. `move_can_pot`
4. `open_microwave`
5. `place_dual_shoes`

它们分别覆盖接触/敲击、堆叠、搬运、铰链操作和双臂协同。每个 domain 必须使用这五项的对应官方 clean archive；不能使用只在部分具身存在的任务，不能删除任务，也不能使用非官方 task config 推断任务覆盖范围。

官方 archive 路径矩阵固定为：

| task | Aloha | ARX-X5 | Piper |
|---|---|---|---|
| `beat_block_hammer` | `dataset/beat_block_hammer/aloha-agilex_clean_50.zip` | `dataset/beat_block_hammer/arx-x5_clean_50.zip` | `dataset/beat_block_hammer/piper_clean_50.zip` |
| `stack_blocks_two` | `dataset/stack_blocks_two/aloha-agilex_clean_50.zip` | `dataset/stack_blocks_two/arx-x5_clean_50.zip` | `dataset/stack_blocks_two/piper_clean_50.zip` |
| `move_can_pot` | `dataset/move_can_pot/aloha-agilex_clean_50.zip` | `dataset/move_can_pot/arx-x5_clean_50.zip` | `dataset/move_can_pot/piper_clean_50.zip` |
| `open_microwave` | `dataset/open_microwave/aloha-agilex_clean_50.zip` | `dataset/open_microwave/arx-x5_clean_50.zip` | `dataset/open_microwave/piper_clean_50.zip` |
| `place_dual_shoes` | `dataset/place_dual_shoes/aloha-agilex_clean_50.zip` | `dataset/place_dual_shoes/arx-x5_clean_50.zip` | `dataset/place_dual_shoes/piper_clean_50.zip` |

### 3.3 官方数据优先顺序

每个 `domain × task` 按下列顺序准备训练数据：

1. 使用官方 `TianxingChen/RoboTwin2.0` 对应的 `*_clean_50.zip`，保留原始 episode id、instruction 和 archive 路径。
2. 只有在官方 clean 数据量不足以满足既定训练预算时，才补充 RoboTwin 官方流程采集的数据；补充数据沿用同一任务、具身、相机、动作和时间规范，并在 manifest 标记 `source=collected`。

每个官方 `*_clean_50.zip` 的 50 个成功 episode 全部作为候选训练数据；采样器按 window 数量而不是 episode 数量工作。自采数据只作为数据量不足时的补充方案，不改变任务集合和实验主流程。

训练集和仿真测试集完全隔离：测试由 RoboTwin 按官方 seed 生成，训练 HDF5 中的 episode/seed 不得被用于测试 seed；不得从测试 rollout 反向加入训练。

## 4. 统一数据契约

### 4.1 官方 XPolicyLab-format HDF5

RoboTwin 官方跟踪代码 `envs/utils/pkl2hdf5.py` 将 collection 输出转换成 XPolicyLab v1.0 HDF5。handler 的主 schema 必须是：

```text
episode_XXXXXXX.hdf5
├── data_format_version                         "v1.0"
├── instructions                                JSON string/list
├── additional_info/frequency                   scalar source fps
├── state/
│   ├── left_ee_poses                           [T, 7] xyz+quat[w,x,y,z]
│   ├── right_ee_poses                          [T, 7]
│   ├── left_ee_joint_states                    [T, 1]
│   ├── right_ee_joint_states                   [T, 1]
│   └── left/right_arm_joint_states             [T, 6]，仅审计
├── action/
│   ├── left_ee_poses                           [T, 7]，下一时刻绝对 pose
│   ├── right_ee_poses                          [T, 7]
│   ├── left_ee_joint_states                    [T, 1]
│   ├── right_ee_joint_states                   [T, 1]
│   └── left/right_arm_joint_states             [T, 6]，仅审计
└── vision/
    ├── cam_head/colors                         [T, encoded image bits]
    ├── cam_left_wrist/colors                   [T, encoded image bits]
    └── cam_right_wrist/colors                  [T, encoded image bits]
```

转换器必须先检查 `data_format_version`、frequency、所有时间长度和三路相机。官方 writer 的对齐关系是 `state=原轨迹[:-1]`、`action=原轨迹[1:]`、`vision=原图像[:-1]`。重建连续绝对 pose 序列时使用 `P[0]=state[0]`、`P[1:]=action[:]`，并断言 `state[1:]` 与 `action[:-1]` 在容差内一致；禁止再把 action 额外平移一帧。

训练语言从 HDF5 的 `instructions` 解析，使用列表中的 seen instruction；官方 collection writer 只将 seen instructions 写入该字段。若某个官方 archive 是较早 schema，必须先通过官方固定提交的 loader 归一化到上述字段，再进入本实验 handler；禁止依据本地非官方转换脚本猜测字段。

如果某个 archive 确实没有 `left/right_ee_poses`，不得把 joint action 冒充 EEF6D；必须用官方 RoboTwin 运动学生成 endpose，并在转换日志中记录方法和误差。

### 4.2 图像

- 读取 `vision/cam_head/colors`、`vision/cam_left_wrist/colors`、`vision/cam_right_wrist/colors`，按同一帧索引与 state pose 对齐。
- 对 encoded bytes 调用官方 `decode_image_bit`；输出已是 HWC、uint8、RGB。
- 三个 view 都必须存在，`image_mask=[True,True,True]`。缺失 view 不得用零图像掩盖；只有做专门的缺视角 ablation 时才允许改变 mask。
- 训练和推理的几何预处理均为 Resize 到 `(224,224)`、BICUBIC、转 tensor、Normalize `(0.485,0.456,0.406)/(0.229,0.224,0.225)`；只在训练启用当前代码的 brightness/contrast/saturation `0.2` ColorJitter。
- 不进行额外 center crop、BGR 交换、深度输入或第三视角输入。

### 4.3 Robot-base frame 的精确定义

为了避免不同 embodiment 的 world root pose 和臂间距进入动作目标，训练/推理使用每个 arm 自己的 RoboTwin robot-base frame：

1. 从 RoboTwin `left_entity_origion_pose` / `right_entity_origion_pose` 得到 `T_W_B^L`、`T_W_B^R`。这些是每条 rollout 的机器人 base 在 world frame 的刚体变换；如果 native HDF5 没有逐帧记录，必须从对应 task config/`scene_info.json` 恢复并把结果写入 manifest，不能使用一个未验证的常数。
2. 从连续 pose 序列 `state/action/{left,right}_ee_poses` 读取 `T_W_E`（RoboTwin 四元数为 scalar-first）。
3. 计算 `T_B_E = (T_W_B)^{-1} T_W_E`。左右臂分别使用自己的 base，但在所有三种具身中都使用同一数学定义。
4. 对 `T_B_E` 的旋转矩阵取前两列，按仓库现有约定交错展平为 6D：`[R00,R01,R10,R11,R20,R21]`。不要改成另一种 6D 排列。
5. 位置保持米制 float32；gripper 先裁剪到 `[0,1]`，再统一为 `g_closed = 1 - g_robtwin`（必须通过一条 episode 的开/闭状态审计确认官方 raw 语义）。

每一帧形成 20 维向量：

```text
[L_x,L_y,L_z,L_r00,L_r01,L_r10,L_r11,L_r20,L_r21,L_grip,
 R_x,R_y,R_z,R_r00,R_r01,R_r10,R_r11,R_r20,R_r21,R_grip]
```

proprio 是当前帧向量；action 是之后 30 个时间点的绝对向量。不得做 position delta、rotation delta、joint delta 或按 episode 的 min-max normalization。模型的现有 `EE6DActionSpace` 会在输入给 transformer 前屏蔽 proprio/action 的 gripper channel，并在输出端对 gripper logit 做 sigmoid；这不是数据格式变化。

### 4.4 时间与 30 点 action window

每个 episode 必须有可审计的时间轴：

- 若 HDF5 有 timestamp，优先使用 timestamp；否则使用 `additional_info/frequency` 和 frame index。不能把所有具身强制按 15 Hz 或 30 Hz 解释。
- 候选起点为 episode 内所有具有有效 current observation、三路图像和 instruction 的 `t0`；不要求 `t0 + 1.0 <= t_last`，因为成功 clean demo 的尾部窗口采用官方末帧 clamp。
- 查询时间为 `q = t0 + linspace(1/30, 1.0, 30)`；episode 结束后的查询时间统一 clamp 到 `t_last`，位置在 episode 内用线性插值，旋转使用 SciPy `Rotation/Slerp`（或数值等价的 SO(3) geodesic interpolation），不能逐元素直接插 quaternion，gripper 用最近邻或分段常值。
- 当查询时间超过 episode 末帧时，必须使用 X-VLA 官方的末帧 clamp 生成 terminal-hold action；clamp 只能发生在当前 episode 内，不能延伸到下一个 episode。
- 输出 `abs_trajectory` 为 `[31,20]`（首行 current proprio，后 30 行 action）；然后由 `action_slice` 取首行和后 30 行。
- 对非 `terminal_hold` 窗口，完全静止过滤仅用于统计/减少无效样本，阈值必须写入 manifest（建议 EEF position 最大变化 `<1e-5 m` 且 gripper 不变时过滤）；`terminal_hold` 窗口不得因静止而被过滤。

四元数/6D 必须有 round-trip 单测：`scalar_first quat -> 6D -> scalar_first quat` 的旋转矩阵误差 `<1e-5`；不要用当前 `quat_to_rotate6d` 的默认 scalar-last。

## 5. Manifest 与 domain-balanced/tempered sampler

### 5.1 推荐 manifest 结构

为每个 `domain × task` 生成一个 manifest（JSON 或 JSONL），再生成一个总 manifest。推荐字段如下，字段名不要随意改动：

```json
{
  "dataset_name": "robotwin2_ft_aloha",
  "robot_type": "robotwin2_ft_aloha",
  "domain_id": 0,
  "embodiment": "aloha-agilex",
  "task_names": ["beat_block_hammer", "stack_blocks_two", "move_can_pot", "open_microwave", "place_dual_shoes"],
  "source": "official",
  "fps_source": "additional_info/frequency",
  "qdur_sec": 1.0,
  "num_actions": 30,
  "camera_keys": ["vision/cam_head/colors", "vision/cam_left_wrist/colors", "vision/cam_right_wrist/colors"],
  "instruction_dir": "/abs/path/.../instruction",
  "datalist": ["/abs/path/.../episode_0000000.hdf5"],
  "episodes": 50,
  "num_observation_frames": 123456,
  "num_action_observation_windows": 98765,
  "num_terminal_hold_windows": 1234,
  "split": "train"
}
```

总 manifest 必须额外保存每个 `(domain,task)` 的 `N_dt` 和每个 domain 的 `N_d=sum_t N_dt`，以及最终采样概率、随机种子、git commit。训练只读取 `split=train`；若保留官方 validation episode，也不得让它进入训练 sampler。

### 5.2 两种必须报告的采样策略

令 `N_dt` 为应用官方末帧 clamp、并完成非 terminal-hold 静止过滤后的 action-observation window 数量；其中 `terminal_hold` 子计数必须单独记录。

**A. Domain-balanced（主结果）**

```text
p(domain=d) = 1/3
p(task=t | domain=d) = 1/5
p(d,t) = 1/15
```

这保证三个具身、五个任务贡献相同的训练机会，即使某个具身 episode 更长也不会吞掉其他 domain。

**B. Tempered（对照）**

使用温度 `T=2`，即指数 `alpha=1/T=0.5`：

```text
p(domain=d) = N_d^0.5 / sum_j N_j^0.5
p(task=t | domain=d) = N_dt^0.5 / sum_u N_du^0.5
p(d,t) = p(domain=d) * p(task=t | domain=d)
```

这就是“按 action-observation 数量计算、但不过度偏向大数据集”的 tempered sampling。不得使用当前 `DATA_WEIGHTS` 的手工常数替代上述公式。每个 sampler 初始化时打印 `N_dt`、`N_d`、`p(d,t)`；连续抽样至少 10,000 次后，经验频率与目标概率的最大绝对误差应小于 2%。

### 5.3 实现要求

建议新增 `RobotWin2FTHandler` 并将 dataset reader 改为 manifest/index 驱动：

- 初始化阶段只扫描 episode 和已缓存的 window index，不在 `__iter__` 中递归 `yield from`。
- 一个 index 至少包含 `hdf5_path、frame_idx、task、domain_id、instruction_path、timestamp`。
- 每个 DataLoader worker 使用 `base_seed + rank * 100003 + worker_id` 的独立 RNG；分布式 rank 之间不能取同一 index 流。
- 训练无限采样时只在 sampler 层无限循环，handler 本身应是有限 episode/window iterator。
- `domain_id` 从 manifest 读取并断言等于 0/1/2，禁止未知 domain 默认为 0（当前 `DATA_DOMAIN_ID.get(...,0)` 会掩盖拼写错误）。
- 每个 batch 必须允许混有不同 domain；不能按 batch 固定一个 domain。

## 6. 模型和微调配置

### 6.1 Checkpoint 与 action 配置

从本地 `models/X-VLA-Pt` 或等价的 `2toINF/X-VLA-Pt` 加载；启动前断言：

```text
config.action_mode == "ee6d"
config.num_actions == 30
config.use_proprio is True
config.num_domains >= 3
action_space.dim_action == 20
```

如果将 foundation checkpoint 换成其他 checkpoint，必须把 checkpoint 的 config、processor 和 tokenizer 一起归档；不能只替换权重文件。

### 6.2 主实验（full fine-tuning）

主结果使用 `train.py` 的 full fine-tuning 分组 optimizer，不使用 `peft_train.py` 作为主结果。建议的首个可复现实验配置如下；后续若修改，必须在 run manifest 中记录：

```text
batch_size       = 8                         # 显存允许时按 GPU 调整，但记录 global batch
learning_rate    = 1e-5
learning_coef    = 0.1                       # VLM/soft prompt 相对系数
weight_decay     = 0.0
betas            = (0.9, 0.95)
iters            = 30000
freeze_steps     = 1000                      # 仅 soft prompt/action head 先训练
warmup_steps     = 2000
use_cosine_decay = true
min_lr_ratio     = 0.1
max_grad_norm    = 1.0
seed              = 0,1,2                    # 至少三次独立 seed
```

`freeze_steps` 期间 VLM 和 transformer core 的学习率必须为 0；之后才按 schedule 打开。每个 checkpoint 保存 model、processor、`state.json`、总 manifest、git commit 和完整命令行。

训练入口应使用 accelerator 的 `device` 移动 tensor，并对 dataloader 做正确的 worker/rank 划分；禁止直接写 `.cuda()` 或因为“iterable 不 prepare”而跳过分布式一致性。

### 6.3 可选 LoRA 对照

若需要报告参数高效微调，单独使用 `peft_train.py`，不能和 full FT 的结果混称。LoRA checkpoint 必须同时保存 base checkpoint 标识、adapter config、processor 和相同的 sampler manifest；评测时先加载同一 base，再加载 adapter。

## 7. RoboTwin 评测协议

### 7.1 坐标和动作反变换

评测 client 每次收到 observation 后：

1. 取 `endpose` 的 world-frame EEF pose 和三路 RGB。
2. 用当前 embodiment 的 `T_W_B^L/R` 转成左右各自的 base-frame proprio，按第 4.3 节拼成 20 维。
3. 请求中发送 `domain_id=0/1/2`、语言指令、`image0=head`、`image1=left wrist`、`image2=right wrist`。
4. 收到 `[30,20]` absolute base-frame action 后，取执行点（主协议为每次只执行第 1 点的 receding horizon），把左右 pose 用 `T_W_E=T_W_B*T_B_E` 变回 world frame。
5. 6D 转四元数时使用 `scalar_first=True`；gripper 概率以 `>=0.5` 判为 closed，并转换到 RoboTwin `take_action(action_type='ee')` 所需的 raw gripper 约定。

主协议的 30 点是 1 秒预测 horizon，不是强制一次性执行 30 个 target。禁止沿用旧 client 中“请求一次、连续执行全部 30 点”的循环，除非明确作为 `exec_points=30` 的单独 ablation。主结果固定 `exec_points=1`，每个 simulator control step 重新请求一次模型。

RoboTwin 的 `take_action(action_type='ee')` 接受 world-frame target pose；不得把模型的 base-frame target 直接传给它。每个 domain 必须做一条“base -> world -> base” round-trip 检查，位置误差 `<1e-5 m`、旋转矩阵误差 `<1e-5`。

### 7.2 测试矩阵与报告

主评测矩阵为 `3 embodiments × 5 tasks = 15` 个单元。每个单元至少 20 个官方测试 episode；同一 checkpoint、同一 seed 列表、同一 instruction split 下完成。至少报告：

- 每个 `domain × task` 的 success rate、有效 episode 数、planner failure 数；
- 每个 domain 的五任务 macro mean；
- 全部 15 单元的 macro mean（不得按 episode 数加权成 micro mean）；
- clean/官方默认设置和 randomized/hard 设置分开列出；
- seen instruction 为主结果，unseen instruction 作为语言泛化对照；
- checkpoint、训练 sampler（balanced/tempered）、seed、RoboTwin commit、task config 路径。

## 8. 必须新增/修改的文件清单

实现 agent 应按以下顺序工作，避免把旧格式与新格式混在一起：

1. `datasets/domain_handler/robotwin2_ft.py`：官方 native HDF5、图像 bit 解码、scalar-first quaternion、base-frame 转换、1 秒/30 点窗口。
2. `datasets/domain_handler/registry.py`：注册三个明确的 `robotwin2_ft_*` handler 或一个读取 manifest domain 配置的专用 handler；不得用模糊名称匹配。
3. `datasets/domain_config.py`：只保存 domain id/默认采样模式等常量；删除/绕过针对本实验的手工 `DATA_WEIGHTS`。
4. `datasets/dataset.py`、`datasets/__init__.py`：manifest index、domain/task sampler、worker/rank seed、严格 schema 断言。
5. `datasets/utils.py`：补充官方 `decode_image_bit` 和 scalar-first 6D helper 的可测试实现；保留其他旧数据集功能。
6. `train.py`（必要时 `peft_train.py`）：accelerator device、dataloader 分片、运行配置保存、每步 domain/task 计数日志。
7. `evaluation/robotwin-2.0/client.py`：base/world 变换、gripper 0/1、scalar-first quaternion、receding-horizon 执行；不再复制旧 `1-2*g` 逻辑。
8. `configs/` 或 `docs/` 下的 manifest、具身 task config 和训练 shell：每个绝对路径、RoboTwin commit、domain id、task 列表都显式写出。

## 9. 分阶段执行与验收门槛

### 阶段 A：具身/仿真 preflight（无模型）

- [x] 三个具身的 asset、URDF、左右 arm dim 均为 `[6,6]`，gripper dim 为 `[1,1]`。
- [x] 五个 task 在三种具身上均能 `setup_demo`，并能成功跑至少一个官方 seed。
- [x] head/left/right 三路相机均存在，分辨率和帧数一致；HDF5 schema audit 通过。
- [x] 官方 `decode_image_bit` 输出 RGB；抽查一帧确认没有额外 BGR 交换。
- [x] pose quaternion 确认为 transforms3d scalar-first；6D round-trip 通过。

### 阶段 B：数据转换与 sampler preflight

- [x] 15 个 `domain × task` 都有 manifest。
- [ ] 每个样本是 `image_input [3,3,224,224]`、`image_mask [3]` 全 True、`proprio [20]`、`action [30,20]`、`domain_id∈{0,1,2}`。
- [x] 所有窗口 horizon 恰为 1.0 秒；尾部查询超出 episode 时仅使用当前 episode 的末帧 clamp，并已标记 `terminal_hold`；所有值 finite。
- [x] `N_dt/N_d`、balanced 和 tempered(T=2) 概率可复算；10,000 次抽样误差 <2%。
- [ ] 一个 batch 同时出现多个 domain，且三 domain 的计数与目标频率一致。

### 阶段 C：模型 smoke test

- [ ] 用每个 domain 各 2 个样本完成 forward/backward 100 steps，无 shape、NaN、BCE target 或 device 错误。
- [ ] 输出固定为 `[B,30,20]`；保存/重新加载 checkpoint 后 processor 和 domain embedding 均存在。
- [ ] 训练日志包含 `loss_position、loss_rotate6D、loss_gripper、domain_count[0..2]、task_count` 和四组 learning rate。

### 阶段 D：最小 rollout

- [ ] 15 个评测单元每个先跑 1 episode；检查 action base/world round-trip、gripper 开闭和相机顺序。
- [ ] 确认 client 每个 control step 都重新请求模型，且没有越过 RoboTwin `take_action` 的 pose frame 约定。
- [ ] 只有 smoke 全部通过后，才运行正式 20-episode × 15-cell × 3-seed 评测。

## 10. 结果目录和可追溯性

建议每个 run 使用如下目录，避免不同采样策略、具身和 seed 互相覆盖：

```text
outputs/robotwin_ft/
├── manifests/<manifest_id>/
├── domain_balanced/seed0/{ckpt-...,run.json,train.log,metrics.json}
├── domain_balanced/seed1/...
├── tempered_T2/seed0/...
└── eval/<checkpoint_tag>/<embodiment>/<task>/results.json
```

`run.json` 至少记录：X-VLA commit、RoboTwin commit、base checkpoint 路径/hash、所有 manifest 路径/hash、三 domain id、task 列表、qdur、num_actions、相机键、图像 decoder、四元数 convention、采样公式、temperature、batch/global batch、学习率、seed、完整启动命令。任何无法由这些文件复现的隐式默认值都视为实验不合格。
