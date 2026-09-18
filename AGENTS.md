# AGENTS.md

## 项目目的

本项目是 X-VLA 的复现/fork，用于在微调阶段复用 X-VLA 的跨数据集混训能力，开展 RoboTwin 2.0 三具身跨域混训实验。项目的主线目标是：

- 使用 `aloha-agilex`、`ARX-X5` 和双 Piper 三种同形态、同自由度的双臂具身；
- 使用三种具身共同提供的五个官方 clean demo 任务；
- 将官方 RoboTwin 数据转换为统一的三相机、proprio、语言和 absolute EEF6D 输入；
- 使用 domain-balanced 和 tempered sampling 进行 X-VLA 微调；
- 通过 RoboTwin 仿真 rollout 评估模型能力。

训练数据只用于训练，最终能力以 RoboTwin 仿真测试为准。外部 RoboTwin 项目是运行时依赖和数据/环境来源，不是本项目的代码归属范围。

## 必读文档

开始修改代码、生成数据或运行实验前，必须先阅读：

- [`docs/experiment_design.md`](docs/experiment_design.md)：本项目的唯一实验实施规范，定义具身、任务、数据 schema、坐标系、1 秒/30 点 action window、采样策略、训练配置、评测协议和验收门槛。
- [`README.md`](README.md)：X-VLA fork 的基本安装、模型和运行说明。
- [`evaluation/robotwin-2.0/README.md`](evaluation/robotwin-2.0/README.md)：当前 RoboTwin 评测入口和调用方式。

如果现有代码、脚本或默认配置与 `docs/experiment_design.md` 冲突，以实验设计文档为准。不得根据 RoboTwin 工作区中未提交、未跟踪或本地修改的文件推断官方接口；这类文件只能作为外部运行环境中的临时内容，不能成为本项目实现依据。

## 文件和产物归属

1. 所有代码和产物都必须放在本项目目录 `/mnt/mnt/data/zxw/cross-embodiment_generalization/X-VLA_reproduction` 下。
   - 包括 Python 代码、配置、转换脚本、manifest、缓存索引、训练日志、checkpoint、评测结果、可视化和临时实验报告。
   - 新增文件应放在已有职责明确的目录中；没有合适目录时再创建最小范围的新目录。
2. 外部资产仅供调用，不在外部目录写入、修改或生成项目产物。
   - RoboTwin 源码、仿真资产、官方数据集和基础 checkpoint 可以从外部路径读取或调用。
   - 不要把本项目的补丁、生成配置、转换结果、日志或 checkpoint 写回 RoboTwin 目录或其他外部项目目录。
   - 需要记录外部依赖时，在本项目内保存路径、版本/commit、文件 hash 和调用参数。
3. 使用绝对路径时，应把它作为运行时输入或配置项记录在本项目内；不要把外部路径硬编码成产物输出位置。
4. 不要覆盖用户已有的修改、未跟踪文件或外部资产。涉及同一路径时，先确认文件归属和当前状态。

## 开发原则

### 正向开发

- 按实验设计文档直接实现当前目标路径，优先完成清晰、可验证的主流程。
- 采用明确的数据契约和边界断言；遇到 schema、shape、坐标系、版本或配置不匹配时尽早报错。
- 不要为了假设中的未来格式加入大范围兼容层、静默回退、自动猜测字段或隐式修复数据。
- 不要用默认值掩盖拼写错误、缺失 domain、缺失相机、错误四元数顺序或不完整 action window。
- 只有在实验设计明确要求时才增加 fallback 或 ablation，并将其与主流程隔离、命名和记录。

### 代码与实验一致性

- 新代码必须保持 domain id、任务集合、相机顺序、absolute EEF6D、robot-base frame、scalar-first quaternion 和 `qdur=1.0 s`/30 点等冻结约定。
- 训练 sampler、数据转换器和评测 client 的输入输出应能由 manifest 和运行记录复现。
- 修改已有模块时，先确认是否会影响其他数据集或评测；RoboTwin 专用逻辑优先放入明确命名的专用 handler/config/client，避免改变旧数据集语义。
- 运行实验前执行与改动范围相称的检查；文档、配置和代码中的关键常量保持一致。

### 变更和验证

- 小而聚焦地修改文件，避免无关重构。
- 新增行为应至少提供一个可重复的 smoke test、schema audit 或命令行验证；纯文档变更至少检查链接、路径和关键参数的一致性。
- 不要把本地生成的缓存、日志、checkpoint 或大数据文件误当作源代码提交；必要时在 `.gitignore` 或文档中明确其位置。
- 报告结果时说明使用的 X-VLA commit、RoboTwin commit、manifest、domain/task、采样策略、seed 和输出路径。

## 外部 RoboTwin 使用边界

外部 RoboTwin 路径当前约定为：

```text
/mnt/mnt/data/zxw/cross-embodiment_generalization/model_test/RoboTwin
```

它只作为官方数据、仿真环境和运行时资产的来源。实现时应遵循 `docs/experiment_design.md` 中固定的官方提交、官方 HDF5/XPolicyLab schema 和五个任务矩阵；不要引用该工作区中为其他项目创建的未跟踪或未提交代码、配置和脚本。

## 完成标准

一项改动只有在以下条件都满足时才算完成：

- 代码/配置/产物位于本项目目录内，外部目录保持只读调用；
- 行为符合 `docs/experiment_design.md`，没有静默改变实验协议；
- 关键路径已进行适当验证，失败原因可以定位；
- 运行所需的外部资产、版本、命令和输出位置已记录，其他 agent 可以按文档复现。
