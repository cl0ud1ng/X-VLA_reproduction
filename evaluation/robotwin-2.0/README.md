# 🧪 Evaluation on RoboTwin-2.0

We evaluate **X-VLA** on the **RoboTwin-2.0** benchmark to assess its ability to handle bimanual tabletop manipulation with multiple object sets, shifting layouts, and varied visual randomness.

---

## 1️⃣ Environment Setup

Prepare the pinned RoboTwin checkout with `bash scripts/bootstrap_robotwin2.sh
--with-rollout`.  The checkout is expected at `third_party/RoboTwin`.

No additional modifications are required for X-VLA evaluation.

---

## 2️⃣ Start the X-VLA Server

Run the X-VLA model as an inference server (in a clean environment to avoid dependency conflicts):

```bash
conda activate X-VLA
python -m deploy --model_path 2toINF/X-VLA-RoboTwin2
```
---

## 3️⃣ Run the Client Evaluation
Launch the RoboTwin-2.0 evaluation client to connect to your X-VLA server:

```bash
cd evaluation/robotwin-2.0
bash eval_robotwin.sh
```
You can configure custome evaluation in `eval_robotwin.sh`, such as log directry, server port number, number of episodes evaluated, task config, etc.

The client will stream observations (images, proprioception, and language) to the X-VLA model, receive predicted actions, and execute them within the RoboTwin-2.0 environment.

## 8-GPU rollout

The reproducible stage-D smoke/benchmark launcher covers the five frozen tasks
on all three domains and keeps one simulator process per GPU:

```bash
bash scripts/run_robotwin2_rollout_8GPU_exec1.sh \
  --host 127.0.0.1 --port 8000 \
  --num-episodes 1
```

Set `NUM_EPISODES`, `ROLLOUT_SEED`, `TASK_CONFIG`, `MODEL_HOST`, `MODEL_PORT`,
and `EVAL_LOG_DIR` through the environment when a different run is required.
The Python scheduler `scripts/run_robotwin2_rollout.py` accepts `--gpus`,
`--num-gpus`, `--tasks`, `--num-tasks`, `--domains`, and `--num-domains` for
smaller matrices or other hardware. The shell wrapper fixes the main protocol
to all eight GPUs, all three domains, all five tasks, `exec_points=1`, and
video capture. `exec_points=1` is the required receding-horizon protocol;
each simulator control step requests a fresh `[30,20]` prediction. Results are
written under `outputs/robotwin_ft/eval_exec1/`, with per-cell logs and summaries
and a top-level `run.json` recording commits, assignments, and conventions.

For the diagnostic `exec_points=10` comparison, use
`scripts/run_robotwin2_rollout_8GPU_exec10.sh`; its default output directory is
`outputs/robotwin_ft/eval_exec10/`. For either wrapper, `--output-dir` overrides
`EVAL_LOG_DIR`, which overrides the wrapper default.

---

## 📊 Results (Using RoboTwin-2.0 Leaderboard Settings)

|    **Settings**   |   Easy  |   Hard  |
| :--------------------: | :--: | :--: |
|     **Success (%)**    | 70.0 | 39.0 |



