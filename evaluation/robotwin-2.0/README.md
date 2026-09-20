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

---

## 📊 Results (Using RoboTwin-2.0 Leaderboard Settings)

|    **Settings**   |   Easy  |   Hard  |
| :--------------------: | :--: | :--: |
|     **Success (%)**    | 70.0 | 39.0 |



