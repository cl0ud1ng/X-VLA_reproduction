# Optional RoboTwin runtime

The `RoboTwin/` directory is intentionally ignored by Git.  A machine that
needs preprocessing or simulation rollout must clone the pinned runtime here:

```bash
git clone --recurse-submodules https://github.com/RoboTwin-Platform/RoboTwin.git third_party/RoboTwin
git -C third_party/RoboTwin checkout --detach 96c1fea
git -C third_party/RoboTwin submodule update --init --checkout
git -C third_party/RoboTwin/XPolicyLab checkout c37109c500be67d0dea6b36bf7337bbd26e763cd
```

Run `scripts/bootstrap_robotwin2.sh` from the project root to validate this
checkout and prepare the model and RoboTwin training data.  The simulator's
large scene assets are machine-local and are not part of the Git repository.

Rollout also needs the official cuRobo v0.7.8 (`d64c4b005459db10c5dd867d8b30a87d5bda9bdb`), cloned into ignored `third_party/curobo/` by bootstrap. A complete CUDA 12.1 toolkit is required to build its extensions. See [deployment](../docs/robotwin2_deployment.md).
