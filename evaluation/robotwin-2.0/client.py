"""RoboTwin-2.0 receding-horizon rollout client.

The model is trained in each arm's robot-base frame. This client converts
observations to that frame before the HTTP request and converts the selected
prediction back to world coordinates before calling RoboTwin.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import json_numpy
import numpy as np
import requests
import transforms3d as t3d
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ORIGINAL_CWD = Path.cwd()
ROBOTWIN_ROOT = (PROJECT_ROOT / "third_party" / "RoboTwin").resolve()
if not ROBOTWIN_ROOT.is_dir():
    raise FileNotFoundError(
        f"RoboTwin checkout not found: {ROBOTWIN_ROOT}. Run scripts/bootstrap_robotwin2.sh."
    )
if str(ROBOTWIN_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOTWIN_ROOT))
os.chdir(ROBOTWIN_ROOT)

TASKS = (
    "beat_block_hammer",
    "stack_blocks_two",
    "move_can_pot",
    "open_microwave",
    "place_dual_shoes",
)
DOMAINS: dict[str, dict[str, Any]] = {
    "aloha-agilex": {"domain_id": 0, "embodiment": ["aloha-agilex"]},
    "ARX-X5": {"domain_id": 1, "embodiment": ["ARX-X5", "ARX-X5", 0.60]},
    "piper-dual": {"domain_id": 2, "embodiment": ["piper", "piper", 0.60]},
}


def _pose_to_matrix(pose: Any) -> np.ndarray:
    """Convert RoboTwin/SAPIEN [p, q(wxyz)] pose to a homogeneous matrix."""
    if hasattr(pose, "p") and hasattr(pose, "q"):
        xyz, quat = np.asarray(pose.p, dtype=np.float64), np.asarray(pose.q, dtype=np.float64)
    else:
        value = np.asarray(pose, dtype=np.float64).reshape(-1)
        if value.size != 7:
            raise ValueError(f"pose must have 7 values, got {value.shape}")
        xyz, quat = value[:3], value[3:]
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = t3d.quaternions.quat2mat(quat)
    matrix[:3, 3] = xyz
    return matrix


def _matrix_to_pose(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ValueError(f"matrix must be 4x4, got {matrix.shape}")
    return np.concatenate([matrix[:3, 3], t3d.quaternions.mat2quat(matrix[:3, :3])]).astype(np.float32)


def _matrix_to_rotate6d(matrix: np.ndarray) -> np.ndarray:
    # Interleaved row-major first two columns: [R00,R01,R10,R11,R20,R21].
    return np.asarray(matrix, dtype=np.float32)[:3, :2].reshape(6)


def _rotate6d_to_matrix(value: np.ndarray) -> np.ndarray:
    value = np.asarray(value, dtype=np.float64).reshape(6)
    a1, a2 = value[[0, 2, 4]], value[[1, 3, 5]]
    n1 = np.linalg.norm(a1)
    if n1 < 1e-8:
        raise ValueError("invalid rotate6d first column")
    b1 = a1 / n1
    b2 = a2 - np.dot(b1, a2) * b1
    n2 = np.linalg.norm(b2)
    if n2 < 1e-8:
        raise ValueError("invalid rotate6d second column")
    b2 /= n2
    return np.stack((b1, b2, np.cross(b1, b2)), axis=1)


def _base_pose(world_pose: Any, base_pose: Any) -> np.ndarray:
    return np.linalg.inv(_pose_to_matrix(base_pose)) @ _pose_to_matrix(world_pose)


def _world_pose(base_xyz: np.ndarray, base_rotate6d: np.ndarray, base_pose: Any) -> np.ndarray:
    local = np.eye(4, dtype=np.float64)
    local[:3, :3] = _rotate6d_to_matrix(base_rotate6d)
    local[:3, 3] = np.asarray(base_xyz, dtype=np.float64)
    return _matrix_to_pose(_pose_to_matrix(base_pose) @ local)


def _check_round_trip(base_pose: Any, world_pose: Any) -> None:
    base = _base_pose(world_pose, base_pose)
    recovered = _pose_to_matrix(base_pose) @ base
    reference = _pose_to_matrix(world_pose)
    pos_error = float(np.max(np.abs(recovered[:3, 3] - reference[:3, 3])))
    rot_error = float(np.max(np.abs(recovered[:3, :3] - reference[:3, :3])))
    if pos_error >= 1e-5 or rot_error >= 1e-5:
        raise AssertionError(f"base/world round-trip failed: position={pos_error}, rotation={rot_error}")


def _base_eef20(obs: dict[str, Any], env: Any) -> np.ndarray:
    endpose = obs.get("endpose", {})
    left_world = np.asarray(endpose["left_endpose"], dtype=np.float64).reshape(7)
    right_world = np.asarray(endpose["right_endpose"], dtype=np.float64).reshape(7)
    left_base = _base_pose(left_world, env.robot.left_entity_origion_pose)
    right_base = _base_pose(right_world, env.robot.right_entity_origion_pose)
    left_raw = float(np.asarray(endpose["left_gripper"]).reshape(-1)[0])
    right_raw = float(np.asarray(endpose["right_gripper"]).reshape(-1)[0])
    # Model convention is g_closed=1; RoboTwin observations use g_open=1.
    left_grip, right_grip = 1.0 - np.clip(left_raw, 0.0, 1.0), 1.0 - np.clip(right_raw, 0.0, 1.0)
    return np.concatenate(
        [left_base[:3, 3], _matrix_to_rotate6d(left_base), [left_grip],
         right_base[:3, 3], _matrix_to_rotate6d(right_base), [right_grip]], axis=0
    ).astype(np.float32)


def _images(obs: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    camera = obs.get("observation", {})
    try:
        images = (
            np.asarray(camera["head_camera"]["rgb"]),
            np.asarray(camera["left_camera"]["rgb"]),
            np.asarray(camera["right_camera"]["rgb"]),
        )
    except KeyError as exc:
        raise KeyError("RoboTwin observation must contain head/left/right RGB cameras") from exc
    for image in images:
        if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
            raise ValueError(f"camera image must be uint8 HxWx3 RGB, got {image.shape} {image.dtype}")
    return images


class ClientModel:
    def __init__(self, host: str, port: int, domain_id: int, timeout: float = 30.0):
        self.url = f"http://{host}:{port}/act"
        self.domain_id, self.timeout = int(domain_id), float(timeout)
        self.instruction, self._round_trip_checked = "", False

    def reset_episode(self) -> None:
        self._round_trip_checked = False

    def set_instruction(self, instruction: str) -> None:
        self.instruction = str(instruction)

    def step(self, obs: dict[str, Any], env: Any) -> np.ndarray:
        head, left, right = _images(obs)
        proprio = _base_eef20(obs, env)
        if not self._round_trip_checked:
            _check_round_trip(env.robot.left_entity_origion_pose, obs["endpose"]["left_endpose"])
            _check_round_trip(env.robot.right_entity_origion_pose, obs["endpose"]["right_endpose"])
            self._round_trip_checked = True
        query = {
            "domain_id": self.domain_id,
            "proprio": json_numpy.dumps(proprio),
            "language_instruction": self.instruction,
            "image0": json_numpy.dumps(head),
            "image1": json_numpy.dumps(left),
            "image2": json_numpy.dumps(right),
        }
        response = requests.post(self.url, json=query, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        action = np.asarray(payload.get("action"), dtype=np.float32)
        if action.ndim == 3 and action.shape[0] == 1:
            action = action[0]
        if action.shape != (30, 20):
            raise ValueError(f"model must return [30,20], got {action.shape}")
        if not np.isfinite(action).all():
            raise ValueError("model action contains non-finite values")
        return action


def load_env(task_name: str, task_config: str, domain: str):
    if domain not in DOMAINS:
        raise ValueError(f"unknown domain {domain!r}; choose from {sorted(DOMAINS)}")
    task = getattr(importlib.import_module(f"envs.{task_name}"), task_name)()
    config_root = Path("env_cfg") / "task_config"
    args = yaml.safe_load((config_root / f"{task_config}.yml").read_text())
    args.update(task_name=task_name, task_config=task_config, embodiment=list(DOMAINS[domain]["embodiment"]))
    embodiment_types = yaml.safe_load((config_root / "_embodiment_config.yml").read_text())
    camera_config = yaml.safe_load((config_root / "_camera_config.yml").read_text())

    def robot_file(name: str) -> str:
        value = embodiment_types[name]["file_path"]
        if not value:
            raise ValueError(f"missing embodiment path for {name}")
        return value

    args["head_camera_h"] = camera_config[args["camera"]["head_camera_type"]]["h"]
    args["head_camera_w"] = camera_config[args["camera"]["head_camera_type"]]["w"]
    embodiment = args["embodiment"]
    if len(embodiment) == 1:
        args["left_robot_file"] = args["right_robot_file"] = robot_file(embodiment[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment) == 3:
        args["left_robot_file"], args["right_robot_file"] = robot_file(embodiment[0]), robot_file(embodiment[1])
        args["embodiment_dis"], args["dual_arm_embodied"] = float(embodiment[2]), False
    else:
        raise ValueError(f"invalid embodiment config: {embodiment}")
    for side in ("left", "right"):
        args[f"{side}_embodiment_config"] = yaml.safe_load(
            (Path(args[f"{side}_robot_file"]) / "config.yml").read_text()
        )
    return task, args


def _rollout(env: Any, policy: ClientModel, *, exec_points: int, max_steps: int) -> tuple[int, list[np.ndarray], int]:
    if not 1 <= exec_points <= 30:
        raise ValueError("exec_points must be in [1,30]")
    obs, frames, request_count = env.get_obs(), [], 0
    if max_steps <= 0:
        max_steps = int(getattr(env, "step_lim", 200))
    for _ in range(max_steps):
        action = policy.step(obs, env)
        request_count += 1
        for point in action[:exec_points]:
            left_pose = _world_pose(point[:3], point[3:9], env.robot.left_entity_origion_pose)
            right_pose = _world_pose(point[10:13], point[13:19], env.robot.right_entity_origion_pose)
            left_raw, right_raw = 1.0 - float(np.clip(point[9], 0.0, 1.0)), 1.0 - float(np.clip(point[19], 0.0, 1.0))
            env.take_action(
                np.concatenate([left_pose, [left_raw], right_pose, [right_raw]]).astype(np.float32), action_type="ee"
            )
            obs = env.get_obs()
            frames.append(np.asarray(obs["observation"]["head_camera"]["rgb"]))
            success = bool(env.check_success())
            if success or getattr(env, "actor_pose", True) is False:
                return int(success), frames, request_count
    return 0, frames, request_count


def eval_episodes(*, task_name: str, domain: str, task_config: str, policy: ClientModel,
                  num_episodes: int, seed: int, output_dir: Path, exec_points: int,
                  max_steps: int, save_video: bool) -> dict[str, Any]:
    env, args = load_env(task_name, task_config, domain)
    args.update(eval_mode=True, render_freq=0, collect_data=False, eval_video_log=False)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for episode_index in range(num_episodes):
        episode_seed = 2000 * (1 + seed) + episode_index
        record: dict[str, Any] = {"episode": episode_index, "seed": episode_seed, "task": task_name, "domain": domain}
        frames: list[np.ndarray] = []
        try:
            env.setup_demo(now_ep_num=episode_index, seed=episode_seed, is_test=True, **args)
            # Keep an initial frame even when planner setup fails, so every
            # attempted episode has an inspectable video artifact.
            try:
                frames.append(np.asarray(env.get_obs()["observation"]["head_camera"]["rgb"]))
            except Exception:
                pass
            if not getattr(env, "plan_success", True):
                record.update(status="planner_failure", success=0)
            else:
                policy.reset_episode()
                policy.set_instruction(task_name.replace("_", " "))
                status, frames, requests_count = _rollout(env, policy, exec_points=exec_points, max_steps=max_steps)
                record.update(status="success" if status else "failure", success=int(status), requests=requests_count)
        except Exception as exc:
            record.update(status="exception", success=0, error=f"{type(exc).__name__}: {exc}")
        finally:
            try:
                env.close_env()
            except Exception as exc:
                record.setdefault("close_error", f"{type(exc).__name__}: {exc}")
        if save_video and frames:
            video_path = output_dir / f"episode_{episode_index:04d}_{record['status']}.mp4"
            imageio.mimsave(video_path, frames, fps=30)
            record.update(video=str(video_path), video_frames=len(frames), video_duration_sec=len(frames) / 30.0)
        records.append(record)
        with (output_dir / "episodes.jsonl").open("a", encoding="utf-8") as file:
            file.write(json.dumps(record) + "\n")
    summary = {
        "task": task_name, "domain": domain, "domain_id": DOMAINS[domain]["domain_id"],
        "num_episodes": num_episodes, "successes": sum(r["success"] for r in records),
        "planner_failures": sum(r["status"] == "planner_failure" for r in records),
        "exceptions": sum(r["status"] == "exception" for r in records),
        "success_rate": sum(r["success"] for r in records) / max(1, len(records)),
        "exec_points": exec_points, "records": records,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task_name", choices=TASKS, required=True)
    parser.add_argument("--domain", choices=tuple(DOMAINS), required=True)
    parser.add_argument("--task_config", default="demo_clean")
    parser.add_argument("--num_episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_path", type=Path, required=True)
    parser.add_argument("--exec_points", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=0, help="0 uses the task's RoboTwin step limit")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--save_video", action="store_true")
    args = parser.parse_args()
    if not args.output_path.is_absolute():
        args.output_path = ORIGINAL_CWD / args.output_path
    args.output_path = args.output_path.resolve()
    if args.device >= 0:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.set_device(args.device)
        except ImportError:
            pass
    policy = ClientModel(args.host, args.port, DOMAINS[args.domain]["domain_id"])
    summary = eval_episodes(
        task_name=args.task_name, domain=args.domain, task_config=args.task_config, policy=policy,
        num_episodes=args.num_episodes, seed=args.seed, output_dir=args.output_path,
        exec_points=args.exec_points, max_steps=args.max_steps, save_video=args.save_video,
    )
    print(json.dumps({k: summary[k] for k in ("task", "domain", "successes", "num_episodes", "success_rate")}, indent=2))
    raise SystemExit(0 if summary["exceptions"] == 0 else 1)


if __name__ == "__main__":
    main()
