"""Native XPolicyLab-v1.0 RoboTwin2 fine-tuning handler.

This module intentionally does not share the legacy ``RobotWin2Handler``:
that handler consumes the old ``endpose/observation`` layout and has different
frequency/frame semantics.  The handler below consumes normalized v1.0 HDF5
files and the manifest window index produced by ``scripts/preprocess_robotwin2``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation, Slerp

from ..utils import decode_image_bit
from .base import DomainHandler


DOMAIN_IDS = {"robotwin2_ft_aloha": 0, "robotwin2_ft_arx_x5": 1, "robotwin2_ft_piper_dual": 2}
CAMERA_KEYS = ("vision/cam_head/colors", "vision/cam_left_wrist/colors", "vision/cam_right_wrist/colors")


def _rot_sf(quat: np.ndarray) -> Rotation:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.shape[-1] != 4:
        raise ValueError(f"RoboTwin quaternion must be [w,x,y,z], got {quat.shape}")
    return Rotation.from_quat(quat[..., [1, 2, 3, 0]])


def _pose_to_base(pose: np.ndarray, base_pose: Sequence[float]) -> np.ndarray:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"pose must be [T,7], got {pose.shape}")
    base = np.eye(4, dtype=np.float64)
    base[:3, :3] = _rot_sf(np.asarray(base_pose)[3:]).as_matrix()
    base[:3, 3] = np.asarray(base_pose, dtype=np.float64)[:3]
    world = np.repeat(np.eye(4, dtype=np.float64)[None], len(pose), axis=0)
    world[:, :3, :3] = _rot_sf(pose[:, 3:]).as_matrix()
    world[:, :3, 3] = pose[:, :3]
    local = np.linalg.inv(base)[None] @ world
    cols = local[:, :3, :3][:, :, :2].reshape(len(pose), 6)
    return np.concatenate([local[:, :3, 3], cols], axis=1).astype(np.float32)


def _eef_series(f: h5py.File, side: str, base_pose: Sequence[float]) -> np.ndarray:
    state = np.asarray(f[f"state/{side}_ee_poses"], dtype=np.float64)
    action = np.asarray(f[f"action/{side}_ee_poses"], dtype=np.float64)
    if len(state) < 1 or len(action) != len(state):
        raise ValueError(f"state/action pose length mismatch for {side}: {len(state)}/{len(action)}")
    # Official writer alignment: state[1:] == action[:-1], no extra shift.
    if len(state) > 1:
        if np.max(np.abs(state[1:, :3] - action[:-1, :3])) >= 1e-5:
            raise ValueError(f"state/action position continuity failed for {side}")
        r0, r1 = _rot_sf(state[1:, 3:]), _rot_sf(action[:-1, 3:])
        if np.max(np.abs((r0.inv() * r1).as_rotvec())) >= 1e-5:
            raise ValueError(f"state/action rotation continuity failed for {side}")
    pose = np.concatenate([state[:1], action], axis=0)
    g0 = np.asarray(f[f"state/{side}_ee_joint_states"], dtype=np.float32).reshape(-1)
    g1 = np.asarray(f[f"action/{side}_ee_joint_states"], dtype=np.float32).reshape(-1)
    grip = np.concatenate([g0[:1], g1])
    if len(grip) != len(pose) or not np.isfinite(grip).all():
        raise ValueError(f"invalid {side} gripper stream")
    # Raw RoboTwin convention is 1=open, 0=closed.  X-VLA is 0=open, 1=closed.
    grip = 1.0 - np.clip(grip, 0.0, 1.0)
    out = np.concatenate([_pose_to_base(pose, base_pose), grip[:, None]], axis=1)
    if out.shape[1] != 10 or not np.isfinite(out).all():
        raise ValueError(f"invalid {side} EEF6D values")
    return out


def _interp(series: np.ndarray, query: np.ndarray, times: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.float64)
    out = np.empty((len(query), 10), dtype=np.float32)
    for col in range(3):
        out[:, col] = np.interp(query, times, series[:, col])
    a1 = series[:, 3:9:2]
    a2 = series[:, 4:9:2]
    b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-12)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-12)
    b3 = np.cross(b1, b2)
    rotations = Rotation.from_matrix(np.stack([b1, b2, b3], axis=2))
    if len(times) == 1:
        mats = np.repeat(rotations.as_matrix(), len(query), axis=0)
    else:
        mats = Slerp(times, rotations)(np.clip(query, times[0], times[-1])).as_matrix()
    out[:, 3:9] = mats[:, :, :2].reshape(len(query), 6)
    nearest = np.clip(np.rint(query / max(times[1] - times[0], 1e-12)).astype(int), 0, len(series) - 1) if len(times) > 1 else np.zeros(len(query), dtype=int)
    out[:, 9] = series[nearest, 9]
    return out


class RobotWin2FTHandler(DomainHandler):
    """Finite episode/window reader for normalized RoboTwin2 HDF5."""

    dataset_name = "robotwin2_ft"

    def __init__(self, meta: dict, num_views: int = 3, image_aug=None, windows: list[dict] | None = None):
        super().__init__(meta, num_views)
        if num_views != 3:
            raise ValueError("RoboTwin2 fine-tuning freezes exactly three camera views")
        self.image_aug = image_aug
        self.windows = windows if windows is not None else []
        self.domain_id = int(meta["domain_id"])
        if self.domain_id not in (0, 1, 2):
            raise ValueError(f"RoboTwin2 domain_id must be 0/1/2, got {self.domain_id}")
        self.base_left = meta["base_pose_left"]
        self.base_right = meta["base_pose_right"]
        self.qdur = float(meta.get("qdur_sec", 1.0))
        self.num_actions = int(meta.get("num_actions", 30))
        if self.qdur != 1.0 or self.num_actions != 30:
            raise ValueError("RoboTwin2 contract requires qdur=1.0 and 30 actions")

    @staticmethod
    def _instruction(f: h5py.File) -> str:
        raw = f["instructions"][()]
        if isinstance(raw, bytes):
            raw = raw.decode("utf-8")
        value = json.loads(str(raw))
        if not isinstance(value, list) or not value or not str(value[0]).strip():
            raise ValueError("instructions must be a non-empty JSON list")
        return str(value[0]).strip()

    def sample_from_entry(self, entry: dict) -> dict:
        path = Path(entry["hdf5_path"])
        frame_idx = int(entry["frame_idx"])
        if int(entry["domain_id"]) != self.domain_id:
            raise ValueError("window/domain mismatch")
        with h5py.File(path, "r") as f:
            version = f["data_format_version"][()]
            if isinstance(version, bytes):
                version = version.decode()
            if version != "v1.0":
                raise ValueError(f"{path}: expected v1.0, got {version!r}")
            freq = float(np.asarray(f["additional_info/frequency"]))
            left, right = _eef_series(f, "left", self.base_left), _eef_series(f, "right", self.base_right)
            candidate_count = len(left) - 1
            if not 0 <= frame_idx < candidate_count:
                raise IndexError(f"frame_idx {frame_idx} outside [0,{candidate_count})")
            times = np.arange(len(left), dtype=np.float64) / freq
            t0 = frame_idx / freq
            query = t0 + np.linspace(1.0 / 30.0, 1.0, 30, dtype=np.float64)
            future_l = _interp(left, query, times)
            future_r = _interp(right, query, times)
            trajectory = np.concatenate([
                np.concatenate([left[frame_idx], right[frame_idx]])[None],
                np.concatenate([future_l, future_r], axis=1),
            ], axis=0)
            instruction = self._instruction(f)
            images = []
            for key in CAMERA_KEYS:
                if key not in f or len(f[key]) != candidate_count:
                    raise ValueError(f"{path}: missing/misaligned camera {key}")
                image = np.asarray(decode_image_bit(f[key][frame_idx]))
                if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                    raise ValueError(f"{path}: decoder did not return RGB uint8 for {key}")
                if self.image_aug is None:
                    image = Image.fromarray(image)
                    tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).float().div(255.0)
                else:
                    tensor = self.image_aug(Image.fromarray(image))
                images.append(tensor)
        if trajectory.shape != (31, 20) or not np.isfinite(trajectory).all():
            raise ValueError(f"invalid trajectory shape/values: {trajectory.shape}")
        proprio = torch.from_numpy(trajectory[0].copy()).float()
        action = torch.from_numpy(trajectory[1:].copy()).float()
        if not torch.isfinite(proprio).all() or not torch.isfinite(action).all():
            raise ValueError("non-finite proprio/action")
        grip = action[:, (9, 19)]
        if torch.any((grip < 0) | (grip > 1)):
            raise ValueError("gripper target must be in [0,1]")
        return {
            "language_instruction": instruction,
            "image_input": torch.stack(images, dim=0),
            "image_mask": torch.ones(3, dtype=torch.bool),
            "proprio": proprio,
            "action": action,
            "domain_id": torch.tensor(self.domain_id, dtype=torch.long),
            "task": str(entry["task"]),
            "terminal_hold": bool(entry.get("terminal_hold", False)),
            "window_key": f"{path}:{frame_idx}",
        }

    def iter_episode(self, traj_idx: int, *, num_actions: int = 30, training: bool = True,
                     image_aug=None, lang_aug_map=None, action_mode="ee6d", **kwargs) -> Iterable[dict]:
        if num_actions != 30 or action_mode != "ee6d":
            raise ValueError("RoboTwin2 handler freezes action_mode=ee6d and num_actions=30")
        entries = [e for e in self.windows if e["hdf5_path"] == self.meta["datalist"][traj_idx]]
        for entry in entries:
            yield self.sample_from_entry(entry)
