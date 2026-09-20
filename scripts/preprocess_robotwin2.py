#!/usr/bin/env python3
"""Normalize official RoboTwin2.0 clean archives for X-VLA fine-tuning.

The public clean archives currently contain RoboTwin's native HDF5 schema
(``endpose/`` and ``observation/``), while the experiment contract consumes
XPolicyLab v1.0.  This script performs the official, tracked native-to-v1.0
field mapping, validates all three RGB cameras, converts the old archive to a
schema-auditable HDF5, and writes an index of 1 s / 30 point absolute EEF6D
windows.  Images stay encoded in the normalized HDF5 and are decoded only by
the official ``decode_image_bit`` implementation at sample time.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation, Slerp

try:
    from robotwin2_decode import decode_image_bit
except Exception as exc:
    raise RuntimeError("scripts/robotwin2_decode.py is required") from exc

TASKS = (
    "beat_block_hammer",
    "stack_blocks_two",
    "move_can_pot",
    "open_microwave",
    "place_dual_shoes",
)
DOMAINS = {
    "aloha-agilex": {"domain_id": 0, "embodiment": "aloha-agilex", "robot_type": "robotwin2_ft_aloha"},
    "arx-x5": {"domain_id": 1, "embodiment": "ARX-X5", "robot_type": "robotwin2_ft_arx_x5"},
    "piper": {"domain_id": 2, "embodiment": "piper-dual", "robot_type": "robotwin2_ft_piper_dual"},
}


def _project_root() -> Path:
    """Return the checkout root without depending on the host machine path."""
    return Path(__file__).resolve().parents[1]


def _portable_path(path: Path, project_root: Path) -> str:
    """Store paths relative to the project so manifests work after a clone."""
    try:
        return path.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(f"path must be inside project root: {path}") from exc
CAMERAS = (
    ("head_camera", "cam_head"),
    ("left_camera", "cam_left_wrist"),
    ("right_camera", "cam_right_wrist"),
)


def _rotation_from_scalar_first(quat: np.ndarray) -> Rotation:
    """SciPy 1.10-compatible transforms3d/RoboTwin [w,x,y,z] reader."""
    q = np.asarray(quat, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quaternion must end in 4, got {q.shape}")
    return Rotation.from_quat(q[..., [1, 2, 3, 0]])


def _pose_matrix(pose: list[float] | tuple[float, ...]) -> np.ndarray:
    p = np.asarray(pose, dtype=np.float64)
    if p.shape != (7,) or not np.isfinite(p).all():
        raise ValueError(f"robot base pose must be finite [x,y,z,w,x,y,z], got {p.shape}")
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _rotation_from_scalar_first(p[3:]).as_matrix()
    T[:3, 3] = p[:3]
    return T


def _pose_to_base(pose: np.ndarray, base: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pose = np.asarray(pose, dtype=np.float64)
    if pose.ndim != 2 or pose.shape[1] != 7:
        raise ValueError(f"expected [T,7] pose, got {pose.shape}")
    T = np.repeat(np.eye(4, dtype=np.float64)[None], len(pose), axis=0)
    T[:, :3, :3] = _rotation_from_scalar_first(pose[:, 3:]).as_matrix()
    T[:, :3, 3] = pose[:, :3]
    Tb = np.linalg.inv(base)[None] @ T
    v6 = Tb[:, :3, :3][:, :, :2].reshape(len(pose), 6)
    return Tb[:, :3, 3].astype(np.float32), v6.astype(np.float32)


def _concat_eef(pose: np.ndarray, gripper: np.ndarray, base: np.ndarray) -> np.ndarray:
    xyz, v6 = _pose_to_base(pose, base)
    g = 1.0 - np.clip(np.asarray(gripper, dtype=np.float32).reshape(-1, 1), 0.0, 1.0)
    out = np.concatenate([xyz, v6, g], axis=1)
    if out.shape[1] != 10 or not np.isfinite(out).all():
        raise ValueError("invalid base-frame EEF6D values")
    return out


def _decode_check(raw: Any) -> tuple[int, int, int]:
    if decode_image_bit is None:
        raise RuntimeError("official decode_image_bit could not be imported")
    image = np.asarray(decode_image_bit(raw))
    if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
        raise ValueError(f"decode_image_bit must return HWC uint8 RGB, got {image.shape}/{image.dtype}")
    return tuple(int(x) for x in image.shape)


def _load_instruction(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    values = data.get("seen") if isinstance(data, dict) else data
    if not isinstance(values, list):
        raise ValueError(f"instruction file must contain a list or seen list: {path}")
    values = [str(v).strip() for v in values if str(v).strip()]
    if not values:
        raise ValueError(f"empty seen instruction list: {path}")
    return values


def _write_normalized(native: Path, output: Path, instructions: list[str], frequency: float, base_left: list[float], base_right: list[float], source_archive: Path) -> tuple[int, tuple[int, int, int]]:
    with h5py.File(native, "r") as src:
        required = [
            "endpose/left_endpose", "endpose/right_endpose",
            "endpose/left_gripper", "endpose/right_gripper",
            "joint_action/left_arm", "joint_action/right_arm",
            "joint_action/left_gripper", "joint_action/right_gripper",
        ]
        for key in required:
            if key not in src:
                raise KeyError(f"{native}: missing required dataset {key}")
        poses = {side: np.asarray(src[f"endpose/{side}_endpose"], dtype=np.float64) for side in ("left", "right")}
        grips = {side: np.asarray(src[f"endpose/{side}_gripper"], dtype=np.float32) for side in ("left", "right")}
        joints = {
            "left_arm": np.asarray(src["joint_action/left_arm"], dtype=np.float32),
            "right_arm": np.asarray(src["joint_action/right_arm"], dtype=np.float32),
            "left_gripper": grips["left"],
            "right_gripper": grips["right"],
        }
        T = len(poses["left"])
        if T < 2 or any(len(v) != T for v in [poses["right"], grips["left"], grips["right"], *joints.values()]):
            raise ValueError(f"{native}: all native streams must have equal length >=2")
        for side, pose in poses.items():
            if pose.shape != (T, 7) or not np.isfinite(pose).all():
                raise ValueError(f"{native}: {side} endpose must be finite [T,7]")
        # Official writer alignment: state=trajectory[:-1], action=trajectory[1:],
        # vision=images[:-1].  We retain world-frame pose in the normalized
        # schema; the window index and handler apply the recorded base transform.
        output.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(output, "w") as dst:
            text = h5py.string_dtype("utf-8")
            dst.create_dataset("data_format_version", data="v1.0", dtype=text)
            dst.create_dataset("instructions", data=json.dumps(instructions, ensure_ascii=False), dtype=text)
            dst.create_group("additional_info").create_dataset("frequency", data=np.asarray(frequency, dtype=np.float32))
            dst.attrs["source_schema"] = "RoboTwin native HDF5 -> official pkl2hdf5 field mapping"
            # Keep provenance portable across machines.  The manifest stores
            # the repository-relative archive path; never embed the host's
            # absolute checkout path in HDF5 metadata.
            dst.attrs["source_archive"] = str(source_archive)
            dst.attrs["base_pose_left"] = json.dumps(base_left)
            dst.attrs["base_pose_right"] = json.dumps(base_right)
            state, action = dst.create_group("state"), dst.create_group("action")
            for side in ("left", "right"):
                state.create_dataset(f"{side}_ee_poses", data=poses[side][:-1].astype(np.float32))
                action.create_dataset(f"{side}_ee_poses", data=poses[side][1:].astype(np.float32))
                state.create_dataset(f"{side}_ee_joint_states", data=grips[side][:-1, None])
                action.create_dataset(f"{side}_ee_joint_states", data=grips[side][1:, None])
                state.create_dataset(f"{side}_arm_joint_states", data=joints[f"{side}_arm"][:-1])
                action.create_dataset(f"{side}_arm_joint_states", data=joints[f"{side}_arm"][1:])
            vision = dst.create_group("vision")
            image_shape = None
            for native_key, target_key in CAMERAS:
                src_key = f"observation/{native_key}/rgb"
                if src_key not in src:
                    raise KeyError(f"{native}: missing required camera {src_key}")
                raw = src[src_key][()]
                if len(raw) != T:
                    raise ValueError(f"{native}: camera {src_key} length {len(raw)} != {T}")
                shape = _decode_check(raw[0])
                if image_shape is None:
                    image_shape = shape
                elif shape != image_shape:
                    raise ValueError(f"{native}: camera resolution mismatch {shape} vs {image_shape}")
                group = vision.create_group(target_key)
                max_len = max(len(bytes(x).rstrip(b"\\0")) for x in raw[:-1])
                encoded = np.asarray([bytes(x).rstrip(b"\\0") for x in raw[:-1]], dtype=f"S{max_len}")
                group.create_dataset("colors", data=encoded)
                group.create_dataset("shape", data=np.asarray(shape, dtype=np.int32))
            assert image_shape is not None
    return T - 1, image_shape


def _continuous_eef(h5_path: Path, base_left: np.ndarray, base_right: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, tuple[int, int, int]]:
    with h5py.File(h5_path, "r") as f:
        version = f["data_format_version"][()]
        if isinstance(version, bytes): version = version.decode()
        if version != "v1.0":
            raise ValueError(f"{h5_path}: expected v1.0, got {version!r}")
        freq = float(np.asarray(f["additional_info/frequency"]))
        if not math.isfinite(freq) or freq <= 0:
            raise ValueError(f"{h5_path}: invalid frequency {freq}")
        state_n = len(f["state/left_ee_poses"])
        if state_n < 1:
            raise ValueError(f"{h5_path}: no state frames")
        poses = {}
        for side, base in (("left", base_left), ("right", base_right)):
            state = np.asarray(f[f"state/{side}_ee_poses"], dtype=np.float64)
            action = np.asarray(f[f"action/{side}_ee_poses"], dtype=np.float64)
            if len(state) != state_n or len(action) != state_n:
                raise ValueError(f"{h5_path}: state/action length mismatch")
            # No hidden one-frame shift: normalized action[k] must be native
            # continuous pose[k+1].  This check is deliberately strict.
            R0 = _rotation_from_scalar_first(state[1:, 3:])
            R1 = _rotation_from_scalar_first(action[:-1, 3:])
            if len(state) > 1 and (np.max(np.abs(state[1:, :3] - action[:-1, :3])) > 1e-5 or np.max(np.abs((R0.inv() * R1).as_rotvec())) > 1e-5):
                raise ValueError(f"{h5_path}: state/action continuity check failed for {side}")
            full_pose = np.concatenate([state[:1], action], axis=0)
            raw_g_state = np.asarray(f[f"state/{side}_ee_joint_states"], dtype=np.float32).reshape(-1)
            raw_g_action = np.asarray(f[f"action/{side}_ee_joint_states"], dtype=np.float32).reshape(-1)
            full_g = np.concatenate([raw_g_state[:1], raw_g_action], axis=0)
            eef = _concat_eef(full_pose, full_g, base)
            poses[side] = eef
        shape = _decode_check(f["vision/cam_head/colors"][0])
        return poses["left"], poses["right"], freq, shape


def _window_entries(h5_path: Path, task: str, domain: str, instruction: list[str], base_left: list[float], base_right: list[float], num_actions: int, qdur: float, static_threshold: float, project_root: Path) -> tuple[list[dict], int, int]:
    left, right, freq, image_shape = _continuous_eef(h5_path, _pose_matrix(base_left), _pose_matrix(base_right))
    # Images/state are available at continuous indices [0, T-2]; the final
    # action-only point is used for terminal-hold interpolation but is never a
    # current observation candidate.
    candidate_count = len(left) - 1
    times = np.arange(len(left), dtype=np.float64) / freq
    entries: list[dict] = []
    terminal_count = 0
    # Use the same camera length checks performed during normalization.
    with h5py.File(h5_path, "r") as f:
        for key in ("vision/cam_head/colors", "vision/cam_left_wrist/colors", "vision/cam_right_wrist/colors"):
            if len(f[key]) != candidate_count:
                raise ValueError(f"{h5_path}: {key} does not align with state frames")
    for frame_idx in range(candidate_count):
        t0 = float(times[frame_idx])
        query = t0 + np.linspace(1.0 / 30.0, qdur, num_actions, dtype=np.float64)
        terminal = bool(np.any(query > times[-1] + 1e-9))
        if terminal:
            terminal_count += 1
        q = np.clip(query, times[0], times[-1])
        def interp(series: np.ndarray) -> np.ndarray:
            # Position is linearly interpolated in metres, orientation uses
            # geodesic SO(3) Slerp, and gripper is nearest-neighbour.  The
            # 6D columns are reconstructed to matrices before Slerp.
            out = np.empty((len(q), 10), dtype=np.float32)
            out[:, :3] = np.stack([np.interp(q, times, series[:, col]) for col in range(3)], axis=1)
            a1 = series[:, 3:9:2]
            a2 = series[:, 4:9:2]
            b1 = a1 / np.maximum(np.linalg.norm(a1, axis=1, keepdims=True), 1e-12)
            b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
            b2 = b2 / np.maximum(np.linalg.norm(b2, axis=1, keepdims=True), 1e-12)
            b3 = np.cross(b1, b2)
            mats = np.stack([b1, b2, b3], axis=2)
            rotations = Rotation.from_matrix(mats)
            out[:, 3:9] = Slerp(times, rotations)(q).as_matrix()[:, :, :2].reshape(len(q), 6)
            nearest = np.clip(np.rint(q * freq).astype(int), 0, len(series) - 1)
            out[:, 9] = series[nearest, 9]
            return out
        future_l = interp(left)
        future_r = interp(right)
        if not terminal:
            pos_delta = max(float(np.max(np.abs(future_l[0, :3] - left[frame_idx, :3]))), float(np.max(np.abs(future_r[0, :3] - right[frame_idx, :3]))))
            grip_same = bool(future_l[0, 9] == left[frame_idx, 9] and future_r[0, 9] == right[frame_idx, 9])
            if pos_delta < static_threshold and grip_same:
                continue
        entries.append({
            "hdf5_path": _portable_path(h5_path, project_root),
            "frame_idx": frame_idx,
            "timestamp_sec": t0,
            "task": task,
            "domain": domain,
            "domain_id": DOMAINS[domain]["domain_id"],
            "instruction": instruction[0],
            "terminal_hold": terminal,
            "qdur_sec": qdur,
            "num_actions": num_actions,
            "image_shape": list(image_shape),
            "base_pose_left": base_left,
            "base_pose_right": base_right,
        })
    return entries, candidate_count, terminal_count


def _read_base_config(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    for domain in DOMAINS:
        item = data.get(domain)
        if not isinstance(item, dict) or "left" not in item or "right" not in item:
            raise ValueError(f"base config missing left/right for {domain}")
        for side in ("left", "right"):
            pose = item[side]
            if not isinstance(pose, list) or len(pose) != 7:
                raise ValueError(f"base config {domain}/{side} must be [x,y,z,w,x,y,z]")
    return data


def _extract_archive(archive: Path, domain: str, target: Path) -> tuple[list[Path], Path, Path]:
    target.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        members = zf.namelist()
        hdf_members = sorted([m for m in members if m.endswith(".hdf5") and Path(m).name.startswith("episode")], key=lambda m: int(Path(m).stem.replace("episode", "")))
        if len(hdf_members) != 50:
            raise ValueError(f"{archive}: expected 50 episode HDF5 files, found {len(hdf_members)}")
        scene_member = next((m for m in members if Path(m).name == "scene_info.json"), None)
        if scene_member is None:
            raise ValueError(f"{archive}: missing scene_info.json")
        episode_files = []
        for member in hdf_members:
            out = target / f"episode_{int(Path(member).stem.replace('episode','')):07d}.hdf5"
            if not out.exists():
                with zf.open(member) as src, out.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
            episode_files.append(out)
        scene = target / "scene_info.json"
        if not scene.exists():
            with zf.open(scene_member) as src, scene.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        return episode_files, scene, Path(member).parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive-root", type=Path, default=Path("data/raw/archives"))
    parser.add_argument("--output-root", type=Path, default=Path("data/processed/robotwin2_ft"))
    parser.add_argument("--manifest-root", type=Path, default=Path("outputs/robotwin_ft/manifests/official_clean_50"))
    parser.add_argument("--base-config", type=Path, default=Path("configs/robotwin2_ft/base_poses.json"))
    parser.add_argument("--native-frequency", type=float, default=15.0)
    parser.add_argument("--qdur", type=float, default=1.0)
    parser.add_argument("--num-actions", type=int, default=30)
    parser.add_argument("--static-threshold", type=float, default=1e-5)
    parser.add_argument("--tasks", nargs="*", choices=TASKS, default=list(TASKS))
    parser.add_argument("--domains", nargs="*", choices=list(DOMAINS), default=list(DOMAINS))
    parser.add_argument("--keep-extracted", action="store_true")
    args = parser.parse_args()
    if args.qdur != 1.0 or args.num_actions != 30:
        raise SystemExit("This experiment freezes --qdur=1.0 and --num-actions=30")
    if args.native_frequency <= 0 or args.static_threshold < 0:
        raise SystemExit("invalid frequency/threshold")
    project_root = _project_root()
    base_config = _read_base_config(args.base_config)
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.manifest_root.mkdir(parents=True, exist_ok=True)
    pairs, all_windows = [], []
    for task in args.tasks:
        for domain in args.domains:
            archive = args.archive_root / task / f"{ {'aloha-agilex':'aloha-agilex','arx-x5':'arx-x5','piper':'piper'}[domain] }_clean_50.zip"
            if not archive.exists():
                raise FileNotFoundError(f"missing archive {archive}; run scripts/download_robotwin2_assets_data.py first")
            pair_root = args.output_root / domain / task
            extract_root = (pair_root / "_native_extract") if args.keep_extracted else Path(tempfile.mkdtemp(prefix=f"robotwin2_{domain}_{task}_", dir=args.output_root))
            try:
                native_files, scene_file, _ = _extract_archive(archive, domain, extract_root)
                scene_info = json.loads(scene_file.read_text(encoding="utf-8"))
                pair_dir = pair_root / "data"
                pair_dir.mkdir(parents=True, exist_ok=True)
                instructions_all = []
                pair_windows = []
                terminal_total = 0
                candidate_total = 0
                for native, idx in zip(native_files, range(50)):
                    # Archive instructions are kept outside the HDF5 in this
                    # older schema.  The HDF5 itself is the source of truth for
                    # observations; use the matching instruction file when it
                    # exists, otherwise fail rather than inventing language.
                    with zipfile.ZipFile(archive) as zf:
                        wanted = [m for m in zf.namelist() if Path(m).name == f"episode{idx}.json"]
                        if not wanted:
                            raise FileNotFoundError(f"{archive}: missing episode{idx}.json")
                        raw_instruction = json.loads(zf.read(wanted[0]).decode("utf-8"))
                    instruction_values = [str(v).strip() for v in raw_instruction.get("seen", []) if str(v).strip()]
                    if not instruction_values:
                        raise ValueError(f"{archive}: episode{idx} has no seen instruction")
                    out_h5 = pair_dir / f"episode_{idx:07d}.hdf5"
                    _write_normalized(native, out_h5, instruction_values, args.native_frequency, base_config[domain]["left"], base_config[domain]["right"], Path(_portable_path(archive, project_root)))
                    entries, candidate_count, terminal_count = _window_entries(out_h5, task, domain, instruction_values, base_config[domain]["left"], base_config[domain]["right"], args.num_actions, args.qdur, args.static_threshold, project_root)
                    pair_windows.extend(entries)
                    candidate_total += candidate_count
                    terminal_total += terminal_count
                    instructions_all.append(instruction_values)
                pair_manifest = {
                    "dataset_name": DOMAINS[domain]["robot_type"],
                    "robot_type": DOMAINS[domain]["robot_type"],
                    "domain_id": DOMAINS[domain]["domain_id"],
                    "embodiment": DOMAINS[domain]["embodiment"],
                    "task_names": [task],
                    "task": task,
                    "source": "official",
                    "source_archive": _portable_path(archive, project_root),
                    "source_schema": "RoboTwin native HDF5 normalized with official envs/utils/pkl2hdf5.py mapping",
                    "robowin_commit": "96c1fea",
                    "xpolicylab_commit": "c37109c",
                    "base_pose_source": "assets/robotwin/embodiments",
                    "base_pose_left": base_config[domain]["left"],
                    "base_pose_right": base_config[domain]["right"],
                    "fps_source": "native archive; official pkl2hdf5 default",
                    "frequency_hz": args.native_frequency,
                    "qdur_sec": args.qdur,
                    "num_actions": args.num_actions,
                    "camera_keys": ["vision/cam_head/colors", "vision/cam_left_wrist/colors", "vision/cam_right_wrist/colors"],
                    "image_shape": [240, 320, 3],
                    "image_decoder": "RoboTwin/data/decode_image_bit.py::decode_image_bit",
                    "quaternion_convention": "scalar_first_wxyz",
                    "action_representation": "absolute_eef6d_robot_base",
                    "instruction_dir": "archive-internal instructions/episodeN.json",
                    "datalist": [_portable_path(pair_dir / f"episode_{idx:07d}.hdf5", project_root) for idx in range(50)],
                    "episodes": 50,
                    "num_observation_frames": candidate_total,
                    "num_action_observation_windows": len(pair_windows),
                    "num_terminal_hold_windows": sum(1 for e in pair_windows if e["terminal_hold"]),
                    "terminal_hold_candidates": terminal_total,
                    "static_filter_threshold_m": args.static_threshold,
                    "split": "train",
                }
                pair_json = args.manifest_root / f"{domain}__{task}.json"
                pair_json.write_text(json.dumps(pair_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                windows_jsonl = args.manifest_root / f"{domain}__{task}.windows.jsonl"
                with windows_jsonl.open("w", encoding="utf-8") as handle:
                    for entry in pair_windows:
                        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                pairs.append(pair_manifest)
                all_windows.extend(pair_windows)
                print(f"[ok] {domain}/{task}: episodes=50 candidates={candidate_total} windows={len(pair_windows)} terminal={pair_manifest['num_terminal_hold_windows']}", flush=True)
            finally:
                if not args.keep_extracted and extract_root.exists():
                    shutil.rmtree(extract_root)
    # Sampling probabilities are derived from action-observation windows, not
    # archive/episode counts.  Save both mandated strategies in the total manifest.
    n_dt = {f"{p['domain_id']}::{p['task']}": p["num_action_observation_windows"] for p in pairs}
    n_d = {str(d): sum(v for k, v in n_dt.items() if k.startswith(f"{d}::")) for d in range(3)}
    balanced = {k: 1.0 / 15.0 for k in n_dt}
    alpha = 0.5
    domain_norm = sum(v ** alpha for v in n_d.values())
    tempered = {}
    for key, value in n_dt.items():
        d = key.split("::", 1)[0]
        task_norm = sum(v ** alpha for k, v in n_dt.items() if k.startswith(f"{d}::"))
        tempered[key] = (n_d[d] ** alpha / domain_norm) * (value ** alpha / task_norm)
    total_manifest = {
        "dataset_name": "robotwin2_ft_official_clean_50",
        "source": "official",
        "robowin_commit": "96c1fea",
        "xpolicylab_commit": "c37109c",
        "tasks": list(args.tasks),
        "domains": DOMAINS,
        "qdur_sec": args.qdur,
        "num_actions": args.num_actions,
        "camera_keys": ["vision/cam_head/colors", "vision/cam_left_wrist/colors", "vision/cam_right_wrist/colors"],
        "quaternion_convention": "scalar_first_wxyz",
        "action_representation": "absolute_eef6d_robot_base",
        "pair_manifests": [_portable_path(args.manifest_root / f"{domain}__{task}.json", project_root) for domain in args.domains for task in args.tasks],
        "pairs": pairs,
        "N_dt": n_dt,
        "N_d": n_d,
        "sampling": {
            "domain_balanced": {"formula": "p(d,t)=1/15", "probabilities": balanced},
            "tempered_T2": {"temperature": 2.0, "alpha": 0.5, "formula": "p(d)=N_d^0.5/sum N_j^0.5; p(t|d)=N_dt^0.5/sum N_du^0.5", "probabilities": tempered},
        },
        "windows_jsonl": [_portable_path(args.manifest_root / f"{domain}__{task}.windows.jsonl", project_root) for domain in args.domains for task in args.tasks],
    }
    (args.manifest_root / "total.json").write_text(json.dumps(total_manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {args.manifest_root / 'total.json'} with {len(all_windows)} windows", flush=True)


if __name__ == "__main__":
    main()
