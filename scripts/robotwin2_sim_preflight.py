#!/usr/bin/env python3
"""Run RoboTwin Stage-A setup/play preflight without modifying the checkout.

A temporary overlay is populated from the project-local RoboTwin checkout at
the frozen ``96c1fea`` commit.
The downloaded project-owned embodiment assets are mounted into that overlay,
and each task/domain is run in a fresh Python process with one seed taken from
its official clean archive.  Results and logs are written under the project.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import traceback
import zipfile
from pathlib import Path

import yaml

TASKS = ("beat_block_hammer", "stack_blocks_two", "move_can_pot", "open_microwave", "place_dual_shoes")
DOMAINS = {
    "aloha-agilex": {"domain_id": 0, "embodiment": ["aloha-agilex"], "asset": "aloha-agilex"},
    "arx-x5": {"domain_id": 1, "embodiment": ["ARX-X5", "ARX-X5", 0.60], "asset": "ARX-X5"},
    "piper": {"domain_id": 2, "embodiment": ["piper", "piper", 0.60], "asset": "piper"},
}


def first_official_seed(archive: Path) -> int:
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if Path(name).name == "seed.txt")
        values = zf.read(member).decode("utf-8").split()
    if not values:
        raise ValueError(f"official archive has empty seed.txt: {archive}")
    return int(values[0])


def archive_fixed_code(robotwin_root: Path, overlay: Path) -> None:
    command = ["git", "-C", str(robotwin_root), "archive", "--format=tar", "96c1fea", "envs", "description", "env_cfg"]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=__import__("io").BytesIO(result.stdout), mode="r:") as tar:
        tar.extractall(overlay)


def prepare_overlay(robotwin_root: Path, project_root: Path, temp_root: Path) -> Path:
    archive_fixed_code(robotwin_root, temp_root)
    assets = temp_root / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    project_assets = project_root / "assets/robotwin/embodiments"
    runtime_assets = assets / "embodiments"
    runtime_assets.mkdir(parents=True, exist_ok=True)
    # Keep the downloaded bundle immutable.  Each runtime embodiment directory
    # consists of symlinks plus generated Curobo yml files in the temporary
    # overlay, matching the official update_embodiment_config_path.py result.
    for asset_name in ("aloha-agilex", "ARX-X5", "piper"):
        source_dir = project_assets / asset_name
        target_dir = runtime_assets / asset_name
        target_dir.mkdir(parents=True, exist_ok=True)
        for child in source_dir.iterdir():
            (target_dir / child.name).symlink_to(child, target_is_directory=child.is_dir())
        for template in source_dir.glob("*_tmp.yml"):
            content = template.read_text(encoding="utf-8").replace("${ASSETS_PATH}", str(temp_root.resolve()))
            (target_dir / template.name.replace("_tmp.yml", ".yml")).write_text(content, encoding="utf-8")
    for name in ("objects", "files", "background_texture"):
        source = robotwin_root / "assets" / name
        if source.exists():
            (assets / name).symlink_to(source, target_is_directory=True)
    task_cfg = temp_root / "env_cfg/task_config"
    task_cfg.mkdir(parents=True, exist_ok=True)
    # Fixed code reads this mapping and resolves the two-arm form itself.
    mapping = {
        "aloha-agilex": {"file_path": str((runtime_assets / "aloha-agilex").resolve())},
        "ARX-X5": {"file_path": str((runtime_assets / "ARX-X5").resolve())},
        "piper": {"file_path": str((runtime_assets / "piper").resolve())},
    }
    (task_cfg / "_embodiment_config.yml").write_text(yaml.safe_dump(mapping, sort_keys=False), encoding="utf-8")
    return temp_root


def extract_official_replay(archive: Path, result_root: Path) -> None:
    traj_dir = result_root / "_traj_data"
    traj_dir.mkdir(parents=True, exist_ok=True)
    target = traj_dir / "episode0.pkl"
    with zipfile.ZipFile(archive) as zf:
        member = next(name for name in zf.namelist() if Path(name).name == "episode0.pkl")
        if not target.exists():
            target.write_bytes(zf.read(member))


def write_config(overlay: Path, robotwin_root: Path, task: str, domain: str, result_root: Path) -> None:
    fixed = subprocess.run(
        ["git", "-C", str(robotwin_root), "show", "96c1fea:env_cfg/task_config/demo_clean.yml"],
        check=True, stdout=subprocess.PIPE, text=True,
    ).stdout
    config = yaml.safe_load(fixed)
    config.update({
        "episode_num": 1,
        "use_seed": True,
        "embodiment": DOMAINS[domain]["embodiment"],
        "render_freq": 0,
        "collect_data": False,
        "save_data": False,
        "eval_video_log": False,
        "save_path": str(result_root.resolve()),
    })
    (overlay / "env_cfg/task_config/preflight.yml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def worker(args: argparse.Namespace) -> int:
    overlay = Path(args.overlay).resolve()
    result_path = Path(args.result).resolve()
    result_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "task": args.task, "domain": args.domain, "domain_id": DOMAINS[args.domain]["domain_id"],
        "embodiment": DOMAINS[args.domain]["embodiment"], "seed": args.seed,
        "robowin_commit": "96c1fea", "setup_demo": False, "play_once": False,
        "plan_success": False, "check_success": False, "camera_names": [],
        "arm_dims": [], "gripper_dims": [], "status": "failed",
    }
    env = None
    try:
        os.chdir(overlay)
        sys.path.insert(0, str(overlay))
        import importlib
        from envs._GLOBAL_CONFIGS import CONFIGS_PATH
        config = yaml.safe_load((Path(CONFIGS_PATH) / "preflight.yml").read_text(encoding="utf-8"))
        embodiment = DOMAINS[args.domain]["embodiment"]
        mapping = yaml.safe_load((Path(CONFIGS_PATH) / "_embodiment_config.yml").read_text(encoding="utf-8"))
        if len(embodiment) == 1:
            left_file = right_file = mapping[embodiment[0]]["file_path"]
            dual = True
        else:
            left_file = mapping[embodiment[0]]["file_path"]
            right_file = mapping[embodiment[1]]["file_path"]
            dual = False
        def read_robot_config(path: str) -> dict:
            return yaml.safe_load((Path(path) / "config.yml").read_text(encoding="utf-8"))
        config.update({
            "task_name": args.task, "task_config": "preflight",
            "left_robot_file": left_file, "right_robot_file": right_file,
            "dual_arm_embodied": dual,
            "embodiment_name": str(embodiment[0]) if len(embodiment) == 1 else f"{embodiment[0]}+{embodiment[1]}",
            "left_embodiment_config": read_robot_config(left_file),
            "right_embodiment_config": read_robot_config(right_file),
            "embodiment_dis": embodiment[2] if len(embodiment) == 3 else None,
            "need_plan": True, "eval_mode": True,
        })
        task_module = importlib.import_module(f"envs.{args.task}")
        env = getattr(task_module, args.task)()
        env.setup_demo(now_ep_num=0, seed=args.seed, is_test=True, **config)
        record["setup_demo"] = True
        record["camera_names"] = list(getattr(env.cameras, "static_camera_name", [])) + ["left_camera", "right_camera"]
        record["arm_dims"] = [len(getattr(env.robot, "left_arm_joints", [])), len(getattr(env.robot, "right_arm_joints", []))]
        # RoboTwin exposes two physical gripper joints per mimic profile, but
        # the task/control contract is one scalar gripper per arm.
        record["gripper_dims"] = [1, 1]
        traj_data = env.load_tran_data(0)
        config["need_plan"] = False
        config["left_joint_path"] = traj_data["left_joint_path"]
        config["right_joint_path"] = traj_data["right_joint_path"]
        env.set_path_lst(config)
        record["replay_official_trajectory"] = True
        env.play_once()
        record["play_once"] = True
        record["plan_success"] = bool(env.plan_success)
        record["check_success"] = bool(env.check_success())
        record["status"] = "pass" if record["plan_success"] and record["check_success"] else "fail"
    except Exception as exc:
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            try: env.close_env(clear_cache=True)
            except Exception as exc: record["close_error"] = f"{type(exc).__name__}: {exc}"
        result_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return 0 if record["status"] == "pass" else 1


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--result-root", type=Path, default=Path("outputs/robotwin_ft/preflight_stage_a"))
    parser.add_argument("--task", choices=TASKS)
    parser.add_argument("--domain", choices=list(DOMAINS))
    parser.add_argument("--tasks", nargs="*", choices=TASKS, default=list(TASKS))
    parser.add_argument("--domains", nargs="*", choices=list(DOMAINS), default=list(DOMAINS))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--overlay")
    parser.add_argument("--result")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if args.worker:
        return worker(args)
    project_root = args.project_root.resolve()
    robotwin_root = project_root / "third_party" / "RoboTwin"
    if not robotwin_root.is_dir():
        raise FileNotFoundError(
            f"RoboTwin checkout not found: {robotwin_root}; run scripts/bootstrap_robotwin2.sh"
        )
    result_root = (project_root / args.result_root).resolve() if not args.result_root.is_absolute() else args.result_root.resolve()
    result_root.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="robotwin2_stage_a_", dir=result_root) as temp:
        overlay = prepare_overlay(robotwin_root, project_root, Path(temp))
        for domain in args.domains:
            spec = DOMAINS[domain]
            for task in args.tasks:
                archive = project_root / "data/raw/archives" / task / f"{ {'aloha-agilex':'aloha-agilex','arx-x5':'arx-x5','piper':'piper'}[domain] }_clean_50.zip"
                seed = first_official_seed(archive)
                pair_root = result_root / f"{domain}__{task}"
                extract_official_replay(archive, pair_root)
                write_config(overlay, robotwin_root, task, domain, pair_root)
                result = result_root / f"{domain}__{task}.json"
                log = result_root / f"{domain}__{task}.log"
                env = os.environ.copy(); env.update({"PYTHONPATH": str(overlay), "PYOPENGL_PLATFORM": "egl"})
                torch_lib = project_root / ".venv/lib/python3.10/site-packages/torch/lib"
                local_cuda_lib = project_root / ".cache/cuda/lib"
                env["LD_LIBRARY_PATH"] = ":".join(str(p) for p in (torch_lib, local_cuda_lib) if p.exists()) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
                command = [sys.executable, str(Path(__file__).resolve()), "--worker", "--overlay", str(overlay), "--task", task, "--domain", domain, "--seed", str(seed), "--result", str(result)]
                started = time.monotonic()
                completed = subprocess.run(command, cwd=overlay, env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
                log.write_text(completed.stdout, encoding="utf-8")
                try: row = json.loads(result.read_text())
                except Exception: row = {"task":task,"domain":domain,"seed":seed,"status":"runner_error","returncode":completed.returncode,"log":str(log.resolve())}
                row.update({"elapsed_sec": round(time.monotonic()-started, 3), "log": str(log.resolve()), "archive": str(archive.resolve())})
                result.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
                rows.append(row)
                print(f"[{row.get('status')}] {domain}/{task} seed={seed}", flush=True)
    summary = {"robowin_commit":"96c1fea","tasks":list(args.tasks),"domains":{d:DOMAINS[d] for d in args.domains},"results":rows,"passed":sum(r.get("status")=="pass" for r in rows),"total":len(rows)}
    (result_root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0 if summary["passed"] == summary["total"] else 1

if __name__ == "__main__":
    raise SystemExit(main())
