#!/usr/bin/env python3
"""Structured RoboTwin-2.0 rollout scheduler.

The scheduler is deliberately independent of a particular GPU count or task
matrix. It creates one sequential worker per selected GPU and distributes the
selected domain/task cells deterministically across those workers. The
8-GPU/main-protocol defaults live in ``run_robotwin2_rollout_8GPU_exec1.sh``.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


DEFAULT_TASKS = (
    "beat_block_hammer",
    "stack_blocks_two",
    "move_can_pot",
    "open_microwave",
    "place_dual_shoes",
)
DEFAULT_DOMAINS = ("aloha-agilex", "ARX-X5", "piper-dual")
DOMAIN_IDS = {"aloha-agilex": 0, "ARX-X5": 1, "piper-dual": 2}


def _git_rev(path: Path) -> str | None:
    try:
        return subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _csv_values(value: str | None, *, name: str) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name} cannot be empty")
    return values


def _select_values(
    explicit: str | None,
    defaults: Sequence[str],
    count: int | None,
    *,
    name: str,
) -> list[str]:
    values = _csv_values(explicit, name=name) or list(defaults)
    if len(set(values)) != len(values):
        raise ValueError(f"{name} contains duplicates: {values}")
    unknown = sorted(set(values) - set(defaults))
    if unknown:
        raise ValueError(f"unknown {name}: {unknown}; choices are {list(defaults)}")
    if count is not None:
        if count < 1 or count > len(values):
            raise ValueError(f"num_{name} must be in [1,{len(values)}], got {count}")
        values = values[:count]
    return values


def _select_gpus(gpus_arg: str | None, num_gpus: int | None) -> list[str]:
    if gpus_arg:
        gpus = _csv_values(gpus_arg, name="gpus") or []
        if num_gpus is not None and len(gpus) != num_gpus:
            raise ValueError(f"--gpus has {len(gpus)} entries but --num-gpus={num_gpus}")
    else:
        count = 1 if num_gpus is None else num_gpus
        if count < 1:
            raise ValueError(f"num_gpus must be positive, got {count}")
        gpus = [str(index) for index in range(count)]
    if len(set(gpus)) != len(gpus):
        raise ValueError(f"gpus contains duplicates: {gpus}")
    return gpus


def _run_gpu(
    gpu: str,
    cells: list[tuple[str, str]],
    args: argparse.Namespace,
    client: Path,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for domain, task in cells:
        cell_dir = args.output_dir / domain / task
        cell_dir.mkdir(parents=True, exist_ok=True)
        log_path = cell_dir / "worker.log"
        command = [
            args.python,
            str(client),
            "--host", args.host,
            "--port", str(args.port),
            "--task_name", task,
            "--domain", domain,
            "--task_config", args.task_config,
            "--num_episodes", str(args.num_episodes),
            "--seed", str(args.seed),
            "--output_path", str(cell_dir),
            "--exec_points", str(args.exec_points),
            "--max_steps", str(args.max_steps),
            "--device", "0",
        ]
        if args.save_video:
            command.append("--save_video")
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        env["PYTHONUNBUFFERED"] = "1"
        started = datetime.now(timezone.utc).isoformat()
        with log_path.open("w", encoding="utf-8") as log:
            log.write("$ " + shlex.join(command) + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=client.parents[1],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        summary_path = cell_dir / "summary.json"
        result: dict[str, Any] = {
            "gpu": int(gpu) if gpu.isdigit() else gpu,
            "domain": domain,
            "task": task,
            "returncode": completed.returncode,
            "started_at": started,
            "log": str(log_path),
            "summary": str(summary_path),
        }
        if summary_path.is_file():
            try:
                result["success_rate"] = json.loads(summary_path.read_text())["success_rate"]
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        results.append(result)
    return results


def _parser() -> argparse.ArgumentParser:
    project_default = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=project_default)
    parser.add_argument("--python", default=os.environ.get("XVLA_PYTHON", str(project_default / ".venv/bin/python")))
    parser.add_argument("--host", default=os.environ.get("MODEL_HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("MODEL_PORT", "8000")))
    parser.add_argument("--gpus", default=os.environ.get("ROLLOUT_GPUS"), help="comma-separated physical GPU ids")
    parser.add_argument("--num-gpus", type=int, default=None, help="number of GPUs; uses 0..N-1 when --gpus is omitted")
    parser.add_argument("--tasks", default=os.environ.get("ROLLOUT_TASKS"), help="comma-separated task names")
    parser.add_argument("--num-tasks", type=int, default=None, help="use the first N tasks from --tasks or the frozen default list")
    parser.add_argument("--domains", default=os.environ.get("ROLLOUT_DOMAINS"), help="comma-separated domain names")
    parser.add_argument("--num-domains", type=int, default=None, help="use the first N domains from --domains or the frozen default list")
    parser.add_argument("--task-config", default=os.environ.get("TASK_CONFIG", "demo_clean"))
    parser.add_argument("--num-episodes", type=int, default=int(os.environ.get("NUM_EPISODES", "1")))
    parser.add_argument("--seed", type=int, default=int(os.environ.get("ROLLOUT_SEED", "0")))
    parser.add_argument("--exec-points", type=int, default=int(os.environ.get("EXEC_POINTS", "1")))
    parser.add_argument("--max-steps", type=int, default=int(os.environ.get("MAX_STEPS", "0")), help="0 uses each task's RoboTwin step limit")
    parser.add_argument("--output-dir", type=Path, default=os.environ.get("EVAL_LOG_DIR"))
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="write the schedule without starting simulators")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        args.project_root = args.project_root.resolve()
        # Keep the venv symlink intact: resolving .venv/bin/python would
        # bypass its site-packages and select the bare .python interpreter.
        args.python = str((args.project_root / args.python).absolute()) if not Path(args.python).is_absolute() else args.python
        args.output_dir = Path(args.output_dir or args.project_root / "outputs/robotwin_ft/eval_rollout").resolve()
        gpus = _select_gpus(args.gpus, args.num_gpus)
        tasks = _select_values(args.tasks, DEFAULT_TASKS, args.num_tasks, name="tasks")
        domains = _select_values(args.domains, DEFAULT_DOMAINS, args.num_domains, name="domains")
    except ValueError as exc:
        parser.error(str(exc))
    if args.num_episodes < 1:
        parser.error("--num-episodes must be positive")
    if not 1 <= args.exec_points <= 30:
        parser.error("--exec-points must be in [1,30]")
    if args.max_steps < 0:
        parser.error("--max-steps must be non-negative")
    client = args.project_root / "evaluation/robotwin-2.0/client.py"
    if not client.is_file():
        parser.error(f"client not found: {client}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = [(domain, task) for domain in domains for task in tasks]
    assignments = {gpu: cells[index::len(gpus)] for index, gpu in enumerate(gpus)}
    run = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "project_root": str(args.project_root),
        "xvla_commit": _git_rev(args.project_root),
        "robotwin_commit": _git_rev(args.project_root / "third_party/RoboTwin"),
        "gpus": gpus,
        "num_gpus": len(gpus),
        "assignments": assignments,
        "tasks": tasks,
        "num_tasks": len(tasks),
        "domains": domains,
        "num_domains": len(domains),
        "domain_ids": {domain: DOMAIN_IDS[domain] for domain in domains},
        "task_config": args.task_config,
        "num_episodes": args.num_episodes,
        "seed": args.seed,
        "exec_points": args.exec_points,
        "max_steps": args.max_steps,
        "model": {"host": args.host, "port": args.port},
        "command": list(argv) if argv is not None else sys.argv[1:],
        "protocol": {
            "qdur_sec": 1.0,
            "num_actions": 30,
            "camera_order": ["head_camera", "left_camera", "right_camera"],
            "action_frame": "per-arm robot-base -> world before take_action",
            "quaternion": "scalar-first wxyz",
            "gripper": "model closed=1; RoboTwin raw open=1",
        },
    }
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "assignments": assignments}, indent=2))
    if args.dry_run:
        return 0
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(_run_gpu, gpu, assignments[gpu], args, client) for gpu in gpus]
        for future in as_completed(futures):
            results.extend(future.result())
    results.sort(key=lambda row: (row["domain"], row["task"]))
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    failures = [row for row in results if row["returncode"] != 0]
    print(json.dumps({"cells": len(results), "failed_cells": len(failures), "results": results}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
