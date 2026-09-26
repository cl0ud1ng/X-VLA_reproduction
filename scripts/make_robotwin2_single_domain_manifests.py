#!/usr/bin/env python3
"""Create auditable single-embodiment views of the RoboTwin total manifest.

The normalized HDF5 files and window indexes remain shared with the official
15-pair manifest.  Each generated manifest keeps exactly one domain and all
five frozen tasks, and recomputes the balanced/tempered sampling probabilities
over that domain's five task pairs.  No data are copied or rewritten.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


DOMAIN_NAMES = {0: "aloha-agilex", 1: "arx-x5", 2: "piper"}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _rel(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _single_manifest(total: dict, domain_id: int, total_path: Path, root: Path) -> dict:
    pairs = [p for p in total.get("pairs", [])
             if int(p.get("domain_id", -1)) == domain_id and p.get("split") == "train"]
    tasks = list(total.get("tasks", []))
    by_task = {str(p["task"]): p for p in pairs}
    if set(by_task) != set(tasks) or len(pairs) != len(tasks):
        raise ValueError(f"domain {domain_id} does not have exactly the frozen task set")

    n_dt = {f"{domain_id}::{task}": int(by_task[task]["num_action_observation_windows"])
            for task in tasks}
    n_d = {str(domain_id): sum(n_dt.values())}
    balanced = {key: 1.0 / len(tasks) for key in n_dt}
    alpha = 0.5
    denom = sum(value ** alpha for value in n_dt.values())
    tempered = {key: (value ** alpha) / denom for key, value in n_dt.items()}

    out_pairs = []
    for task in tasks:
        pair = dict(by_task[task])
        original_index = pair.get("windows_path")
        if original_index is None:
            original_index = f"outputs/robotwin_ft/manifests/official_clean_50/{DOMAIN_NAMES[domain_id]}__{task}.windows.jsonl"
        pair["windows_path"] = original_index
        out_pairs.append(pair)

    domains = {name: value for name, value in total.get("domains", {}).items()
               if int(value.get("domain_id", -1)) == domain_id}
    if not domains:
        raise ValueError(f"total manifest has no domain metadata for {domain_id}")
    return {
        "dataset_name": f"robotwin2_ft_single_{DOMAIN_NAMES[domain_id]}",
        "source": total.get("source", "official"),
        "source_total_manifest": _rel(total_path, root),
        "single_domain": True,
        "domain_id": domain_id,
        "domain_name": DOMAIN_NAMES[domain_id],
        "robowin_commit": total.get("robowin_commit"),
        "xpolicylab_commit": total.get("xpolicylab_commit"),
        "tasks": tasks,
        "domains": domains,
        "qdur_sec": total.get("qdur_sec"),
        "num_actions": total.get("num_actions"),
        "camera_keys": total.get("camera_keys"),
        "quaternion_convention": total.get("quaternion_convention"),
        "action_representation": total.get("action_representation"),
        "pair_manifests": [
            f"outputs/robotwin_ft/manifests/official_clean_50/{DOMAIN_NAMES[domain_id]}__{task}.json"
            for task in tasks
        ],
        "pairs": out_pairs,
        "N_dt": n_dt,
        "N_d": n_d,
        "sampling": {
            "domain_balanced": {
                "formula": f"p(domain={domain_id})=1; p(task|domain)=1/{len(tasks)}",
                "probabilities": balanced,
            },
            "tempered_T2": {
                "temperature": 2.0,
                "alpha": alpha,
                "formula": "p(task|domain)=N_dt^0.5/sum_u N_du^0.5",
                "probabilities": tempered,
            },
        },
        "windows_jsonl": [p["windows_path"] for p in out_pairs],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=Path,
                        default=Path("outputs/robotwin_ft/manifests/official_clean_50/total.json"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("outputs/robotwin_ft/manifests/single_domain"))
    parser.add_argument("--domains", nargs="*", type=int, choices=(0, 1, 2), default=[0, 1, 2])
    args = parser.parse_args()
    root = _project_root()
    total_path = args.total if args.total.is_absolute() else root / args.total
    output_dir = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    total = json.loads(total_path.read_text(encoding="utf-8"))
    if total.get("qdur_sec") != 1.0 or total.get("num_actions") != 30:
        raise ValueError("total manifest is not the frozen qdur=1.0/30 contract")
    output_dir.mkdir(parents=True, exist_ok=True)
    for domain_id in args.domains:
        manifest = _single_manifest(total, domain_id, total_path, root)
        path = output_dir / f"{DOMAIN_NAMES[domain_id]}.json"
        path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[ok] domain={domain_id} tasks={len(manifest['pairs'])} windows={sum(manifest['N_dt'].values())} -> {path}")


if __name__ == "__main__":
    main()
