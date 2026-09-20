#!/usr/bin/env python3
"""Verify that a freshly prepared training bundle is clone-portable."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def assert_portable(value: object, where: str) -> None:
    if isinstance(value, str):
        if value.startswith("/") or "/mnt/" in value or "/home/" in value:
            raise ValueError(f"absolute host path in {where}: {value}")
    elif isinstance(value, dict):
        for key, item in value.items():
            assert_portable(item, f"{where}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            assert_portable(item, f"{where}[{index}]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("outputs/robotwin_ft/manifests/official_clean_50/total.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/X-VLA-Pt"))
    args = parser.parse_args()
    root = project_root()
    manifest_path = args.manifest if args.manifest.is_absolute() else root / args.manifest
    model_dir = args.model_dir if args.model_dir.is_absolute() else root / args.model_dir
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert_portable(manifest, str(manifest_path))
    if manifest.get("qdur_sec") != 1.0 or manifest.get("num_actions") != 30:
        raise ValueError("manifest action contract is not qdur=1.0/30 actions")
    if set(manifest.get("tasks", [])) != {
        "beat_block_hammer", "stack_blocks_two", "move_can_pot", "open_microwave", "place_dual_shoes"
    }:
        raise ValueError("manifest task set is incomplete")
    if len(manifest.get("pairs", [])) != 15:
        raise ValueError("manifest must contain 15 domain-task pairs")
    window_count = 0
    checked_paths = set()
    domain_names = {0: "aloha-agilex", 1: "arx-x5", 2: "piper"}
    for pair in manifest["pairs"]:
        index = manifest_path.parent / f"{domain_names[int(pair['domain_id'])]}__{pair['task']}.windows.jsonl"
        if not index.exists():
            raise FileNotFoundError(index)
        rows = [json.loads(line) for line in index.read_text(encoding="utf-8").splitlines() if line.strip()]
        window_count += len(rows)
        for row in rows:
            assert_portable(row, f"{index}:{row.get('frame_idx')}")
            h5 = Path(row["hdf5_path"])
            h5_path = h5 if h5.is_absolute() else root / h5
            if ".." in h5.parts:
                raise ValueError(f"Path escapes project: {h5}")
            if h5_path not in checked_paths:
                if not h5_path.is_file():
                    raise FileNotFoundError(h5)
                checked_paths.add(h5_path)
    if window_count != sum(int(value) for value in manifest["N_dt"].values()):
        raise ValueError("window index count does not match manifest N_dt")
    required_model = ("config.json", "model.safetensors", "tokenizer.json", "preprocessor_config.json")
    missing = [name for name in required_model if not (model_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing model files: {missing}")
    print(json.dumps({"status": "pass", "windows": window_count, "model_dir": str(model_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
