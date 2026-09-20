#!/usr/bin/env python3
"""Download the fixed RoboTwin2.0 training archives and embodiment assets.

The files are kept under this repository (never written into the external
RoboTwin checkout).  The defaults are the 15 clean archives frozen in
``docs/experiment_design.md`` plus the official ``embodiments.zip`` asset
bundle.  A small JSON record is written after each successful download so a
partially completed run can be resumed and audited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from huggingface_hub import hf_hub_download


REPO_ID = "TianxingChen/RoboTwin2.0"
REVISION = "981c92aa34d8f94d4cff47e0d5bc2f7d4e0af042"
TASKS = (
    "beat_block_hammer",
    "stack_blocks_two",
    "move_can_pot",
    "open_microwave",
    "place_dual_shoes",
)
ARCHIVE_NAMES = (
    "aloha-agilex_clean_50.zip",
    "arx-x5_clean_50.zip",
    "piper_clean_50.zip",
)


def _safe_name(value: str) -> str:
    if not value or value in {".", ".."} or re.fullmatch(r"[A-Za-z0-9_.-]+", value) is None:
        raise ValueError(f"unsafe path component: {value!r}")
    return value


def _files() -> Iterable[tuple[str, str, Path]]:
    yield "embodiments.zip", "embodiments.zip", Path("assets") / "embodiments.zip"
    for task in TASKS:
        _safe_name(task)
        for archive in ARCHIVE_NAMES:
            _safe_name(archive)
            rel = f"dataset/{task}/{archive}"
            yield rel, rel, Path("data/raw/archives") / task / archive


def record_file(remote: str, local: str, path: Path, status: str) -> dict:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    lock = json.loads((Path(__file__).resolve().parents[1] / "configs/robotwin2_ft/assets_lock.json").read_text())
    expected = lock[remote]
    if path.stat().st_size != expected["size"] or digest.hexdigest() != expected["sha256"]:
        raise ValueError(f"Checksum mismatch: {path}; preserve/replace this file before retrying")
    return {"remote": remote, "local": local, "status": status,
            "size": path.stat().st_size, "sha256": digest.hexdigest()}


def _download_one(repo_id: str, revision: str, remote: str, destination: Path, output_root: Path, force: bool) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = destination.resolve().relative_to(output_root.resolve()).as_posix()
    if destination.exists() and destination.stat().st_size > 0 and not force:
        return record_file(remote, local, destination, "exists")
    # hf_hub_download writes atomically in its cache/local_dir; copy the
    # completed path to our explicit project-owned output only afterwards.
    cache_dir = destination.parent / ".hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        filename=remote,
        local_dir=str(cache_dir),
        force_download=force,
    )
    source = Path(path)
    os.replace(source, destination)
    return record_file(remote, local, destination, "downloaded")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("."))
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--revision", default=REVISION)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--asset-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    if args.asset_only and args.data_only:
        parser.error("--asset-only and --data-only are mutually exclusive")

    root = args.output_root.resolve()
    items = list(_files())
    if args.asset_only:
        items = items[:1]
    elif args.data_only:
        items = items[1:]
    jobs = [(remote, remote, root / local) for remote, _, local in items]
    records: list[dict] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_download_one, args.repo_id, args.revision, remote, local, root, args.force): remote
            for remote, _, local in jobs
        }
        for future in as_completed(futures):
            remote = futures[future]
            try:
                record = future.result()
                print(f"[{record['status']}] {remote} ({record['size']} bytes)", flush=True)
                records.append(record)
            except Exception:
                print(f"[failed] {remote}", flush=True)
                raise

    records.sort(key=lambda item: item["remote"])
    audit = {
        "repo_id": args.repo_id,
        "revision": args.revision,
        "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": records,
    }
    out = root / "data/raw/download_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
