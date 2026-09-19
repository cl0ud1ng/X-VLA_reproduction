#!/usr/bin/env python3
"""Stage-B sample and sampler preflight for the RoboTwin2 FT manifest."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader


def _max_sampling_error(probabilities: dict[str, float], draws: int, seed: int) -> float:
    keys = sorted(probabilities)
    p = np.asarray([probabilities[k] for k in keys], dtype=np.float64)
    sample = np.random.default_rng(seed).choice(len(keys), size=draws, p=p)
    freq = np.bincount(sample, minlength=len(keys)) / float(draws)
    return float(np.max(np.abs(freq - p)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("outputs/robotwin_ft/manifests/official_clean_50/total.json"))
    ap.add_argument("--samples-per-pair", type=int, default=2)
    ap.add_argument("--draws", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", type=Path, default=Path("outputs/robotwin_ft/preflight_stage_b.json"))
    args = ap.parse_args()
    if args.samples_per_pair < 1 or args.draws < 10000:
        raise SystemExit("Stage B requires at least one sample/pair and 10,000 sampler draws")

    # Import after argument parsing so --help works without torch in a docs env.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from datasets.dataset import ManifestWindowDataset

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    report = {
        "manifest": str(args.manifest.resolve()),
        "seed": args.seed,
        "draws": args.draws,
        "sample_contract": {"image_input": [3, 3, 224, 224], "image_mask": [3], "proprio": [20], "action": [30, 20]},
        "pairs": {},
        "batch_domain_counts": {},
        "sampling": {},
    }
    decoded = []
    # Validate every pair's transformed sample and explicitly include a
    # terminal-hold entry whenever the pair has one.
    for mode in ("domain_balanced", "tempered_T2"):
        ds = ManifestWindowDataset(args.manifest, training=False, sampler_mode=mode, base_seed=args.seed)
        for key in ds._pair_keys:
            entries = ds._entries[key]
            chosen = entries[:args.samples_per_pair]
            terminal = next((e for e in reversed(entries) if e.get("terminal_hold")), None)
            if terminal is not None and terminal not in chosen:
                chosen.append(terminal)
            pair_report = report["pairs"].setdefault(key, {"manifest_windows": len(entries), "checked": 0, "terminal_hold_checked": False})
            for entry in chosen:
                sample = ds._yield_entry(entry)
                expected = {
                    "image_input": (3, 3, 224, 224),
                    "image_mask": (3,),
                    "proprio": (20,),
                    "action": (30, 20),
                }
                for name, shape in expected.items():
                    got = tuple(sample[name].shape)
                    if got != shape:
                        raise AssertionError(f"{key}: {name} shape {got} != {shape}")
                if sample["image_mask"].tolist() != [True, True, True]:
                    raise AssertionError(f"{key}: image mask is not all true")
                for name in ("proprio", "action"):
                    if not torch.isfinite(sample[name]).all():
                        raise AssertionError(f"{key}: non-finite {name}")
                if torch.any((sample["action"][:, (9, 19)] < 0) | (sample["action"][:, (9, 19)] > 1)):
                    raise AssertionError(f"{key}: gripper target outside [0,1]")
                if bool(entry.get("terminal_hold")):
                    pair_report["terminal_hold_checked"] = True
                pair_report["checked"] += 1
                decoded.append(sample)
        probabilities = ds.probabilities
        err = _max_sampling_error(probabilities, args.draws, args.seed)
        report["sampling"][mode] = {"probabilities": probabilities, "max_abs_error": err}
        if err >= 0.02:
            raise AssertionError(f"{mode}: max sampling error {err} >= 0.02")

    # A real collated batch must mix domains.  Use six checked samples (two per
    # domain) so this check is deterministic and independent of random luck.
    mixed = []
    for domain in (0, 1, 2):
        mixed.extend([s for s in decoded if int(s["domain_id"]) == domain][:args.samples_per_pair])
    batch = next(iter(DataLoader(mixed, batch_size=len(mixed), num_workers=0)))
    counts = Counter(int(x) for x in batch["domain_id"].tolist())
    report["batch_domain_counts"] = {str(k): int(v) for k, v in sorted(counts.items())}
    if set(counts) != {0, 1, 2}:
        raise AssertionError(f"batch does not contain all domains: {counts}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"pairs": len(report["pairs"]), "checked_samples": len(decoded), "batch_domain_counts": report["batch_domain_counts"], "sampling": report["sampling"], "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
