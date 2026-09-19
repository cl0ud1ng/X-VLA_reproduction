#!/usr/bin/env python3
"""Stage-C 100-step, multi-GPU X-VLA training smoke on RoboTwin samples.

The acceptance run is intended for the project's 8x4090 host and exercises
the full-model path during the documented freeze phase through Accelerate/DDP.
It deliberately has no reduced single-GPU fallback: a successful report must
record world_size=8.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from accelerate import Accelerator, DataLoaderConfiguration


def _decode_language(value):
    if isinstance(value, torch.Tensor):
        return [bytes(row.tolist()).split(b"\0", 1)[0].decode("utf-8") for row in value]
    return value


def _move_batch(batch, processor, device):
    language_values = _decode_language(batch["language_instruction"])
    lang = processor.encode_language(language_values)
    inputs = {}
    for key, value in {**batch, **lang}.items():
        if key in {"language_instruction", "task", "task_id", "terminal_hold", "window_key"}:
            continue
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device=device, non_blocking=True)
    return inputs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, default=Path("models/X-VLA-Pt"))
    ap.add_argument("--manifest", type=Path, default=Path("outputs/robotwin_ft/manifests/official_clean_50/total.json"))
    ap.add_argument("--output-dir", type=Path, default=Path("outputs/robotwin_ft/stage_c_smoke"))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.steps != 100:
        raise SystemExit("Stage C acceptance is frozen at exactly 100 steps")

    accelerator = Accelerator(
        mixed_precision="fp16",
        dataloader_config=DataLoaderConfiguration(split_batches=True, even_batches=True),
        log_with=None,
    )
    if accelerator.num_processes != 8:
        raise SystemExit(f"Stage C main smoke requires 8 processes, got {accelerator.num_processes}")
    torch.manual_seed(args.seed + accelerator.process_index)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed + accelerator.process_index)
    device = accelerator.device

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from datasets.dataset import ManifestWindowDataset
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    # Eight examples make split_batches=True produce one example per GPU;
    # there are at least two windows for each of the three domains.
    ds = ManifestWindowDataset(args.manifest, training=False, sampler_mode="domain_balanced", base_seed=args.seed)
    entries = []
    for domain in (0, 1, 2):
        keys = [key for key in ds._pair_keys if key.startswith(f"{domain}::")]
        entries.extend(ds._entries[keys[0]][:2])
    entries.extend(ds._entries[[key for key in ds._pair_keys if key.startswith("0::")][1]][:2])
    samples = [ds._yield_entry(entry) for entry in entries]
    expected_domains = Counter(int(sample["domain_id"]) for sample in samples)
    if any(expected_domains[d] < 2 for d in (0, 1, 2)):
        raise AssertionError(f"insufficient per-domain smoke samples: {expected_domains}")
    loader = DataLoader(samples, batch_size=8, shuffle=False, num_workers=0, drop_last=True)

    if accelerator.is_main_process:
        print(f"loading model from {args.models} on world_size={accelerator.num_processes}", flush=True)
    model = XVLA.from_pretrained(str(args.models), local_files_only=True)
    processor = XVLAProcessor.from_pretrained(str(args.models), local_files_only=True)
    if model.config.action_mode != "ee6d" or model.config.num_actions != 30 or not model.config.use_proprio:
        raise AssertionError("checkpoint violates ee6d/30/proprio contract")
    if model.config.num_domains < 3 or model.action_space.dim_action != 20:
        raise AssertionError("checkpoint lacks three domains or 20-D action space")
    # Stage C is the documented first 1,000-step freeze phase: the full
    # model participates in forward/backward, while only soft prompts and
    # action heads receive gradients/optimizer state.
    for parameter in model.vlm.parameters():
        parameter.requires_grad_(False)
    for parameter in model.transformer.parameters():
        parameter.requires_grad_(False)
    for module in (model.transformer.soft_prompt_hub, model.transformer.action_decoder, model.transformer.action_encoder):
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = AdamW(trainable, lr=1e-6, betas=(0.9, 0.95), weight_decay=0.0, foreach=False)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    model.train()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.jsonl"
    if accelerator.is_main_process:
        metrics_path.unlink(missing_ok=True)
    accelerator.wait_for_everyone()
    t_start = time.time()
    train_iter = iter(loader)
    with metrics_path.open("a", encoding="utf-8") if accelerator.is_main_process else open("/dev/null", "w") as metrics:
        for step in range(100):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                batch = next(train_iter)
            optimizer.zero_grad(set_to_none=True)
            inputs = _move_batch(batch, processor, device)
            with accelerator.autocast():
                losses = model(**inputs)
                total = sum(losses.values())
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite total loss at step {step}: {losses}")
            accelerator.backward(total)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            gathered_total = accelerator.gather(total.detach().reshape(1)).mean().item()
            gathered_tasks = accelerator.gather(batch["task_id"].detach())
            gathered_domains = accelerator.gather(batch["domain_id"].detach())
            if accelerator.is_main_process:
                task_count = Counter(int(x) for x in gathered_tasks.tolist())
                domain_count = Counter(int(x) for x in gathered_domains.tolist())
                # batch is the global batch on the main process because
                # split_batches keeps the original collated metadata there.
                record = {
                    "step": step + 1,
                    "loss_total": float(gathered_total),
                    "loss_position": float(losses["position_loss"].detach().cpu()),
                    "loss_rotate6D": float(losses["rotate6D_loss"].detach().cpu()),
                    "loss_gripper": float(losses["gripper_loss"].detach().cpu()),
                    "domain_count": {str(domain): int(domain_count.get(domain, 0)) for domain in range(3)},
                    "task_count": {str(task): int(count) for task, count in sorted(task_count.items())},
                    "lr_vlm": 0.0,
                    "lr_transformer_core": 0.0,
                    "lr_soft_prompts": 1e-6,
                    "lr_action_heads": 1e-6,
                    "world_size": accelerator.num_processes,
                }
                metrics.write(json.dumps(record) + "\n")
                metrics.flush()
                if (step + 1) % 10 == 0:
                    print(f"step={step + 1}/100 loss={record['loss_total']:.5f}", flush=True)

    # Check fixed output shape locally on every rank.
    with torch.no_grad(), accelerator.autocast():
        inference_inputs = _move_batch(batch, processor, device)
        inference_inputs.pop("action", None)
        inference_model = accelerator.unwrap_model(model)
        action = inference_model.generate_actions(**inference_inputs, steps=1)
    if tuple(action.shape[1:]) != (30, 20) or not torch.isfinite(action).all():
        raise AssertionError(f"generate_actions returned {tuple(action.shape)} on rank {accelerator.process_index}")
    accelerator.wait_for_everyone()

    checkpoint = args.output_dir / "checkpoint-100"
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(checkpoint, safe_serialization=True)
        processor.save_pretrained(checkpoint)
        repo_root = Path(__file__).resolve().parents[1]
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo_root, text=True).strip()
        git_dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=repo_root, text=True).strip())
        state = {
            "global_step": 100,
            "xvla_commit": git_commit,
            "git_dirty": git_dirty,
            "base_checkpoint": str(args.models.resolve()),
            "manifest": str(args.manifest.resolve()),
            "sampler_mode": "domain_balanced",
            "seed": args.seed,
            "device": str(device),
            "world_size": accelerator.num_processes,
            "global_batch_size": 8,
            "mixed_precision": "fp16",
            "freeze_steps": 1000,
            "trainable_groups": ["soft_prompts", "action_heads"],
            "output_shape": [8, 30, 20],
            "domain_counts": {str(k): int(v) for k, v in expected_domains.items()},
        }
        (checkpoint / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        # Reload on the main rank and verify processor/domain embeddings survive.
        reloaded = XVLA.from_pretrained(str(checkpoint), local_files_only=True).to(device)
        reloaded_processor = XVLAProcessor.from_pretrained(str(checkpoint), local_files_only=True)
        soft_prompt = getattr(reloaded.transformer, "soft_prompt_hub", None)
        if soft_prompt is None or soft_prompt.num_embeddings < 3:
            raise AssertionError("reloaded checkpoint is missing domain soft-prompt embeddings")
        if reloaded_processor.tokenizer is None or reloaded_processor.image_processor is None:
            raise AssertionError("reloaded checkpoint is missing processor components")
        report = dict(state)
        report.update({"elapsed_sec": time.time() - t_start,
                       "checkpoint_reload_verified": True,
                       "checkpoint_retained": False,
                       "processor_reloaded": True,
                       "domain_embedding_rows": int(soft_prompt.num_embeddings),
                       "metrics": str(metrics_path.resolve())})
        (args.output_dir / "state.json").write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        # The 3.3 GB smoke checkpoint is only needed for this reload check.
        # Preserve the audit report/state and remove the duplicate weights.
        del reloaded
        shutil.rmtree(checkpoint)
        (args.output_dir / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2), flush=True)
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
