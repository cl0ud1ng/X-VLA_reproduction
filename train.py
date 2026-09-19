# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

import os
import math
import time
import json
import random
import argparse
import shutil
import subprocess
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.optim import AdamW

from accelerate import Accelerator, DataLoaderConfiguration
from datasets import create_dataloader
from models.modeling_xvla import XVLA
from models.processing_xvla import XVLAProcessor

import logging
import os
import sys
import psutil

# ============================================================
# logger
# ============================================================
def get_logger(name="train", output_dir=None, accelerator=None, level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False 
    if logger.handlers:
        return logger
    is_main = accelerator is None or accelerator.is_main_process
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    datefmt = "%H:%M:%S"
    formatter = logging.Formatter(fmt=fmt, datefmt=datefmt)
    if is_main:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(level)
        logger.addHandler(ch)
    if output_dir and is_main:
        os.makedirs(output_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(output_dir, "train.log"), mode="a")
        fh.setFormatter(formatter)
        fh.setLevel(level)
        logger.addHandler(fh)
    return logger


# ============================================================
# Argument Parser
# ============================================================
def get_args_parser():
    parser = argparse.ArgumentParser("XVLA Training", add_help=False)

    # I/O
    parser.add_argument("--models", type=str, required=True, help="Path or HF repo for pretrained XVLA")
    parser.add_argument("--output_dir", type=str, default="runnings", help="Directory to save checkpoints")

    # Data
    parser.add_argument("--train_metas_path", type=str, required=True, help="Path to training metadata")
    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size")
    parser.add_argument("--global_batch_size", type=int, default=0, help="Expected global batch; 0 derives it")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--sampler_mode", choices=("domain_balanced", "tempered_T2"), default="domain_balanced")
    parser.add_argument("--num_workers", type=int, default=0)

    # Optimizer
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--learning_coef", type=float, default=1.0, help="LR multiplier for soft prompts")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.95))
    parser.add_argument("--max_grad_norm", type=float, default=1.0)

    # Schedule
    parser.add_argument("--iters", type=int, default=1000000)
    parser.add_argument("--freeze_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=2000)
    parser.add_argument("--use_cosine_decay", action="store_true", default=False)
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)

    # Logging / saving
    parser.add_argument("--save_interval", type=int, default=50000)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--disable_checkpoint", action="store_true")

    # System
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base_seed", type=int, default=0, help="Manifest sampler base seed")
    parser.add_argument("--mixed_precision", choices=("no", "fp16", "bf16"), default="no")
    parser.add_argument("--report_to", choices=("none", "tensorboard", "wandb", "all"), default="tensorboard")
    parser.add_argument("--wandb_project", type=str, default="xvla-robotwin2-ft")
    parser.add_argument("--wandb_entity", type=str, default="")
    parser.add_argument("--wandb_run_name", type=str, default="")
    parser.add_argument("--wandb_mode", choices=("online", "offline", "disabled"), default="online")

    return parser


# ============================================================
# Utilities
# ============================================================
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True


def build_optimizer(model: XVLA, lr: float, weight_decay: float, betas=(0.9, 0.95), lr_coef_soft=1.0):
    """Split param groups by module type with different learning rates."""
    vlm_params = list(model.vlm.parameters())
    soft_prompt_params = list(model.transformer.soft_prompt_hub.parameters())
    action_params = list(model.transformer.action_decoder.parameters()) + list(model.transformer.action_encoder.parameters())
    exclude = set(map(id, vlm_params + soft_prompt_params + action_params))
    transformer_core_params = [p for p in model.parameters() if id(p) not in exclude]
    param_groups = [
        {"name": "vlm", "params": vlm_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "transformer_core", "params": transformer_core_params, "lr": 0.0, "weight_decay": weight_decay},
        {"name": "soft_prompts", "params": soft_prompt_params, "lr": lr * lr_coef_soft, "weight_decay": weight_decay},
        {"name": "action_heads", "params": action_params, "lr": lr, "weight_decay": weight_decay},
    ]
    return AdamW(param_groups, betas=betas, foreach=False)


def set_group_lr(optim: torch.optim.Optimizer, name: str, lr: float):
    for g in optim.param_groups: 
        if g["name"] == name: g["lr"] = lr


def get_group_lr(optim: torch.optim.Optimizer, name: str) -> float:
    for g in optim.param_groups:
        if g["name"] == name: return g["lr"]
    return 0.0


def linear_warmup_cosine(step, start, warmup, total, base_lr, min_ratio):
    """Linear warmup followed by cosine decay."""
    if step < start: return 0.0
    progress = step - start
    if progress < warmup:
        return base_lr * (progress / max(1, warmup))
    remain = max(1, total - (start + warmup))
    ratio = 0.5 * (1 + math.cos(math.pi * min(1.0, (progress - warmup) / remain)))
    return base_lr * (min_ratio + (1 - min_ratio) * ratio)


def update_group_lrs(optim, step, args):
    """Elegant group-wise LR scheduler."""
    base = {
        "vlm": args.learning_rate * args.learning_coef,
        "transformer_core": args.learning_rate,
        "soft_prompts": args.learning_rate * args.learning_coef,
        "action_heads": args.learning_rate,
    }
    def schedule(step, base_lr):
        return linear_warmup_cosine(step, args.freeze_steps, args.warmup_steps, args.iters, base_lr, args.min_lr_ratio)
    if step < args.freeze_steps:
        set_group_lr(optim, "vlm", 0.0)
        set_group_lr(optim, "transformer_core", 0.0)
        set_group_lr(optim, "soft_prompts", base["soft_prompts"])
        set_group_lr(optim, "action_heads", base["action_heads"])
    else:
        for name, base_lr in base.items():
            new_lr = schedule(step, base_lr) if args.use_cosine_decay else base_lr
            set_group_lr(optim, name, new_lr)


# ============================================================
# Main Training
# ============================================================
def main(args):
    output_dir = Path(args.output_dir)
    if args.report_to == "none":
        log_with = None
    elif args.report_to == "all":
        log_with = ["tensorboard", "wandb"]
    else:
        log_with = args.report_to
    accelerator = Accelerator(
        mixed_precision=None if args.mixed_precision == "no" else args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_config=DataLoaderConfiguration(split_batches=False, even_batches=True),
        log_with=log_with,
        project_dir=output_dir
    )
    tracker_kwargs = {}
    if args.report_to in ("wandb", "all"):
        wandb_kwargs = {
            "project": args.wandb_project,
            "mode": args.wandb_mode,
        }
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        if args.wandb_run_name:
            wandb_kwargs["name"] = args.wandb_run_name
        tracker_kwargs["wandb"] = wandb_kwargs
    run_config = dict(vars(args))
    run_config["git_commit"] = "unknown"
    try:
        run_config["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        pass
    accelerator.init_trackers("XVLA-Training", config=run_config, init_kwargs=tracker_kwargs)
    
    accelerator.wait_for_everyone()
    logger = get_logger(__name__, output_dir=output_dir, accelerator=accelerator)
    
    set_seed(args.seed + accelerator.process_index)
    logger.info(f"Args: {args}")
    try:
        git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_commit = "unknown"

    # Load model & processor and fail early on the frozen RoboTwin contract.
    model = XVLA.from_pretrained(args.models)
    processor = XVLAProcessor.from_pretrained(args.models)
    if model.action_mode != "ee6d" or model.num_actions != 30 or not model.use_proprio:
        raise ValueError("RoboTwin training requires action_mode=ee6d, num_actions=30, use_proprio=True")
    if getattr(model.config, "num_domains", 0) < 3 or model.action_space.dim_action != 20:
        raise ValueError("RoboTwin training requires at least three domains and 20-D EE6D actions")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("batch_size and gradient_accumulation_steps must be positive")
    global_batch_size = args.batch_size * accelerator.num_processes * args.gradient_accumulation_steps
    if args.global_batch_size and args.global_batch_size != global_batch_size:
        raise ValueError(f"global_batch_size={args.global_batch_size} != per_device batch {args.batch_size} * "
                         f"world_size {accelerator.num_processes} * accumulation {args.gradient_accumulation_steps}")

    train_dataloader = create_dataloader(
        batch_size=args.batch_size,
        metas_path=args.train_metas_path,
        num_actions=model.num_actions,
        action_mode=model.action_mode,
        training=True,
        sampler_mode=args.sampler_mode,
        num_workers=args.num_workers,
        base_seed=args.base_seed,
    )

    # Optimizer. Frozen-stage groups stay at LR=0 and are opened by the
    # schedule; DDP must keep them registered for the later unfreeze boundary.
    optim = build_optimizer(
        model=model,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=tuple(args.betas),
        lr_coef_soft=args.learning_coef,
    )
    model, optim, train_dataloader = accelerator.prepare(model, optim, train_dataloader)

    # Training loop
    model.train()
    global_step, t0 = 0, time.time()
    logger.info(f"🚀 Start training for {args.iters} optimizer steps | world_size={accelerator.num_processes} | "
                f"per_device_batch={args.batch_size} global_batch={global_batch_size} "
                f"gradient_accumulation={args.gradient_accumulation_steps}")
    optim.zero_grad(set_to_none=True)
    last_step_time = time.time()
    
    for batch in train_dataloader:
        with accelerator.accumulate(model):
            # Encode language
            language_values = batch["language_instruction"]
            if isinstance(language_values, torch.Tensor):
                language_values = [bytes(row.tolist()).split(b"\0", 1)[0].decode("utf-8") for row in language_values]
            lang = processor.encode_language(language_values)
            # Metadata is retained for audit logging but is not a model input.
            task_values = batch.get("task_id", [])
            terminal_values = batch.get("terminal_hold", [])
            batch.pop("language_instruction", None)
            for metadata_key in ("task_id", "terminal_hold", "window_key"):
                batch.pop(metadata_key, None)
            inputs = {**batch, **lang}
            moved = {}
            for key, value in inputs.items():
                if not isinstance(value, torch.Tensor):
                    continue
                moved[key] = value.to(accelerator.device, non_blocking=True)
            inputs = moved
            # Update LR per group; VLM/core stay at zero through freeze_steps.
            update_group_lrs(optim, global_step, args)

            # Forward & backward. Accelerator scales the loss across the
            # configured accumulation window and suppresses intermediate DDP
            # all-reduces.
            with accelerator.autocast():
                loss_dict: Dict[str, torch.Tensor] = model(**inputs)
                loss = sum(loss_dict.values())
            accelerator.backward(loss)

        if accelerator.sync_gradients:
            grad_norm = torch.tensor(0.0, device=accelerator.device)
            if args.max_grad_norm:
                grad_norm = accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            optim.zero_grad(set_to_none=True)
            global_step += 1

        # Logging only at optimizer-step boundaries.
        if accelerator.sync_gradients and global_step % args.log_interval == 0:
            logs = {k: v.detach().float().item() for k, v in loss_dict.items()}
            # Stable metric names are part of the Stage-C acceptance contract.
            logs["loss_position"] = logs.get("position_loss", logs.get("loss_position", 0.0))
            logs["loss_rotate6D"] = logs.get("rotate6D_loss", logs.get("loss_rotate6D", 0.0))
            logs["loss_gripper"] = logs.get("gripper_loss", logs.get("loss_gripper", 0.0))
            gathered_domains = accelerator.gather(batch["domain_id"].detach())
            domain_counts = torch.bincount(gathered_domains.cpu(), minlength=3)
            for domain_id in range(3):
                logs[f"domain_count[{domain_id}]"] = int(domain_counts[domain_id])
            if isinstance(task_values, torch.Tensor):
                gathered_tasks = accelerator.gather(task_values.detach())
                task_counts = torch.bincount(gathered_tasks.cpu(), minlength=5)
                for task_id in range(len(task_counts)):
                    logs[f"task_count[{task_id}]"] = int(task_counts[task_id])
            if isinstance(terminal_values, torch.Tensor):
                gathered_terminal = accelerator.gather(terminal_values.detach().to(torch.int64))
                logs["terminal_hold_count"] = int(gathered_terminal.sum().item())
            logs["loss_total"] = float(loss.detach().item())
            logs["grad_norm"] = float(grad_norm.detach().float().item())
            logs["batch/per_device"] = args.batch_size
            logs["batch/global"] = global_batch_size
            logs["batch/gradient_accumulation"] = args.gradient_accumulation_steps
            logs["step_time_sec"] = time.time() - last_step_time
            last_step_time = time.time()
            logs.update({f"lr_{g['name']}": g["lr"] for g in optim.param_groups})
            accelerator.log(logs, step=global_step)

            if accelerator.is_main_process:
                dt = (time.time() - t0) / args.log_interval
                t0 = time.time()
                cpu_mem = psutil.Process(os.getpid()).memory_info().rss / 1024**3
                gpu_mem = torch.cuda.memory_allocated(accelerator.device) / 1024**3 if torch.cuda.is_available() else 0.0
                logger.info(
                    f"[{global_step}/{args.iters}] "
                    f"loss={logs['loss_total']:.4f} "
                    f"lr_core={logs['lr_transformer_core']:.2e} "
                    f"lr_vlm={logs['lr_vlm']:.2e} ({dt:.2f}s/it) "
                    f"USED_CPU={cpu_mem:.2e} GB "
                    f"USED_GPU={gpu_mem:.2e} GB "
                )
        
        # Checkpointing
        if accelerator.sync_gradients and accelerator.is_main_process:
            if not args.disable_checkpoint and (global_step == args.iters or global_step % args.save_interval == 0):
                save_dir = os.path.join(output_dir, f"ckpt-{global_step}")
                accelerator.print(f"💾 Saving model to {save_dir}")
                accelerator.unwrap_model(model).save_pretrained(save_dir, safe_serialization=True)
                processor.save_pretrained(save_dir)
                with open(os.path.join(save_dir, "state.json"), "w") as f:
                    json.dump({"global_step": global_step, "sampler_mode": args.sampler_mode,
                               "manifest": os.path.abspath(args.train_metas_path),
                               "seed": args.seed, "git_commit": git_commit,
                               "world_size": accelerator.num_processes,
                               "per_device_batch_size": args.batch_size,
                               "global_batch_size": global_batch_size,
                               "gradient_accumulation_steps": args.gradient_accumulation_steps,
                               "mixed_precision": args.mixed_precision,
                               "command": " ".join(sys.argv)}, f, indent=2)
                shutil.copy2(args.train_metas_path, os.path.join(save_dir, "total_manifest.json"))
        if global_step >= args.iters:
            break

    accelerator.end_training()

# ============================================================
# Entry
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser("XVLA training script", parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
