# ------------------------------------------------------------------------------
# X-VLA dataset readers
# ------------------------------------------------------------------------------
from __future__ import annotations

import io
import json
import random
from pathlib import Path
from typing import Dict, Iterable

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .utils import action_slice
from .domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID


class ManifestWindowDataset(IterableDataset):
    """Manifest/index driven RoboTwin2 reader.

    The manifest is the source of truth for window counts and sampling
    probabilities.  Each worker/rank owns an independent RNG stream; the
    finite ``iter_finite`` path is used by preflight while training uses the
    infinite sampler in ``__iter__``.
    """

    def __init__(self, manifest_path: str | Path, *, training: bool = True,
                 sampler_mode: str = "domain_balanced", base_seed: int = 0,
                 image_aug=None, finite: bool = False, max_samples: int | None = None):
        super().__init__()
        self.manifest_path = Path(manifest_path)
        if not self.manifest_path.exists():
            raise FileNotFoundError(self.manifest_path)
        self.total = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        self.training = bool(training)
        self.sampler_mode = sampler_mode
        self.base_seed = int(base_seed)
        self.finite = bool(finite)
        self.max_samples = max_samples
        if sampler_mode not in ("domain_balanced", "tempered_T2"):
            raise ValueError(f"unknown RoboTwin sampler mode: {sampler_mode}")
        if self.total.get("qdur_sec") != 1.0 or self.total.get("num_actions") != 30:
            raise ValueError("RoboTwin manifest must freeze qdur=1.0 and num_actions=30")
        self.image_aug = image_aug or transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
            if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ])
        from .domain_handler.robotwin2_ft import RobotWin2FTHandler
        self._handlers = {}
        self._entries = {}
        self._pair_keys = []
        for pair in self.total.get("pairs", []):
            if pair.get("split") != "train":
                continue
            domain_id = int(pair.get("domain_id", -1))
            if domain_id not in (0, 1, 2):
                raise ValueError(f"manifest has invalid domain_id={domain_id}")
            task = str(pair.get("task", ""))
            key = f"{domain_id}::{task}"
            if key in self._entries:
                raise ValueError(f"duplicate manifest pair {key}")
            index_path = self.manifest_path.parent / f"{self._domain_name(domain_id)}__{task}.windows.jsonl"
            # Prefer an explicitly recorded path if present in the pair record.
            index_path_recorded = pair.get("windows_path")
            if index_path_recorded:
                index_path = Path(index_path_recorded)
            if not index_path.exists():
                raise FileNotFoundError(index_path)
            entries = [json.loads(line) for line in index_path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if len(entries) != int(pair["num_action_observation_windows"]):
                raise ValueError(f"{key}: index count does not match manifest")
            for entry in entries:
                if int(entry.get("domain_id", -1)) != domain_id or entry.get("task") != task:
                    raise ValueError(f"{key}: malformed window entry")
                if entry.get("num_actions") != 30 or entry.get("qdur_sec") != 1.0:
                    raise ValueError(f"{key}: window contract mismatch")
            meta = dict(pair)
            meta["windows_path"] = str(index_path)
            handler = RobotWin2FTHandler(meta, num_views=3, image_aug=self.image_aug, windows=entries)
            self._handlers[key] = handler
            self._entries[key] = entries
            self._pair_keys.append(key)
        expected = {f"{d}::{t}" for d in range(3) for t in self.total.get("tasks", [])}
        if set(self._pair_keys) != expected:
            raise ValueError(f"manifest must contain all 15 train pairs; missing={sorted(expected-set(self._pair_keys))}")
        probabilities = self.total["sampling"].get(sampler_mode, {}).get("probabilities")
        if not probabilities:
            raise ValueError(f"manifest has no {sampler_mode} probabilities")
        self.probabilities = {k: float(probabilities[k]) for k in self._pair_keys}
        total_prob = sum(self.probabilities.values())
        if not np.isfinite(total_prob) or abs(total_prob - 1.0) > 1e-8:
            raise ValueError(f"sampler probabilities must sum to one, got {total_prob}")
        self._cdf = np.cumsum([self.probabilities[k] for k in self._pair_keys])

    @staticmethod
    def _domain_name(domain_id: int) -> str:
        return {0: "aloha-agilex", 1: "arx-x5", 2: "piper"}[domain_id]

    def __len__(self):
        return sum(len(v) for v in self._entries.values())

    def sampler_stats(self) -> dict:
        n_dt = {k: len(self._entries[k]) for k in self._pair_keys}
        n_d = {str(d): sum(v for k, v in n_dt.items() if k.startswith(f"{d}::")) for d in range(3)}
        return {"N_dt": n_dt, "N_d": n_d, "probabilities": self.probabilities, "sampler_mode": self.sampler_mode}

    def _rng(self):
        info = get_worker_info()
        worker_id = info.id if info is not None else 0
        rank = 0
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            rank = torch.distributed.get_rank()
        seed = self.base_seed + rank * 100003 + worker_id
        return np.random.default_rng(seed)

    def sample_entries(self, per_domain: int = 2, per_task: bool = False) -> list[dict]:
        """Return deterministic entries for smoke/audit without decoding them."""
        out = []
        if per_task:
            for key in self._pair_keys:
                out.extend(self._entries[key][:per_domain])
        else:
            for domain in range(3):
                keys = [k for k in self._pair_keys if k.startswith(f"{domain}::")]
                for key in keys[:1]:
                    out.extend(self._entries[key][:per_domain])
        return out

    def _yield_entry(self, entry: dict):
        key = f"{int(entry['domain_id'])}::{entry['task']}"
        return self._handlers[key].sample_from_entry(entry)

    def iter_finite(self):
        entries = []
        for key in self._pair_keys:
            entries.extend(self._entries[key])
        info = get_worker_info()
        if info is not None:
            entries = entries[info.id::info.num_workers]
        if self.max_samples is not None:
            entries = entries[: self.max_samples]
        for entry in entries:
            yield self._yield_entry(entry)

    def __iter__(self):
        if self.finite or not self.training:
            yield from self.iter_finite()
            return
        rng = self._rng()
        while True:
            draw = float(rng.random())
            idx = int(np.searchsorted(self._cdf, draw, side="right"))
            idx = min(idx, len(self._pair_keys) - 1)
            key = self._pair_keys[idx]
            entry = self._entries[key][int(rng.integers(len(self._entries[key])))]
            yield self._yield_entry(entry)


class InfiniteDataReader(IterableDataset):
    """Legacy reader retained for non-RoboTwin datasets."""
    def __init__(self, metas_path: str, num_actions: int = 10, num_views: int = 3,
                 training: bool = True, action_mode: str = "ee6d", lang_aug: str = None):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.metas: Dict[str, dict] = {}
        try:
            from mmengine import fileio
            read = lambda p: fileio.get(p)
            isdir = fileio.isdir
            list_files = lambda p: fileio.list_dir_or_file(p, suffix=".json", recursive=True, list_dir=False)
            join = fileio.join_path
        except ImportError:
            import os
            read = lambda p: Path(p).read_bytes()
            isdir = os.path.isdir
            list_files = lambda p: [x.name for x in Path(p).glob("*.json")]
            join = os.path.join
        if isdir(metas_path):
            meta_files, root = list_files(metas_path), metas_path
        else:
            meta_files, root = [metas_path], ""
        for file in meta_files:
            file_path = join(root, file)
            meta = json.loads(read(file_path))
            if "dataset_name" in meta and "datalist" in meta:
                self.metas[meta["dataset_name"]] = meta
            elif meta.get("codebase_version") == "v2.1":
                raise NotImplementedError("legacy LeRobot metadata requires the full mmengine environment")
            else:
                raise NotImplementedError(f"unrecognized meta file format: {file}")
        self.image_aug = transforms.Compose([
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.) if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ])

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        from .domain_handler.registry import get_handler_cls
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        if self.training:
            random.shuffle(traj_indices)
        robot_type = meta.get("robot_type", dataset_name)
        Handler = get_handler_cls(robot_type)
        handler = Handler(meta=meta, num_views=self.num_views)
        for traj_idx in traj_indices:
            for sample in handler.iter_episode(traj_idx, num_actions=self.num_actions, training=self.training,
                                               image_aug=self.image_aug,
                                               lang_aug_map=meta.get("lang_aug_map"), action_mode=self.action_mode):
                domain_id = DATA_DOMAIN_ID.get(robot_type)
                if domain_id is None:
                    raise KeyError(f"No domain id for legacy dataset {robot_type}")
                sample["domain_id"] = torch.tensor(domain_id, dtype=torch.long)
                idx_for_delta = sample.pop("idx_for_delta", [])
                idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
                sample.update(action_slice(sample.pop("abs_trajectory", None), idx_for_delta, idx_for_mask_proprio))
                yield sample
        if self.training:
            yield from self._iter_one_dataset(dataset_name)

    def __iter__(self):
        names = list(self.metas.keys())
        if not self.training:
            for name in names:
                yield from self._iter_one_dataset(name)
            return
        gens = [iter(self._iter_one_dataset(name)) for name in names]
        weights = [DATA_WEIGHTS.get(name, 1.0) for name in names]
        while True:
            idx = random.choices(range(len(names)), weights=weights, k=1)[0]
            yield next(gens[idx])
