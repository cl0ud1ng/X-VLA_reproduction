import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .dataset import InfiniteDataReader, ManifestWindowDataset


def worker_init_fn(worker_id: int):
    """Seed each worker without collapsing distributed ranks to one stream."""
    info = torch.utils.data.get_worker_info()
    base_seed = int(info.seed if info is not None else torch.initial_seed()) % (2**32)
    import random
    import numpy as np
    np.random.seed(base_seed)
    random.seed(base_seed)
    torch.manual_seed(base_seed)


def create_dataloader(batch_size: int, metas_path: str, num_actions: int,
                      training: bool, action_mode: str, *, sampler_mode: str = "domain_balanced",
                      num_workers: int = 0, base_seed: int = 0, finite: bool = False,
                      max_samples: int | None = None):
    """Create either the RoboTwin manifest reader or a legacy reader.

    ``num_workers=0`` is the deterministic default for smoke/audit.  Training
    can raise it after verifying that each worker receives a unique seed.
    """
    path = Path(metas_path)
    is_robotwin_manifest = False
    if path.is_file():
        try:
            with path.open(encoding="utf-8") as handle:
                is_robotwin_manifest = "pairs" in json.load(handle)
        except (OSError, json.JSONDecodeError):
            pass
    if is_robotwin_manifest:
        dataset = ManifestWindowDataset(path, training=training, sampler_mode=sampler_mode,
                                        base_seed=base_seed, finite=finite, max_samples=max_samples)
    else:
        dataset = InfiniteDataReader(metas_path, num_actions=num_actions, training=training, action_mode=action_mode)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available(), worker_init_fn=worker_init_fn,
                      persistent_workers=bool(num_workers), drop_last=training)
