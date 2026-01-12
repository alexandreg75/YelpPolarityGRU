"""
Utils génériques.

Fonctions attendues :
- set_seed(seed: int) -> None
- get_device(prefer: str | None = "auto") -> str
- count_parameters(model) -> int
- save_config_snapshot(config: dict, out_dir: str) -> None
"""

import os
import random
from typing import Optional

import numpy as np
import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # safe even if no cuda
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(prefer: Optional[str] = "auto") -> str:
    if prefer is None:
        prefer = "auto"
    prefer = str(prefer).lower()
    if prefer == "cpu":
        return "cpu"
    if prefer == "cuda":
        return "cuda" if torch.cuda.is_available() else "cpu"
    # auto
    return "cuda" if torch.cuda.is_available() else "cpu"


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config_snapshot(config: dict, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "config_snapshot.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False, allow_unicode=True)
