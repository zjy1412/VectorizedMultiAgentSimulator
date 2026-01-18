from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


@dataclass
class EpisodeStats:
    """Track episode returns/lengths for a vectorized env (per-env)."""

    num_envs: int
    device: torch.device

    def __post_init__(self):
        self.reset()

    def reset(self):
        self.ep_returns = torch.zeros(self.num_envs, device=self.device)
        self.ep_lengths = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.completed_returns: List[float] = []
        self.completed_lengths: List[int] = []

    def step(self, reward: Tensor, done: Tensor):
        # reward: [num_envs], done: [num_envs] bool
        self.ep_returns += reward
        self.ep_lengths += 1
        if done.any():
            idx = done.nonzero(as_tuple=False).squeeze(-1)
            self.completed_returns.extend(self.ep_returns[idx].detach().cpu().tolist())
            self.completed_lengths.extend(self.ep_lengths[idx].detach().cpu().tolist())
            self.ep_returns[idx] = 0.0
            self.ep_lengths[idx] = 0

    def mean_return(self) -> float:
        if not self.completed_returns:
            return float("nan")
        return float(np.mean(self.completed_returns))

    def mean_length(self) -> float:
        if not self.completed_lengths:
            return float("nan")
        return float(np.mean(self.completed_lengths))


def write_csv_row(path: str | Path, fieldnames: Sequence[str], row: Dict):
    path = Path(path)
    ensure_dir(path.parent)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerow(row)


def concat_agent_obs(obs_list: List[Tensor]) -> Tensor:
    """Concatenate per-agent observations into a global state: [num_envs, sum(obs_dim_i)]."""

    return torch.cat(obs_list, dim=-1)


