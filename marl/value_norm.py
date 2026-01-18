from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class ValueNorm:
    """Running mean/std normalizer for value targets.

    This is a common stabilization trick in PPO/MAPPO implementations:
    - Keep the critic predicting *raw* values (no architectural change).
    - Normalize both value predictions and return targets using a running mean/std
      computed from returns, and compute value loss in normalized space.

    This improves numerical conditioning when returns have large scale shifts.
    """

    eps: float = 1e-5
    device: torch.device | None = None

    def __post_init__(self):
        dev = self.device if self.device is not None else torch.device("cpu")
        self.mean = torch.zeros((), device=dev)
        self.var = torch.ones((), device=dev)
        self.count = torch.tensor(0.0, device=dev)

    @torch.no_grad()
    def update(self, x: Tensor) -> None:
        # x: any shape; we treat it as a batch of scalars
        x = x.detach().to(self.mean.device).reshape(-1)
        if x.numel() == 0:
            return
        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = torch.tensor(float(x.numel()), device=self.mean.device)

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var.clamp(min=self.eps)
        self.count = tot_count

    def normalize(self, x: Tensor) -> Tensor:
        std = torch.sqrt(self.var + self.eps)
        return (x - self.mean) / std

