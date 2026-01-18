from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import Tensor


@dataclass
class PPOHyperParams:
    gamma: float = 0.99
    gae_lambda: float = 0.9
    clip_param: float = 0.2
    entropy_coeff: float = 0.0
    vf_loss_coeff: float = 1.0
    vf_clip_param: float = float("inf")
    max_grad_norm: float = 0.5
    num_sgd_iter: int = 40
    sgd_minibatch_size: int = 4096
    lr: float = 5e-5


def compute_gae(
    rewards: Tensor,
    dones: Tensor,
    values: Tensor,
    last_value: Tensor,
    gamma: float,
    gae_lambda: float,
) -> Tuple[Tensor, Tensor]:
    """Generalized Advantage Estimation.

    Args:
        rewards: [T, ...]
        dones: [T, ...] bool-like (1 when episode ended at that step)
        values: [T, ...]
        last_value: [...]
    Returns:
        advantages: [T, ...]
        returns: [T, ...]
    """
    T = rewards.shape[0]
    advantages = torch.zeros_like(rewards)
    last_adv = torch.zeros_like(last_value)

    # Convert dones to float mask: 0 when terminal (no bootstrap), 1 otherwise.
    not_done = 1.0 - dones.to(dtype=rewards.dtype)

    for t in reversed(range(T)):
        next_value = last_value if t == T - 1 else values[t + 1]
        delta = rewards[t] + gamma * next_value * not_done[t] - values[t]
        last_adv = delta + gamma * gae_lambda * not_done[t] * last_adv
        advantages[t] = last_adv

    returns = advantages + values
    return advantages, returns


def normalize_advantages(advantages: Tensor, eps: float = 1e-8) -> Tensor:
    return (advantages - advantages.mean()) / (advantages.std(unbiased=False) + eps)


