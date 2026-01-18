from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _init_layer(layer: nn.Module, std: float = math.sqrt(2), bias_const: float = 0.0):
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer


class MLP(nn.Module):
    """Simple MLP used for both actor and critic."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dims=(256, 256), activation=nn.Tanh):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers.append(_init_layer(nn.Linear(last, h)))
            layers.append(activation())
            last = h
        layers.append(_init_layer(nn.Linear(last, out_dim), std=0.01))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


@dataclass(frozen=True)
class TanhNormal:
    """Tanh-squashed Normal distribution helper.

    We sample `z ~ Normal(mean, std)` then apply `a = tanh(z)`.
    Log-prob includes the tanh change-of-variables correction.
    """

    mean: Tensor
    log_std: Tensor

    def _normal(self) -> torch.distributions.Normal:
        std = self.log_std.exp()
        return torch.distributions.Normal(self.mean, std)

    def sample_with_log_prob(self) -> Tuple[Tensor, Tensor]:
        normal = self._normal()
        z = normal.rsample()
        a = torch.tanh(z)
        # Change of variables correction for tanh.
        # log|det(d tanh(z)/dz)| = sum log(1 - tanh(z)^2)
        logp = normal.log_prob(z) - torch.log(1.0 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=-1)
        return a, logp

    def log_prob_from_action(self, action: Tensor) -> Tensor:
        # Inverse tanh with numerical stability.
        a = action.clamp(-1 + 1e-6, 1 - 1e-6)
        z = 0.5 * (torch.log1p(a) - torch.log1p(-a))
        normal = self._normal()
        logp = normal.log_prob(z) - torch.log(1.0 - a.pow(2) + 1e-6)
        return logp.sum(dim=-1)

    def entropy(self) -> Tensor:
        # Entropy of tanh-squashed Normal doesn't have a closed form; use Normal entropy as proxy.
        return self._normal().entropy().sum(dim=-1)


class TanhGaussianPolicy(nn.Module):
    """Gaussian policy with tanh squashing to [-1, 1] (then optionally scaled by caller)."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.actor = MLP(obs_dim, act_dim, hidden_dims=hidden_dims, activation=nn.Tanh)
        # State-independent log_std (common PPO practice).
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, obs: Tensor) -> TanhNormal:
        mean = self.actor(obs)
        log_std = self.log_std.expand_as(mean).clamp(-20, 2)
        return TanhNormal(mean=mean, log_std=log_std)

    def act(self, obs: Tensor, deterministic: bool = False) -> Tuple[Tensor, Tensor, Tensor]:
        dist = self.dist(obs)
        if deterministic:
            a = torch.tanh(dist.mean)
            logp = dist.log_prob_from_action(a)
        else:
            a, logp = dist.sample_with_log_prob()
        ent = dist.entropy()
        return a, logp, ent

    def log_prob(self, obs: Tensor, action: Tensor) -> Tensor:
        return self.dist(obs).log_prob_from_action(action)


class ValueNetwork(nn.Module):
    """Scalar value function V(s) (decentralized or centralized depending on input)."""

    def __init__(self, in_dim: int, hidden_dims=(256, 256)):
        super().__init__()
        self.critic = MLP(in_dim, 1, hidden_dims=hidden_dims, activation=nn.Tanh)

    def forward(self, x: Tensor) -> Tensor:
        return self.critic(x).squeeze(-1)


class DeepSetsCritic(nn.Module):
    """Permutation-invariant centralized critic (DeepSets-style).

    Instead of concatenating all agents' observations (which scales linearly in number of agents
    and is sensitive to agent ordering), we encode each agent observation with a shared encoder,
    pool across agents, and predict a single team value.

    Input:
        obs: Tensor of shape [batch, n_agents, obs_dim]
    Output:
        value: Tensor of shape [batch]
    """

    def __init__(self, obs_dim: int, embed_dim: int = 128, hidden_dims=(256, 256)):
        super().__init__()
        self.encoder = MLP(obs_dim, embed_dim, hidden_dims=hidden_dims, activation=nn.Tanh)
        self.value_head = MLP(embed_dim, 1, hidden_dims=hidden_dims, activation=nn.Tanh)

    def forward(self, obs: Tensor) -> Tensor:
        # obs: [B, A, D]
        B, A, D = obs.shape
        emb = self.encoder(obs.reshape(B * A, D)).reshape(B, A, -1)  # [B, A, E]
        pooled = emb.mean(dim=1)  # [B, E]
        return self.value_head(pooled).squeeze(-1)


