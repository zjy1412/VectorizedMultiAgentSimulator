from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

import torch

# Ensure repo root is importable.
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.networks import DeepSetsCritic, TanhGaussianPolicy, ValueNetwork  # noqa: E402
from marl.utils import EpisodeStats, concat_agent_obs, set_global_seed  # noqa: E402
from vmas import make_env  # noqa: E402


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved checkpoint deterministically.")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--num_envs", type=int, default=32)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


@torch.no_grad()
def _team_reward(rews: List[torch.Tensor]) -> torch.Tensor:
    return torch.stack(rews, dim=0).mean(dim=0)


def _build_models(payload: dict, device: torch.device):
    algo = payload["algo"]
    obs_dim = int(payload["obs_dim"])
    act_dim = int(payload["act_dim"])
    n_agents = int(payload["n_agents"])

    if algo == "ippo":
        policy = TanhGaussianPolicy(obs_dim, act_dim).to(device)
        value = ValueNetwork(obs_dim).to(device)
    elif algo == "mappo":
        policy = TanhGaussianPolicy(obs_dim, act_dim).to(device)
        critic_kind = payload.get("mappo_critic", "concat")
        if critic_kind == "deepsets":
            value = DeepSetsCritic(obs_dim).to(device)
        else:
            value = ValueNetwork(obs_dim * n_agents).to(device)
    elif algo == "cppo":
        policy = TanhGaussianPolicy(obs_dim * n_agents, act_dim * n_agents).to(device)
        value = ValueNetwork(obs_dim * n_agents).to(device)
    else:
        raise ValueError(f"Unknown algo: {algo}")

    policy.load_state_dict(payload["policy_state_dict"])
    value.load_state_dict(payload["value_state_dict"])
    policy.eval()
    value.eval()
    return algo, policy, value, n_agents, obs_dim, act_dim


@torch.no_grad()
def _deterministic_actions(
    algo: str,
    policy: TanhGaussianPolicy,
    obs: List[torch.Tensor],
    n_agents: int,
    act_dim: int,
) -> List[torch.Tensor]:
    if algo in ("ippo", "mappo"):
        actions = []
        for i in range(n_agents):
            a, _, _ = policy.act(obs[i], deterministic=True)
            actions.append(a)
        return actions

    # CPPO: centralized joint action, then split.
    gobs = concat_agent_obs(obs)
    a_joint, _, _ = policy.act(gobs, deterministic=True)
    actions = []
    cursor = 0
    for _ in range(n_agents):
        actions.append(a_joint[:, cursor : cursor + act_dim])
        cursor += act_dim
    return actions


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    set_global_seed(args.seed)

    payload = torch.load(args.ckpt, map_location="cpu")
    algo, policy, _, n_agents, obs_dim, act_dim = _build_models(payload, device=device)

    env = make_env(
        scenario=payload.get("scenario", "balance"),
        num_envs=args.num_envs,
        device=str(device),
        continuous_actions=True,
        wrapper=None,
        max_steps=args.max_steps,
        dict_spaces=False,
        terminated_truncated=False,
        n_agents=n_agents,
    )

    obs = env.reset(seed=args.seed)
    stats = EpisodeStats(num_envs=env.num_envs, device=env.device)
    completed = 0

    while completed < args.episodes:
        actions = _deterministic_actions(algo, policy, obs, n_agents=n_agents, act_dim=act_dim)
        obs, rews, dones, _ = env.step(actions)
        stats.step(_team_reward(rews), dones)
        completed = len(stats.completed_returns)
        if dones.any():
            # Let VMAS reset those environments.
            done_idx = dones.nonzero(as_tuple=False).squeeze(-1).tolist()
            for i in done_idx:
                env.scenario.env_reset_world_at(i)
                env.steps[i] = 0
            obs = env.get_from_scenario(True, False, False, False)[0]

    print(f"Checkpoint: {args.ckpt}")
    print(f"Algo: {algo}, episodes: {args.episodes}, mean_return: {stats.mean_return():.3f}, mean_len: {stats.mean_length():.1f}")


if __name__ == "__main__":
    main()


