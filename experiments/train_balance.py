from __future__ import annotations

import sys
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch

# Ensure the project root (which contains `marl/`) is on sys.path when running as a script:
#   python experiments/train_balance.py ...
# This is not needed when running as a module:
#   python -m experiments.train_balance ...
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.algos import TrainConfig, make_trainer
from marl.ppo import PPOHyperParams
from marl.utils import EpisodeStats, concat_agent_obs, ensure_dir, set_global_seed, write_csv_row
from vmas import make_env


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CPPO/MAPPO/IPPO on VMAS Balance (pure PyTorch).")
    p.add_argument("--algo", choices=["ippo", "mappo", "cppo"], required=True)
    p.add_argument(
        "--mappo_critic",
        choices=["concat", "deepsets"],
        default="concat",
        help="Only used when --algo mappo.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--buffer_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument(
        "--value_norm",
        default=False,
        action=argparse.BooleanOptionalAction,
        help="Use running return statistics to normalize value loss targets/predictions.",
    )

    # Paper-like defaults (VMAS paper, Sec. 6): 400 iters, 60000 env interactions per iter.
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--train_batch_size", type=int, default=60000)
    # Choose num_envs so that train_batch_size is divisible and close to horizon=200.
    p.add_argument("--num_envs", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=200)

    # Scenario config
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--scenario", type=str, default="balance")
    p.add_argument("--continuous_actions", action="store_true", default=True)

    # Logging/output
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument(
        "--save_ckpt",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Save model checkpoints (policy/value weights + metadata) at the end of training.",
    )
    p.add_argument(
        "--save_best_ckpt",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Also save the best checkpoint (by episode_return_mean) during training.",
    )
    return p.parse_args()


def _build_env(args: argparse.Namespace):
    env = make_env(
        scenario=args.scenario,
        num_envs=args.num_envs,
        # Use the resolved torch device string to avoid mismatches when CUDA is unavailable.
        device=str(args.device),
        continuous_actions=args.continuous_actions,
        wrapper=None,
        max_steps=args.max_steps,
        dict_spaces=False,
        terminated_truncated=False,
        n_agents=args.n_agents,
    )
    return env


def _get_obs_act_dims(env) -> Tuple[int, int]:
    # Tuple spaces: per-agent Box spaces
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    obs_dim = int(obs_space.shape[0])
    act_dim = int(act_space.shape[0])
    return obs_dim, act_dim


@torch.no_grad()
def _team_reward(rews: List[torch.Tensor]) -> torch.Tensor:
    # rews: list[n_agents] each [num_envs]
    return torch.stack(rews, dim=0).mean(dim=0)


def _reset_done_envs_with_next_obs(env, next_obs: List[torch.Tensor], done: torch.Tensor) -> List[torch.Tensor]:
    if not done.any():
        return next_obs

    done_idx = done.nonzero(as_tuple=False).squeeze(-1).tolist()
    for i in done_idx:
        env.scenario.env_reset_world_at(i)
        env.steps[i] = 0
    # After resetting, query fresh observations for all envs (vectorized).
    result = env.get_from_scenario(
        get_observations=True,
        get_rewards=False,
        get_infos=False,
        get_dones=False,
    )
    return result[0] if isinstance(result, list) and len(result) == 1 else result


def _collect_rollout(env, trainer, rollout_steps: int, buffer_device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    # Seed the VMAS environment for reproducibility.
    obs = env.reset(seed=seed)
    # Storage lists (T-long), each entry contains tensors for all envs.
    obs_steps = []
    gobs_steps = []
    act_steps = []
    a_joint_steps = []
    logp_steps = []
    logp_joint_steps = []
    rew_steps = []
    team_rew_steps = []
    done_steps = []
    val_steps = []

    stats = EpisodeStats(num_envs=env.num_envs, device=env.device)
    # Track per-episode info metrics (from agent_0).
    info_sums: Dict[str, torch.Tensor] = {}

    for _ in range(rollout_steps):
        actions, logps, vals = trainer.act(obs)
        next_obs, rews, dones, infos = env.step(actions)

        team_rew = _team_reward(rews)
        stats.step(team_rew, dones)

        # Accumulate a couple of scalar info channels (optional).
        if isinstance(infos, list) and len(infos) > 0 and isinstance(infos[0], dict):
            for k, v in infos[0].items():
                if isinstance(v, torch.Tensor) and v.shape == team_rew.shape:
                    info_sums.setdefault(k, torch.zeros_like(v))
                    info_sums[k] += v

        # Store on buffer_device (default CPU) to keep GPU memory bounded.
        obs_t = torch.stack(obs, dim=1)  # [num_envs, n_agents, obs_dim]
        act_t = torch.stack(actions, dim=1)  # [num_envs, n_agents, act_dim]

        obs_steps.append(obs_t.detach().to(buffer_device))
        act_steps.append(act_t.detach().to(buffer_device))
        logp_steps.append(logps.detach().to(buffer_device))
        done_steps.append(dones.detach().to(buffer_device))

        if trainer.cfg.algo in ("mappo", "cppo"):
            gobs = concat_agent_obs(obs)  # [num_envs, total_obs_dim]
            gobs_steps.append(gobs.detach().to(buffer_device))
            team_rew_steps.append(team_rew.detach().to(buffer_device))
            val_steps.append(vals.detach().to(buffer_device))
        else:
            # IPPO stores per-agent rewards/values.
            rew_t = torch.stack(rews, dim=1)  # [num_envs, n_agents]
            rew_steps.append(rew_t.detach().to(buffer_device))
            val_steps.append(vals.detach().to(buffer_device))  # [num_envs, n_agents]

        if trainer.cfg.algo == "cppo":
            # Recompute joint action tensor from per-agent actions (same splitting convention as CPPO.act).
            a_joint = torch.cat(actions, dim=-1)
            a_joint_steps.append(a_joint.detach().to(buffer_device))
            logp_joint_steps.append(logps[:, 0].detach().to(buffer_device))

        obs = _reset_done_envs_with_next_obs(env, next_obs, dones)

    # Last observations for bootstrapping
    last_obs = torch.stack(obs, dim=1).detach().to(buffer_device)  # [num_envs, n_agents, obs_dim]
    last_gobs = concat_agent_obs(obs).detach().to(buffer_device)  # [num_envs, total_obs_dim]

    batch: Dict[str, torch.Tensor] = {
        "done": torch.stack(done_steps, dim=0),  # [T, num_envs]
        "stats_ep_return_mean": torch.tensor(stats.mean_return()),
        "stats_ep_len_mean": torch.tensor(stats.mean_length()),
    }
    for k, v in info_sums.items():
        batch[f"info_sum/{k}"] = v.detach().cpu()

    if trainer.cfg.algo == "ippo":
        batch.update(
            {
                "obs": torch.stack(obs_steps, dim=0),  # [T, num_envs, n_agents, obs_dim]
                "act": torch.stack(act_steps, dim=0),  # [T, num_envs, n_agents, act_dim]
                "logp": torch.stack(logp_steps, dim=0),  # [T, num_envs, n_agents]
                "rew": torch.stack(rew_steps, dim=0),  # [T, num_envs, n_agents]
                "val": torch.stack(val_steps, dim=0),  # [T, num_envs, n_agents]
                "last_obs": last_obs,
            }
        )
    elif trainer.cfg.algo == "mappo":
        batch.update(
            {
                "obs": torch.stack(obs_steps, dim=0),
                "gobs": torch.stack(gobs_steps, dim=0),  # [T, num_envs, total_obs_dim]
                "act": torch.stack(act_steps, dim=0),
                "logp": torch.stack(logp_steps, dim=0),
                "team_rew": torch.stack(team_rew_steps, dim=0),  # [T, num_envs]
                "val": torch.stack(val_steps, dim=0),  # [T, num_envs]
                "last_obs": last_obs,  # for deepsets critic bootstrapping
                "last_gobs": last_gobs,
            }
        )
    else:  # cppo
        batch.update(
            {
                "gobs": torch.stack(gobs_steps, dim=0),
                "a_joint": torch.stack(a_joint_steps, dim=0),
                "logp_joint": torch.stack(logp_joint_steps, dim=0),
                "team_rew": torch.stack(team_rew_steps, dim=0),
                "val": torch.stack(val_steps, dim=0),
                "last_gobs": last_gobs,
            }
        )

    return batch


def main() -> None:
    args = _parse_args()
    resolved_device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    resolved_buffer_device = torch.device(
        args.buffer_device if args.buffer_device == "cpu" or torch.cuda.is_available() else "cpu"
    )
    # Overwrite args strings with resolved device strings for env creation.
    args.device = str(resolved_device)
    args.buffer_device = str(resolved_buffer_device)

    set_global_seed(args.seed)
    env = _build_env(args)

    obs_dim, act_dim = _get_obs_act_dims(env)
    total_obs_dim = obs_dim * args.n_agents
    total_act_dim = act_dim * args.n_agents

    if args.train_batch_size % args.num_envs != 0:
        raise ValueError(
            f"train_batch_size must be divisible by num_envs. Got {args.train_batch_size} and {args.num_envs}."
        )
    rollout_steps = args.train_batch_size // args.num_envs

    cfg = TrainConfig(
        algo=args.algo,
        device=resolved_device,
        buffer_device=resolved_buffer_device,
        n_agents=args.n_agents,
        obs_dim=obs_dim,
        act_dim=act_dim,
        total_act_dim=total_act_dim,
        total_obs_dim=total_obs_dim,
        hyper=PPOHyperParams(),
        mappo_critic=args.mappo_critic,
        value_norm=args.value_norm,
    )
    trainer = make_trainer(cfg)

    out_root = Path(args.out_dir) / args.scenario / args.algo
    ensure_dir(out_root)
    csv_path = out_root / f"seed_{args.seed}.csv"
    ckpt_dir = out_root / "checkpoints"
    ensure_dir(ckpt_dir)

    fieldnames = [
        "iteration",
        "timesteps_total",
        "rollout_steps",
        "episode_return_mean",
        "episode_len_mean",
        "policy_loss",
        "value_loss",
        "entropy",
        "total_loss",
        "seconds",
    ]

    timesteps_total = 0
    best_return = float("-inf")
    for it in range(1, args.iters + 1):
        t0 = time.time()
        batch = _collect_rollout(
            env,
            trainer,
            rollout_steps,
            buffer_device=resolved_buffer_device,
            seed=args.seed,
        )
        timesteps_total += args.train_batch_size
        metrics = trainer.update(batch)
        dt = time.time() - t0

        row = {
            "iteration": it,
            "timesteps_total": timesteps_total,
            "rollout_steps": rollout_steps,
            "episode_return_mean": float(batch["stats_ep_return_mean"].item()),
            "episode_len_mean": float(batch["stats_ep_len_mean"].item()),
            "seconds": dt,
            **metrics,
        }
        write_csv_row(csv_path, fieldnames, row)

        if args.save_ckpt and args.save_best_ckpt:
            ep_ret = row["episode_return_mean"]
            if ep_ret > best_return:
                best_return = ep_ret
                ckpt_path = ckpt_dir / f"best_seed_{args.seed}.pt"
                payload = {
                    "algo": args.algo,
                    "scenario": args.scenario,
                    "seed": args.seed,
                    "iteration": it,
                    "timesteps_total": timesteps_total,
                    "episode_return_mean": ep_ret,
                    "n_agents": args.n_agents,
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "mappo_critic": args.mappo_critic if args.algo == "mappo" else None,
                    "value_norm": args.value_norm,
                    "policy_state_dict": trainer.policy.state_dict(),
                    "value_state_dict": trainer.value.state_dict(),
                }
                torch.save(payload, ckpt_path)

        if it % args.log_every == 0:
            print(
                f"[{args.algo}] it={it} steps={timesteps_total} "
                f"ep_ret={row['episode_return_mean']:.3f} ep_len={row['episode_len_mean']:.1f} "
                f"sec={dt:.2f}"
            )

    if args.save_ckpt:
        ckpt_path = ckpt_dir / f"final_seed_{args.seed}.pt"
        payload = {
            "algo": args.algo,
            "scenario": args.scenario,
            "seed": args.seed,
            "iteration": args.iters,
            "timesteps_total": timesteps_total,
            "episode_return_mean": row["episode_return_mean"],
            "n_agents": args.n_agents,
            "obs_dim": obs_dim,
            "act_dim": act_dim,
            "mappo_critic": args.mappo_critic if args.algo == "mappo" else None,
            "value_norm": args.value_norm,
            "policy_state_dict": trainer.policy.state_dict(),
            "value_state_dict": trainer.value.state_dict(),
        }
        torch.save(payload, ckpt_path)

    print(f"Done. Logs: {csv_path}")
    if args.save_ckpt:
        print(f"Checkpoint dir: {ckpt_dir}")


if __name__ == "__main__":
    main()


