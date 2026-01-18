from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Tuple

import torch
import torch.nn as nn
from torch import Tensor

from marl.networks import DeepSetsCritic, TanhGaussianPolicy, ValueNetwork
from marl.ppo import PPOHyperParams, compute_gae, normalize_advantages
from marl.utils import concat_agent_obs
from marl.value_norm import ValueNorm


AlgoName = Literal["ippo", "mappo", "cppo"]


@dataclass
class TrainConfig:
    algo: AlgoName
    device: torch.device
    buffer_device: torch.device
    n_agents: int
    obs_dim: int
    act_dim: int
    total_act_dim: int
    total_obs_dim: int
    hyper: PPOHyperParams
    # Only used by MAPPO.
    mappo_critic: Literal["concat", "deepsets"] = "concat"
    # Stabilization trick: normalize value loss targets/predictions using running return statistics.
    value_norm: bool = False


def _maybe_to(x: Tensor, device: torch.device) -> Tensor:
    return x.to(device=device, non_blocking=True)


class IPPO:
    """Independent PPO with parameter sharing (one actor/critic for all agents)."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.policy = TanhGaussianPolicy(cfg.obs_dim, cfg.act_dim).to(cfg.device)
        self.value = ValueNetwork(cfg.obs_dim).to(cfg.device)
        self.value_norm = ValueNorm(device=torch.device("cpu")) if cfg.value_norm else None
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=cfg.hyper.lr,
        )

    @torch.no_grad()
    def act(self, obs: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        # obs: list[n_agents] each [num_envs, obs_dim]
        actions = []
        logps = []
        values = []
        for i in range(self.cfg.n_agents):
            o = obs[i]
            a, logp, _ = self.policy.act(o, deterministic=False)
            v = self.value(o)
            actions.append(a)
            logps.append(logp)
            values.append(v)
        # Stack agent-wise for storage: [num_envs, n_agents]
        logps_t = torch.stack(logps, dim=-1)
        values_t = torch.stack(values, dim=-1)
        return actions, logps_t, values_t

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        # batch shapes:
        # obs: [T, num_envs, n_agents, obs_dim]
        # act: [T, num_envs, n_agents, act_dim]
        # logp: [T, num_envs, n_agents]
        # rew: [T, num_envs, n_agents]
        # done: [T, num_envs]
        # val: [T, num_envs, n_agents]
        hp = self.cfg.hyper
        obs = _maybe_to(batch["obs"], self.cfg.device)
        act = _maybe_to(batch["act"], self.cfg.device)
        old_logp = _maybe_to(batch["logp"], self.cfg.device)
        rew = _maybe_to(batch["rew"], self.cfg.device)
        done = _maybe_to(batch["done"], self.cfg.device)
        val = _maybe_to(batch["val"], self.cfg.device)
        last_obs = _maybe_to(batch["last_obs"], self.cfg.device)  # [num_envs, n_agents, obs_dim]

        with torch.no_grad():
            last_val = self.value(last_obs.reshape(-1, self.cfg.obs_dim)).reshape(
                -1, self.cfg.n_agents
            )  # [num_envs, n_agents]

        # Compute per-agent advantages/returns using the same done mask per env.
        adv, ret = compute_gae(
            rewards=rew,
            dones=done.unsqueeze(-1).expand_as(rew),
            values=val,
            last_value=last_val,
            gamma=hp.gamma,
            gae_lambda=hp.gae_lambda,
        )
        adv = normalize_advantages(adv)
        if self.value_norm is not None:
            # Update using raw returns (CPU stats to keep it cheap and deterministic across devices).
            self.value_norm.update(ret)

        # Flatten (T, num_envs, n_agents) -> B
        T, E, A = rew.shape
        B = T * E * A
        obs_f = obs.reshape(B, self.cfg.obs_dim)
        act_f = act.reshape(B, self.cfg.act_dim)
        old_logp_f = old_logp.reshape(B)
        adv_f = adv.reshape(B)
        ret_f = ret.reshape(B)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        for _ in range(hp.num_sgd_iter):
            idx = torch.randperm(B, device=self.cfg.device)
            for start in range(0, B, hp.sgd_minibatch_size):
                mb = idx[start : start + hp.sgd_minibatch_size]
                mb_obs = obs_f[mb]
                mb_act = act_f[mb]
                mb_old_logp = old_logp_f[mb]
                mb_adv = adv_f[mb]
                mb_ret = ret_f[mb]

                new_logp = self.policy.log_prob(mb_obs, mb_act)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - hp.clip_param, 1.0 + hp.clip_param) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                new_v = self.value(mb_obs)
                if hp.vf_clip_param != float("inf"):
                    v_clipped = mb_ret + (new_v - mb_ret).clamp(-hp.vf_clip_param, hp.vf_clip_param)
                    value_loss = 0.5 * torch.max((new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()
                else:
                    if self.value_norm is None:
                        value_loss = 0.5 * (new_v - mb_ret).pow(2).mean()
                    else:
                        value_loss = 0.5 * (
                            self.value_norm.normalize(new_v) - self.value_norm.normalize(mb_ret)
                        ).pow(2).mean()

                ent = self.policy.dist(mb_obs).entropy().mean()
                loss = policy_loss + hp.vf_loss_coeff * value_loss - hp.entropy_coeff * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    hp.max_grad_norm,
                )
                self.opt.step()

                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(ent.detach().cpu())
                metrics["total_loss"] += float(loss.detach().cpu())

        denom = float(hp.num_sgd_iter * max(1, (B + hp.sgd_minibatch_size - 1) // hp.sgd_minibatch_size))
        return {k: v / denom for k, v in metrics.items()}


class MAPPO:
    """MAPPO: decentralized actor (shared), centralized critic."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.policy = TanhGaussianPolicy(cfg.obs_dim, cfg.act_dim).to(cfg.device)
        if cfg.mappo_critic == "concat":
            self.value = ValueNetwork(cfg.total_obs_dim).to(cfg.device)
        elif cfg.mappo_critic == "deepsets":
            self.value = DeepSetsCritic(cfg.obs_dim).to(cfg.device)
        else:
            raise ValueError(f"Unsupported MAPPO critic: {cfg.mappo_critic}")
        self.value_norm = ValueNorm(device=torch.device("cpu")) if cfg.value_norm else None
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=cfg.hyper.lr,
        )

    @torch.no_grad()
    def act(self, obs: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        # Centralized critic uses concatenated obs.
        if self.cfg.mappo_critic == "concat":
            global_obs = concat_agent_obs(obs)
            v = self.value(global_obs)  # [num_envs]
        else:
            stacked = torch.stack(obs, dim=1)  # [num_envs, n_agents, obs_dim]
            v = self.value(stacked)
        actions = []
        logps = []
        for i in range(self.cfg.n_agents):
            a, logp, _ = self.policy.act(obs[i], deterministic=False)
            actions.append(a)
            logps.append(logp)
        logps_t = torch.stack(logps, dim=-1)  # [num_envs, n_agents]
        return actions, logps_t, v

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        # batch shapes:
        # obs: [T, num_envs, n_agents, obs_dim]
        # gobs: [T, num_envs, total_obs_dim] (only needed if critic == "concat")
        # act: [T, num_envs, n_agents, act_dim]
        # logp: [T, num_envs, n_agents]
        # team_rew: [T, num_envs]
        # done: [T, num_envs]
        # val: [T, num_envs]
        hp = self.cfg.hyper
        obs = _maybe_to(batch["obs"], self.cfg.device)
        gobs = _maybe_to(batch["gobs"], self.cfg.device) if "gobs" in batch else None
        act = _maybe_to(batch["act"], self.cfg.device)
        old_logp = _maybe_to(batch["logp"], self.cfg.device)
        team_rew = _maybe_to(batch["team_rew"], self.cfg.device)
        done = _maybe_to(batch["done"], self.cfg.device)
        val = _maybe_to(batch["val"], self.cfg.device)
        last_gobs = _maybe_to(batch["last_gobs"], self.cfg.device) if "last_gobs" in batch else None

        with torch.no_grad():
            if self.cfg.mappo_critic == "concat":
                assert last_gobs is not None
                last_val = self.value(last_gobs)  # [num_envs]
            else:
                # last obs is already available in the batch (stored via obs[-1] at rollout time)
                last_obs = _maybe_to(batch["last_obs"], self.cfg.device)  # [num_envs, n_agents, obs_dim]
                last_val = self.value(last_obs)

        adv, ret = compute_gae(team_rew, done, val, last_val, hp.gamma, hp.gae_lambda)
        adv = normalize_advantages(adv)
        if self.value_norm is not None:
            self.value_norm.update(ret)

        # Actor update uses per-agent logp but shared team advantage.
        T, E = team_rew.shape
        A = self.cfg.n_agents
        B_actor = T * E * A
        obs_f = obs.reshape(B_actor, self.cfg.obs_dim)
        act_f = act.reshape(B_actor, self.cfg.act_dim)
        old_logp_f = old_logp.reshape(B_actor)
        adv_f = adv.unsqueeze(-1).expand(T, E, A).reshape(B_actor)

        # Critic update uses env-level samples.
        B_critic = T * E
        ret_f = ret.reshape(B_critic)
        if self.cfg.mappo_critic == "concat":
            assert gobs is not None
            critic_in = gobs.reshape(B_critic, self.cfg.total_obs_dim)
        else:
            critic_in = obs.reshape(B_critic, self.cfg.n_agents, self.cfg.obs_dim)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}

        for _ in range(hp.num_sgd_iter):
            idx_actor = torch.randperm(B_actor, device=self.cfg.device)
            idx_critic = torch.randperm(B_critic, device=self.cfg.device)

            # Interleave actor and critic minibatches to keep code simple.
            mb_steps = max(
                (B_actor + hp.sgd_minibatch_size - 1) // hp.sgd_minibatch_size,
                (B_critic + hp.sgd_minibatch_size - 1) // hp.sgd_minibatch_size,
            )
            for i in range(mb_steps):
                mb_a = idx_actor[
                    i * hp.sgd_minibatch_size : (i + 1) * hp.sgd_minibatch_size
                ]
                mb_c = idx_critic[
                    i * hp.sgd_minibatch_size : (i + 1) * hp.sgd_minibatch_size
                ]

                policy_loss = torch.tensor(0.0, device=self.cfg.device)
                ent = torch.tensor(0.0, device=self.cfg.device)
                if mb_a.numel() > 0:
                    mb_obs = obs_f[mb_a]
                    mb_act = act_f[mb_a]
                    mb_old_logp = old_logp_f[mb_a]
                    mb_adv = adv_f[mb_a]
                    new_logp = self.policy.log_prob(mb_obs, mb_act)
                    ratio = torch.exp(new_logp - mb_old_logp)
                    surr1 = ratio * mb_adv
                    surr2 = torch.clamp(
                        ratio, 1.0 - hp.clip_param, 1.0 + hp.clip_param
                    ) * mb_adv
                    policy_loss = -torch.min(surr1, surr2).mean()
                    ent = self.policy.dist(mb_obs).entropy().mean()

                value_loss = torch.tensor(0.0, device=self.cfg.device)
                if mb_c.numel() > 0:
                    mb_ret = ret_f[mb_c]
                    mb_in = critic_in[mb_c]
                    new_v = self.value(mb_in)
                    if hp.vf_clip_param != float("inf"):
                        v_clipped = mb_ret + (new_v - mb_ret).clamp(
                            -hp.vf_clip_param, hp.vf_clip_param
                        )
                        value_loss = 0.5 * torch.max(
                            (new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)
                        ).mean()
                    else:
                        if self.value_norm is None:
                            value_loss = 0.5 * (new_v - mb_ret).pow(2).mean()
                        else:
                            value_loss = 0.5 * (
                                self.value_norm.normalize(new_v)
                                - self.value_norm.normalize(mb_ret)
                            ).pow(2).mean()

                loss = policy_loss + hp.vf_loss_coeff * value_loss - hp.entropy_coeff * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    hp.max_grad_norm,
                )
                self.opt.step()

                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(ent.detach().cpu())
                metrics["total_loss"] += float(loss.detach().cpu())

        denom = float(hp.num_sgd_iter * mb_steps)
        return {k: v / denom for k, v in metrics.items()}


class CPPO:
    """Centralized PPO: single 'super-agent' controlling all agents jointly."""

    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.policy = TanhGaussianPolicy(cfg.total_obs_dim, cfg.total_act_dim).to(cfg.device)
        self.value = ValueNetwork(cfg.total_obs_dim).to(cfg.device)
        self.value_norm = ValueNorm(device=torch.device("cpu")) if cfg.value_norm else None
        self.opt = torch.optim.Adam(
            list(self.policy.parameters()) + list(self.value.parameters()),
            lr=cfg.hyper.lr,
        )

    @torch.no_grad()
    def act(self, obs: List[Tensor]) -> Tuple[List[Tensor], Tensor, Tensor]:
        gobs = concat_agent_obs(obs)
        a_joint, logp, _ = self.policy.act(gobs, deterministic=False)  # [num_envs, total_act_dim]
        v = self.value(gobs)

        # Split joint action back into per-agent actions.
        actions = []
        cursor = 0
        for _ in range(self.cfg.n_agents):
            actions.append(a_joint[:, cursor : cursor + self.cfg.act_dim])
            cursor += self.cfg.act_dim
        # For consistency with other algos, return logp expanded to [num_envs, n_agents]
        # (each agent shares the same joint logp).
        logps = logp.unsqueeze(-1).expand(-1, self.cfg.n_agents)
        return actions, logps, v

    def update(self, batch: Dict[str, Tensor]) -> Dict[str, float]:
        # batch shapes:
        # gobs: [T, num_envs, total_obs_dim]
        # a_joint: [T, num_envs, total_act_dim]
        # logp_joint: [T, num_envs]
        # team_rew: [T, num_envs]
        # done: [T, num_envs]
        # val: [T, num_envs]
        hp = self.cfg.hyper
        gobs = _maybe_to(batch["gobs"], self.cfg.device)
        a_joint = _maybe_to(batch["a_joint"], self.cfg.device)
        old_logp = _maybe_to(batch["logp_joint"], self.cfg.device)
        team_rew = _maybe_to(batch["team_rew"], self.cfg.device)
        done = _maybe_to(batch["done"], self.cfg.device)
        val = _maybe_to(batch["val"], self.cfg.device)
        last_gobs = _maybe_to(batch["last_gobs"], self.cfg.device)

        with torch.no_grad():
            last_val = self.value(last_gobs)

        adv, ret = compute_gae(team_rew, done, val, last_val, hp.gamma, hp.gae_lambda)
        adv = normalize_advantages(adv)
        if self.value_norm is not None:
            self.value_norm.update(ret)

        T, E = team_rew.shape
        B = T * E
        gobs_f = gobs.reshape(B, self.cfg.total_obs_dim)
        a_f = a_joint.reshape(B, self.cfg.total_act_dim)
        old_logp_f = old_logp.reshape(B)
        adv_f = adv.reshape(B)
        ret_f = ret.reshape(B)

        metrics = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "total_loss": 0.0}
        for _ in range(hp.num_sgd_iter):
            idx = torch.randperm(B, device=self.cfg.device)
            for start in range(0, B, hp.sgd_minibatch_size):
                mb = idx[start : start + hp.sgd_minibatch_size]
                mb_obs = gobs_f[mb]
                mb_act = a_f[mb]
                mb_old_logp = old_logp_f[mb]
                mb_adv = adv_f[mb]
                mb_ret = ret_f[mb]

                new_logp = self.policy.log_prob(mb_obs, mb_act)
                ratio = torch.exp(new_logp - mb_old_logp)
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - hp.clip_param, 1.0 + hp.clip_param) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                new_v = self.value(mb_obs)
                if hp.vf_clip_param != float("inf"):
                    v_clipped = mb_ret + (new_v - mb_ret).clamp(-hp.vf_clip_param, hp.vf_clip_param)
                    value_loss = 0.5 * torch.max((new_v - mb_ret).pow(2), (v_clipped - mb_ret).pow(2)).mean()
                else:
                    if self.value_norm is None:
                        value_loss = 0.5 * (new_v - mb_ret).pow(2).mean()
                    else:
                        value_loss = 0.5 * (
                            self.value_norm.normalize(new_v) - self.value_norm.normalize(mb_ret)
                        ).pow(2).mean()

                ent = self.policy.dist(mb_obs).entropy().mean()
                loss = policy_loss + hp.vf_loss_coeff * value_loss - hp.entropy_coeff * ent

                self.opt.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value.parameters()),
                    hp.max_grad_norm,
                )
                self.opt.step()

                metrics["policy_loss"] += float(policy_loss.detach().cpu())
                metrics["value_loss"] += float(value_loss.detach().cpu())
                metrics["entropy"] += float(ent.detach().cpu())
                metrics["total_loss"] += float(loss.detach().cpu())

        denom = float(hp.num_sgd_iter * max(1, (B + hp.sgd_minibatch_size - 1) // hp.sgd_minibatch_size))
        return {k: v / denom for k, v in metrics.items()}


def make_trainer(cfg: TrainConfig):
    if cfg.algo == "ippo":
        return IPPO(cfg)
    if cfg.algo == "mappo":
        return MAPPO(cfg)
    if cfg.algo == "cppo":
        return CPPO(cfg)
    raise ValueError(f"Unknown algo: {cfg.algo}")


