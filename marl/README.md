# MARL (Homework)

This folder contains a minimal, pure-PyTorch implementation of PPO-based MARL baselines used in the VMAS paper:

- **CPPO**: centralized actor + centralized critic (treats the multi-agent system as one "super-agent").
- **MAPPO**: decentralized actor (shared parameters) + centralized critic.
- **IPPO**: decentralized actor + decentralized critic (shared parameters).

## MAPPO critic improvement (DeepSets)

The standard MAPPO centralized critic often uses **concatenation** of all agents' observations:

1. Build a global state by concatenating per-agent observations.
2. Predict a single team value \(V(s)\).

While simple, this has two practical issues:

- **Scalability**: the critic input dimension grows linearly with the number of agents.
- **Order sensitivity**: concatenation depends on agent ordering; swapping agent indices changes the input.

This repo also includes an optional **DeepSets-style critic** (`DeepSetsCritic`) which is **permutation-invariant**:

- Encode each agent observation with a shared encoder.
- Pool (mean) across agents.
- Predict a single team value from the pooled embedding.

This is often a better fit when you want the critic to generalize across different team sizes and not depend on arbitrary agent indexing.

## How to use

The training entrypoint is:

- `experiments/train_balance.py`

For MAPPO, you can select the critic type:

- `--mappo_critic concat` (default; paper-style)
- `--mappo_critic deepsets` (improved critic)


