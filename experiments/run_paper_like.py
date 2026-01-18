from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run paper-like sweeps (10 seeds) for Balance.")
    p.add_argument("--algo", choices=["ippo", "mappo", "cppo"], required=True)
    p.add_argument("--seeds", type=int, default=10)
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--buffer_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--train_batch_size", type=int, default=60000)
    p.add_argument("--num_envs", type=int, default=300)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--n_agents", type=int, default=4)
    p.add_argument("--out_dir", type=str, default="results")
    # Forward any additional args to `experiments.train_balance`:
    # Example:
    #   python -m experiments.run_paper_like ... -- --value_norm --no-save_best_ckpt
    p.add_argument(
        "train_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to experiments.train_balance (prefix with `--`).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]

    for seed in range(args.seeds):
        cmd = [
            sys.executable,
            "-m",
            "experiments.train_balance",
            "--algo",
            args.algo,
            "--seed",
            str(seed),
            "--device",
            args.device,
            "--buffer_device",
            args.buffer_device,
            "--iters",
            str(args.iters),
            "--train_batch_size",
            str(args.train_batch_size),
            "--num_envs",
            str(args.num_envs),
            "--max_steps",
            str(args.max_steps),
            "--n_agents",
            str(args.n_agents),
            "--out_dir",
            args.out_dir,
        ]
        if args.train_args:
            # Depending on how argparse is invoked, the remainder may include a literal "--".
            extra = list(args.train_args)
            if extra and extra[0] == "--":
                extra = extra[1:]
            cmd.extend(extra)
        print(f"Running seed {seed}: {' '.join(cmd)}")
        # Ensure `marl/` is importable by running from the repository root.
        subprocess.check_call(cmd, cwd=str(root))


if __name__ == "__main__":
    main()


