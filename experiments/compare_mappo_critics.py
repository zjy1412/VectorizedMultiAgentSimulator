from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from pathlib import Path
from typing import List


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MAPPO critics: concat vs DeepSets.")
    p.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    p.add_argument("--buffer_device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--train_batch_size", type=int, default=12000)
    p.add_argument("--num_envs", type=int, default=60)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_agents_list", nargs="+", type=int, default=[3, 4, 6, 8])
    p.add_argument("--out_dir", type=str, default="results_compare")
    return p.parse_args()


def _read_last_return(csv_path: Path) -> float:
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        last = None
        for row in r:
            last = row
        if last is None:
            raise ValueError(f"Empty csv: {csv_path}")
        return float(last["episode_return_mean"])


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parents[1]

    critics = ["concat", "deepsets"]
    summary_rows = []

    for n_agents in args.n_agents_list:
        for critic in critics:
            out_dir = Path(args.out_dir) / f"n_agents_{n_agents}" / critic
            cmd = [
                sys.executable,
                "-m",
                "experiments.train_balance",
                "--algo",
                "mappo",
                "--mappo_critic",
                critic,
                "--seed",
                str(args.seed),
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
                str(n_agents),
                "--out_dir",
                str(out_dir),
                "--log_every",
                "10",
            ]
            print(f"Running: {' '.join(cmd)}")
            subprocess.check_call(cmd, cwd=str(root))

            csv_path = out_dir / "balance" / "mappo" / f"seed_{args.seed}.csv"
            last_ret = _read_last_return(csv_path)
            summary_rows.append(
                {"n_agents": n_agents, "critic": critic, "last_episode_return_mean": last_ret}
            )

    summary_path = Path(args.out_dir) / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["n_agents", "critic", "last_episode_return_mean"])
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()


