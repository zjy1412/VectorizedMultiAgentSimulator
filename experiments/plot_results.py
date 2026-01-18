from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot aggregated results (mean±std) across seeds.")
    p.add_argument("--scenario", type=str, default="balance")
    p.add_argument("--out_dir", type=str, default="results")
    p.add_argument("--algos", nargs="+", default=["cppo", "mappo", "ippo"])
    p.add_argument("--seeds", type=int, default=10)
    return p.parse_args()


def _read_series(csv_path: Path, key: str = "episode_return_mean") -> List[float]:
    xs: List[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row[key]))
    return xs


def _mean_std(values: List[List[float]]) -> Tuple[List[float], List[float]]:
    # values: [seeds][iters]
    import math

    iters = min(len(v) for v in values)
    mean = []
    std = []
    for t in range(iters):
        vs = [v[t] for v in values]
        m = sum(vs) / len(vs)
        mean.append(m)
        var = sum((x - m) ** 2 for x in vs) / len(vs)
        std.append(math.sqrt(var))
    return mean, std


def main() -> None:
    args = _parse_args()
    root = Path(args.out_dir) / args.scenario
    fig_dir = Path(args.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Lazy import matplotlib to avoid hard dependency during training.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for algo in args.algos:
        series = []
        for seed in range(args.seeds):
            csv_path = root / algo / f"seed_{seed}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing: {csv_path}")
            series.append(_read_series(csv_path))
        mean, std = _mean_std(series)
        xs = list(range(1, len(mean) + 1))
        plt.plot(xs, mean, label=algo)
        plt.fill_between(xs, [m - s for m, s in zip(mean, std)], [m + s for m, s in zip(mean, std)], alpha=0.2)

    plt.xlabel("Training iteration")
    plt.ylabel("Mean episode return")
    plt.title(f"{args.scenario}: CPPO vs MAPPO vs IPPO (mean±std over seeds)")
    plt.legend()
    out_path = fig_dir / f"{args.scenario}_cppo_mappo_ippo.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()


