from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Overlay results from multiple out_dir roots on the same plot (mean±std over seeds)."
    )
    p.add_argument(
        "--out_dirs",
        nargs="+",
        required=True,
        help="One or more result roots produced by experiments/train_balance.py or experiments/run_paper_like.py.",
    )
    p.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels for each out_dir. Defaults to folder names.",
    )
    p.add_argument("--scenario", type=str, default="balance")
    p.add_argument("--algo", type=str, default="mappo", choices=["cppo", "mappo", "ippo"])
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--metric", type=str, default="episode_return_mean")
    p.add_argument("--out_dir", type=str, default="results_compare")
    return p.parse_args()


def _read_series(csv_path: Path, key: str) -> List[float]:
    xs: List[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            xs.append(float(row[key]))
    return xs


def _mean_std(values: List[List[float]]) -> Tuple[List[float], List[float]]:
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
    out_dirs = [Path(p) for p in args.out_dirs]
    if args.labels is None:
        labels = [p.name for p in out_dirs]
    else:
        if len(args.labels) != len(out_dirs):
            raise ValueError("If provided, --labels must have the same length as --out_dirs.")
        labels = args.labels

    # Lazy import matplotlib to keep training dependencies minimal.
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    for root, label in zip(out_dirs, labels):
        series = []
        for seed in range(args.seeds):
            csv_path = root / args.scenario / args.algo / f"seed_{seed}.csv"
            if not csv_path.exists():
                raise FileNotFoundError(f"Missing: {csv_path}")
            series.append(_read_series(csv_path, args.metric))
        mean, std = _mean_std(series)
        xs = list(range(1, len(mean) + 1))
        plt.plot(xs, mean, label=label)
        plt.fill_between(
            xs,
            [m - s for m, s in zip(mean, std)],
            [m + s for m, s in zip(mean, std)],
            alpha=0.2,
        )

    plt.xlabel("Training iteration")
    plt.ylabel(args.metric)
    plt.title(f"{args.scenario}/{args.algo}: compare runs (mean±std over seeds)")
    plt.legend()

    fig_dir = Path(args.out_dir) / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    safe_labels = "_vs_".join([l.replace("/", "_") for l in labels])
    out_path = fig_dir / f"{args.scenario}_{args.algo}_{safe_labels}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

