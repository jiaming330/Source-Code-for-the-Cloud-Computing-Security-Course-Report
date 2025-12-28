"""Plot MOON vs FedAvg curves.

Usage (from repo root):
  python code/plot_moon_vs_fedavg.py

Or specify paths:
  python code/plot_moon_vs_fedavg.py --moon results/moon_curve.csv --fedavg results/fedavg_curve.csv --out results/moon_vs_fedavg.png
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def load_csv(path: Path):
    xs, ys = [], []
    with path.open(newline="", encoding="utf-8") as f:
        for r in csv.reader(f):
            if not r:
                continue
            xs.append(int(r[0]))
            ys.append(float(r[1]))
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--moon", type=str, default="results/moon_curve.csv")
    ap.add_argument("--fedavg", type=str, default="results/fedavg_curve.csv")
    ap.add_argument("--out", type=str, default="results/moon_vs_fedavg.png")
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[1]
    moon_path = (repo_root / args.moon).resolve()
    fed_path = (repo_root / args.fedavg).resolve()
    out_path = (repo_root / args.out).resolve()

    mx, my = load_csv(moon_path)
    fx, fy = load_csv(fed_path)

    plt.figure()
    plt.plot(mx, my, label="MOON")
    plt.plot(fx, fy, label="FedAvg")
    plt.xlabel("Communication Round")
    plt.ylabel("Global Test Accuracy")
    plt.legend()
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
