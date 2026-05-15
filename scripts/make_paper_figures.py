from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from acar.reporting import plot_accuracy, plot_emi, plot_faithfulness


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/tables/metrics.csv")
    parser.add_argument("--fig-dir", default="results/figures")
    args = parser.parse_args()

    df = pd.read_csv(args.metrics)
    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    plot_faithfulness(df, fig_dir / "faithfulness.png")
    plot_emi(df, fig_dir / "emi.png")
    plot_accuracy(df, fig_dir / "accuracy.png")


if __name__ == "__main__":
    main()
