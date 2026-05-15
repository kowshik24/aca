from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def ensure_dirs(root: str) -> dict[str, Path]:
    base = Path(root)
    paths = {
        "root": base,
        "tables": base / "tables",
        "figures": base / "figures",
        "checkpoints": base / "checkpoints",
        "logs": base / "logs",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def write_metrics_table(path: Path, rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def plot_faithfulness(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="split", y="faithfulness", hue="method")
    plt.ylim(0, 1)
    plt.title("Counterfactual Faithfulness by Split")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_emi(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="split", y="emi", hue="method")
    plt.ylim(0, 1)
    plt.title("Sufficiency (EMI proxy) by Split")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()


def plot_accuracy(df: pd.DataFrame, out_path: Path) -> None:
    plt.figure(figsize=(9, 5))
    sns.barplot(data=df, x="split", y="acc", hue="method")
    plt.ylim(0, 1)
    plt.title("Intervened Accuracy by Split")
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
