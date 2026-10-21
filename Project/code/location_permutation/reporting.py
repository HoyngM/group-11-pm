"""Tables and box plots of the location importance, in the style of the reference output."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files; no window needed
import matplotlib.pyplot as plt
import pandas as pd


def summarise(importance: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of the F1 drop per itemset, most important first."""
    summary = importance.groupby(["itemset", "size"])["importance"].agg(["mean", "std"]).reset_index()
    return summary.sort_values("mean", ascending=False).reset_index(drop=True)


def to_wide(importance: pd.DataFrame) -> pd.DataFrame:
    """One column per itemset, one row per (fold, repetition), columns ordered by mean importance."""
    wide = importance.pivot_table(index=["fold", "repetition"], columns="itemset", values="importance")
    return wide[wide.mean().sort_values(ascending=False).index]


def box_plot(importance: pd.DataFrame, path: Path, title: str, top: int = 20) -> None:
    """Box plot of the ``top`` most important itemsets (decrease in F1)."""
    wide = to_wide(importance).iloc[:, :top]
    figure, axes = plt.subplots(figsize=(11, max(3, 0.4 * wide.shape[1] + 1.5)))
    wide.plot.box(vert=False, whis=10, ax=axes)
    axes.invert_yaxis()
    axes.axvline(0, color="k", linestyle="--")
    axes.set_xlabel("Decrease in f1 score")
    axes.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
