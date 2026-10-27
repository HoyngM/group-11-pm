"""Tables and box plots of the location importance.

Input of every function: the long table from ``CrossValidationProcessPermutation.run`` with one
row per (itemset, fold, repetition) and the columns ``itemset``, ``size``, ``fold``, ``iter``,
``importance``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write image files; no window needed
import matplotlib.pyplot as plt  # noqa: E402  (must follow the backend choice)
import pandas as pd  # noqa: E402


def summarise(importance: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of the F1 drop per itemset, most important first."""
    # One row per itemset with the mean and std over its 50 values (5 folds x 10 repetitions).
    summary = importance.groupby(["itemset", "size"])["importance"].agg(["mean", "std"]).reset_index()
    return summary.sort_values("mean", ascending=False).reset_index(drop=True)


def to_wide(importance: pd.DataFrame) -> pd.DataFrame:
    """One column per itemset and one row per (fold, iter), ordered by mean importance."""
    # pivot_table turns the long table (one row per value) into a wide one (one column per itemset).
    location_importance = importance.pivot_table(index=["fold", "iter"], columns="itemset", values="importance")
    # Most important column first.
    return location_importance[location_importance.mean().sort_values(ascending=False).index]


def box_plot(importance: pd.DataFrame, path: Path, title: str, top: int = 20) -> None:
    """Box plot of the ``top`` most important itemsets (decrease in F1), saved to ``path``."""
    location_importance = to_wide(importance).iloc[:, :top]  # the first ``top`` columns = most important
    # The figure grows with the number of itemsets, so that the labels stay readable.
    figure, axes = plt.subplots(figsize=(11, max(3, 0.4 * location_importance.shape[1] + 1.5)))
    # One horizontal box per itemset; whis=10 stretches the whiskers over (almost) all values.
    axes.boxplot([location_importance[column] for column in location_importance.columns],
                 orientation="horizontal", whis=10, tick_labels=list(location_importance.columns))
    axes.invert_yaxis()  # most important itemset at the top
    axes.axvline(0, color="k", linestyle="--")  # dashed line at "no effect"
    axes.set_xlabel("Decrease in f1 score")
    axes.set_title(title)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)  # free the memory; many plots are made in one run
