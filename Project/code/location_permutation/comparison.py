"""Comparison of the location importance of two itemset selection strategies.

Why this module exists
----------------------
The assignment asks whether the importance of activities and their locations changes when
the itemsets are chosen by IMPresseD instead of by Apriori. ``run_location_importance.py``
computes, per dataset and strategy, the location importance of every selected itemset: the
mean drop in F1 over 5 folds x 10 permutations. It writes these numbers to
``results/importance/<dataset>/evaluated_on_<train|test>/<strategy>_summary.csv``.

This module reads two of those summary files (one per strategy) and puts them side by side.
It does not train models or permute anything itself, so it runs in seconds.

What is compared (see ``StrategyComparison.metrics``)
-----------------------------------------------------
1. *Which itemsets are tested*: the number of itemsets per strategy, how many itemsets both
   strategies selected, and the Jaccard similarity of the two itemset lists.
2. *Which itemsets come out on top*: how many of the ``top_k`` most important itemsets are
   the same, and the Jaccard similarity of the activities that appear in those top-k lists.
   The second number is looser: ``{a, b}`` and ``{a}`` share activity ``a``.
3. *Whether the importance itself changes*: the Spearman rank correlation of the mean F1 drop
   of the itemsets that both strategies selected. A value near 1 means that an itemset gets
   the same rank whichever strategy selected it. This is a sanity check: both strategies use
   the same model and the same permutation, so they should agree on shared itemsets.
4. *Whether the variant finds something new*: the best itemset of each strategy, and how many
   itemsets that only the second strategy selected beat the best itemset of the first.

Output (see ``ComparisonReport.run``)
-------------------------------------
* ``comparison_metrics.csv``: one row per (dataset, evaluation setting) with the metrics above.
* ``<dataset>_<train|test>_top<k>.csv``: the top-k itemsets of both strategies, rank by rank.
* ``comparison_<train|test>.png``: one panel per dataset with the mean F1 drop (and standard
  deviation) of the most important itemsets of both strategies.

Itemsets are written as strings with the activities sorted and joined by ``", "``, for
example ``"376400, ac370000"``, exactly as in the summary files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # write files; no window needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Marker shape and colour of the first (Apriori) and second (IMPresseD) strategy in the figure.
MARKERS = {0: ("o", "tab:blue"), 1: ("D", "tab:orange")}


def activities_of(itemsets: pd.Series) -> set[str]:
    """All activities that occur in the itemsets (written as ``"a, b, c"``)."""
    return {activity for itemset in itemsets for activity in itemset.split(", ")}


def jaccard(first: set, second: set) -> float:
    """Size of the intersection divided by the size of the union (1 = identical, 0 = disjoint)."""
    union = first | second
    return len(first & second) / len(union) if union else float("nan")


@dataclass
class StrategySummary:
    """The location importance of all itemsets of one strategy on one dataset.

    ``table`` has the columns of ``<strategy>_summary.csv``: ``itemset``, ``size`` (number of
    activities), ``mean`` and ``std`` (of the F1 drop over folds and repetitions). The rows are
    sorted from most to least important.
    """

    name: str
    table: pd.DataFrame

    @classmethod
    def from_folder(cls, folder: Path, name: str) -> "StrategySummary":
        """Read ``<folder>/<name>_summary.csv`` as written by ``run_location_importance.py``."""
        table = pd.read_csv(folder / f"{name}_summary.csv")
        return cls(name, table.sort_values("mean", ascending=False).reset_index(drop=True))

    def top(self, k: int) -> pd.DataFrame:
        """The ``k`` itemsets with the largest mean F1 drop."""
        return self.table.head(k)


class StrategyComparison:
    """Compares the location importance of two strategies on one dataset.

    ``first`` is the reference (Apriori), ``second`` the variant (IMPresseD). ``top_k`` is the
    number of most important itemsets that is compared; the rank correlation uses every
    itemset that both strategies selected.
    """

    def __init__(self, first: StrategySummary, second: StrategySummary, top_k: int = 10) -> None:
        self.first, self.second, self.top_k = first, second, top_k

    def metrics(self) -> dict[str, float]:
        """All comparison numbers for this dataset as one dictionary (one row of the metrics table)."""
        first, second = self.first.table, self.second.table
        # Itemsets that both strategies selected, with the importance under each strategy.
        shared = first.merge(second, on="itemset", suffixes=("_first", "_second"))
        top_first, top_second = self.first.top(self.top_k), self.second.top(self.top_k)
        top_shared = set(top_first.itemset) & set(top_second.itemset)

        # Spearman needs at least three pairs and some variation in both columns; otherwise
        # the correlation is undefined and reported as NaN.
        if len(shared) >= 3 and shared["mean_first"].nunique() > 1 and shared["mean_second"].nunique() > 1:
            correlation, p_value = spearmanr(shared["mean_first"], shared["mean_second"])
        else:
            correlation, p_value = float("nan"), float("nan")

        a, b = self.first.name, self.second.name
        return {
            # 1. Which itemsets are tested
            f"itemsets_{a}": len(first),
            f"itemsets_{b}": len(second),
            "itemsets_shared": len(shared),
            "jaccard_itemsets": jaccard(set(first.itemset), set(second.itemset)),
            # 2. Which itemsets come out on top
            f"top{self.top_k}_overlap": len(top_shared),
            f"top{self.top_k}_jaccard_activities": jaccard(activities_of(top_first.itemset),
                                                           activities_of(top_second.itemset)),
            # 3. Whether the importance of the same itemset changes
            "spearman_shared": correlation,
            "spearman_p_value": p_value,
            # 4. Whether the variant finds something new
            f"best_{a}": first.itemset.iloc[0],
            f"best_{a}_importance": first["mean"].iloc[0],
            f"best_{b}": second.itemset.iloc[0],
            f"best_{b}_importance": second["mean"].iloc[0],
            # itemsets that only the second strategy selected and that beat the best of the first
            f"{b}_only_above_best_{a}": int(((~second.itemset.isin(first.itemset))
                                             & (second["mean"] > first["mean"].iloc[0])).sum()),
        }

    def top_table(self) -> pd.DataFrame:
        """The ``top_k`` itemsets of both strategies side by side, rank by rank.

        ``<strategy>_also_selected_by_other`` tells whether the other strategy tested the same
        itemset at all (not whether it is in the other top-k).
        """
        columns = []
        for summary in (self.first, self.second):
            top = summary.top(self.top_k)[["itemset", "mean", "std"]].reset_index(drop=True)
            in_other = self.second if summary is self.first else self.first
            top["also_selected_by_other"] = top.itemset.isin(in_other.table.itemset)
            columns.append(top.add_prefix(f"{summary.name}_"))
        table = pd.concat(columns, axis=1)
        table.insert(0, "rank", range(1, len(table) + 1))
        return table

    def plot(self, axes: plt.Axes, title: str, top: int = 8) -> None:
        """Mean F1 drop (with standard deviation) of the union of the ``top`` itemsets of both strategies.

        Each row is one itemset, sorted by its highest importance under either strategy. An
        itemset without a marker for a strategy was not selected by that strategy. When both
        markers are present they are drawn slightly apart so that they do not overlap.
        """
        union = list(dict.fromkeys([*self.first.top(top).itemset, *self.second.top(top).itemset]))
        tables = [s.table.set_index("itemset") for s in (self.first, self.second)]
        best = {itemset: max(t["mean"].get(itemset, -np.inf) for t in tables) for itemset in union}
        union.sort(key=best.get)  # matplotlib draws row 0 at the bottom: most important ends on top
        rows = np.arange(len(union))
        for index, (summary, table) in enumerate(zip((self.first, self.second), tables)):
            present = [i for i, itemset in enumerate(union) if itemset in table.index]
            values = table.loc[[union[i] for i in present]]
            marker, colour = MARKERS[index]
            axes.errorbar(values["mean"], rows[present] + (index - 0.5) * 0.25, xerr=values["std"],
                          fmt=marker, color=colour, capsize=2, markersize=5, label=summary.name)
        # Itemsets of three activities get one activity per line so the labels stay narrow.
        axes.set_yticks(rows, [u.replace(", ", ",\n") if u.count(",") > 1 else u for u in union], fontsize=7)
        axes.axvline(0, color="k", linestyle="--", linewidth=0.8)
        axes.set_xlabel("Decrease in F1")
        axes.set_title(title, fontsize=10)


class ComparisonReport:
    """Runs the comparison for several datasets and evaluation settings and writes the output.

    ``results_directory`` is the folder that ``run_location_importance.py`` wrote to
    (``results/importance`` by default). ``first`` and ``second`` are strategy names as used in
    ``importance_config.yaml``.
    """

    def __init__(self, results_directory: Path, first: str = "apriori", second: str = "impressed",
                 top_k: int = 10, plot_top: int = 8) -> None:
        self.results_directory = results_directory
        self.first, self.second = first, second
        self.top_k, self.plot_top = top_k, plot_top

    def comparison(self, dataset: str, evaluate_on: str) -> StrategyComparison:
        """Load the two summary files of one dataset and evaluation setting."""
        folder = self.results_directory / dataset / f"evaluated_on_{evaluate_on}"
        return StrategyComparison(StrategySummary.from_folder(folder, self.first),
                                  StrategySummary.from_folder(folder, self.second), self.top_k)

    def run(self, datasets: list[str], evaluations: list[str], output_directory: Path) -> pd.DataFrame:
        """Write all tables and figures to ``output_directory`` and return the metrics table.

        One figure is made per evaluation setting (``train`` or ``test``), with one panel per
        dataset, so the datasets can be compared at a glance.
        """
        output_directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for evaluate_on in evaluations:
            figure, all_axes = plt.subplots(1, len(datasets), figsize=(4.2 * len(datasets), 6.5), squeeze=False)
            for dataset, axes in zip(datasets, all_axes[0]):
                comparison = self.comparison(dataset, evaluate_on)
                rows.append({"dataset": dataset, "evaluated_on": evaluate_on, **comparison.metrics()})
                comparison.top_table().to_csv(
                    output_directory / f"{dataset}_{evaluate_on}_top{self.top_k}.csv", index=False)
                comparison.plot(axes, dataset, self.plot_top)
            handles, labels = all_axes[0][0].get_legend_handles_labels()
            figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
            figure.suptitle(f"Location importance per strategy (evaluated on {evaluate_on} cases)")
            figure.tight_layout(rect=(0, 0.04, 1, 1))
            figure.savefig(output_directory / f"comparison_{evaluate_on}.png", dpi=200)
            plt.close(figure)
        metrics = pd.DataFrame(rows)
        metrics.to_csv(output_directory / "comparison_metrics.csv", index=False)
        return metrics
