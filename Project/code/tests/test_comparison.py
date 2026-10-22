"""Comparison of two strategies on hand-made summaries."""

import pandas as pd
import pytest

from location_permutation.comparison import StrategyComparison, StrategySummary


def summary(name: str, rows: list[tuple[str, float]]) -> StrategySummary:
    table = pd.DataFrame(rows, columns=["itemset", "mean"])
    table["size"] = table.itemset.str.count(",") + 1
    table["std"] = 0.01
    return StrategySummary(name, table.sort_values("mean", ascending=False).reset_index(drop=True))


@pytest.fixture
def comparison() -> StrategyComparison:
    apriori = summary("apriori", [("a", 0.30), ("b", 0.20), ("c", 0.10), ("a, b", 0.05)])
    impressed = summary("impressed", [("a", 0.25), ("b", 0.15), ("c", 0.05), ("c, d", 0.40)])
    return StrategyComparison(apriori, impressed, top_k=2)


def test_overlap_and_rank_correlation(comparison):
    metrics = comparison.metrics()
    assert metrics["itemsets_shared"] == 3
    assert metrics["jaccard_itemsets"] == pytest.approx(3 / 5)
    assert metrics["top2_overlap"] == 1  # {a, b} against {c, d}, a
    assert metrics["spearman_shared"] == pytest.approx(1.0)  # a > b > c in both


def test_itemsets_only_found_by_the_second_strategy(comparison):
    metrics = comparison.metrics()
    assert metrics["best_impressed"] == "c, d"
    assert metrics["impressed_only_above_best_apriori"] == 1


def test_top_table_marks_shared_itemsets(comparison):
    table = comparison.top_table()
    assert list(table.apriori_also_selected_by_other) == [True, True]
    assert list(table.impressed_also_selected_by_other) == [False, True]
