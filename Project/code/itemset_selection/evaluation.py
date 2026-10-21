"""Measures to compare selection strategies with each other and with a gold standard."""

from __future__ import annotations

from .selector import SelectionResult


def _activities(result: SelectionResult) -> set[str]:
    return set().union(*result.itemsets) if result.itemsets else set()


def compare_strategies(reference: SelectionResult, other: SelectionResult) -> dict:
    """How much do the selected itemsets and activities of two strategies overlap?"""
    itemsets_a, itemsets_b = set(reference.itemsets), set(other.itemsets)
    activities_a, activities_b = _activities(reference), _activities(other)
    itemset_union = itemsets_a | itemsets_b
    activity_union = activities_a | activities_b
    # Single activities dominate the totals, so also compare the larger itemsets alone.
    multi_a = {i for i in itemsets_a if len(i) > 1}
    multi_b = {i for i in itemsets_b if len(i) > 1}
    multi_union = multi_a | multi_b
    return {
        "shared_itemsets": len(itemsets_a & itemsets_b),
        "itemset_jaccard": (
            len(itemsets_a & itemsets_b) / len(itemset_union) if itemset_union else 0.0
        ),
        "shared_multi_itemsets": len(multi_a & multi_b),
        "multi_itemset_jaccard": (
            len(multi_a & multi_b) / len(multi_union) if multi_union else 0.0
        ),
        "shared_activities": len(activities_a & activities_b),
        "activity_jaccard": (
            len(activities_a & activities_b) / len(activity_union) if activity_union else 0.0
        ),
    }


def evaluate_against_gold(result: SelectionResult, gold: dict) -> dict:
    """Did the strategy select the itemsets that we planted in a simulated log?

    Returns the recall of the planted location itemsets and of the existence-only
    itemsets, plus the share of selected itemsets that consist only of noise
    activities.
    """
    selected = set(result.itemsets)
    location = [frozenset(items) for items in gold["location_itemsets"]]
    existence = [frozenset(items) for items in gold["existence_only_itemsets"]]
    noise = set(gold["noise_activities"])

    found_location = [i for i in location if i in selected]
    found_existence = [i for i in existence if i in selected]
    only_noise = [i for i in selected if i <= noise]
    return {
        "selected_itemsets": len(selected),
        "location_recall": len(found_location) / len(location) if location else 0.0,
        "existence_recall": len(found_existence) / len(existence) if existence else 0.0,
        "missed_location_itemsets": [sorted(i) for i in location if i not in selected],
        "noise_only_share": len(only_noise) / len(selected) if selected else 0.0,
    }
