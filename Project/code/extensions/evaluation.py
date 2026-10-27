"""Compare selection strategies with each other and with the gold standard of a simulated log."""

from __future__ import annotations

from extensions.itemset_selection import SelectionResult


def jaccard(first: set, second: set) -> float:
    """Shared elements divided by all elements (0 if both sets are empty).

    1 = the two sets are identical, 0 = they have nothing in common.
    """
    union = first | second  # every element that is in at least one of the sets
    return len(first & second) / len(union) if union else 0.0  # & = elements in both sets


def compare_strategies(reference: SelectionResult, other: SelectionResult) -> dict:
    """Overlap (Jaccard) of the itemsets and activities selected by two strategies."""
    itemsets_a, itemsets_b = set(reference.itemsets), set(other.itemsets)
    # Single activities dominate the totals, so the larger itemsets are also compared on their own.
    multi_a = {itemset for itemset in itemsets_a if len(itemset) > 1}
    multi_b = {itemset for itemset in itemsets_b if len(itemset) > 1}
    return {
        "itemset_jaccard": jaccard(itemsets_a, itemsets_b),
        "multi_itemset_jaccard": jaccard(multi_a, multi_b),
        # set().union(*itemsets) merges all itemsets into one set of activities.
        "activity_jaccard": jaccard(set().union(*itemsets_a), set().union(*itemsets_b)),
    }


def evaluate_against_gold(result: SelectionResult, gold: dict) -> dict:
    """Recall of the planted itemsets and the share of selected itemsets that are pure noise.

    Recall = share of the planted itemsets that the strategy selected (1 = it found all of them).
    """
    selected = set(result.itemsets)
    location = [frozenset(items) for items in gold["location_itemsets"]]
    existence = [frozenset(items) for items in gold["existence_only_itemsets"]]
    noise = set(gold["noise_activities"])
    return {
        # True counts as 1 in sum(), so this counts how many planted itemsets were selected.
        "location_recall": sum(i in selected for i in location) / len(location),
        "existence_recall": sum(i in selected for i in existence) / len(existence),
        # An itemset is pure noise when all of its activities are noise activities (<= means "subset of").
        "noise_only_share": sum(i <= noise for i in selected) / len(selected) if selected else 0.0,
    }
