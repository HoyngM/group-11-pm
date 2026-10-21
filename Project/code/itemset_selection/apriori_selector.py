"""Baseline strategy of Vazifehdoostirani et al. (2024): frequent itemsets via Apriori."""

from __future__ import annotations

from itertools import combinations

import pandas as pd

from .event_log import EventLog
from .selector import ItemsetSelector, SelectionResult, sort_itemsets


class AprioriItemsetSelector(ItemsetSelector):
    """Frequent itemsets of activities, found with the Apriori algorithm.

    As in the original paper, every single activity is analysed and, on top of
    that, the ``top_k`` most frequent itemsets with more than one activity.

    Args:
        min_support: minimum fraction of cases that must contain the itemset.
        top_k: number of multi-activity itemsets to keep.
        max_length: largest itemset size that is considered.
        singletons: ``"all"`` keeps every single activity, ``"frequent"`` only
            those reaching ``min_support``, ``"none"`` keeps none.
    """

    def __init__(
        self,
        min_support: float = 0.5,
        top_k: int = 10,
        max_length: int = 3,
        singletons: str = "all",
        name: str = "apriori",
    ) -> None:
        if singletons not in {"all", "frequent", "none"}:
            raise ValueError("singletons must be 'all', 'frequent' or 'none'")
        self.name = name
        self.min_support = min_support
        self.top_k = top_k
        self.max_length = max_length
        self.singletons = singletons

    def select(self, log: EventLog) -> SelectionResult:
        transactions = [frozenset(trace) for trace in log.traces]
        support = self._frequent_itemsets(transactions)
        names = log.activity_names

        chosen = []
        if self.singletons == "all":
            single_support = self._single_supports(transactions, len(names))
            chosen += [(frozenset([a]), s) for a, s in enumerate(single_support)]
        elif self.singletons == "frequent":
            chosen += [(i, s) for i, s in support.items() if len(i) == 1]

        multi = [(i, s) for i, s in support.items() if len(i) > 1]
        multi.sort(key=lambda pair: (-pair[1], sorted(pair[0])))
        chosen += multi[: self.top_k]

        rows = [
            {
                "itemset": " + ".join(sorted(names[a] for a in itemset)),
                "size": len(itemset),
                "support": value,
            }
            for itemset, value in chosen
        ]
        itemsets = sort_itemsets(
            frozenset(names[a] for a in itemset) for itemset, _ in chosen
        )
        return SelectionResult(self.name, itemsets, pd.DataFrame(rows))

    @staticmethod
    def _single_supports(transactions, number_of_activities: int) -> list[float]:
        counts = [0] * number_of_activities
        for transaction in transactions:
            for activity in transaction:
                counts[activity] += 1
        return [count / len(transactions) for count in counts]

    def _frequent_itemsets(self, transactions) -> dict[frozenset[int], float]:
        """Level-wise Apriori search, up to ``max_length`` activities."""
        total = len(transactions)

        def support_of(candidates):
            counts = {candidate: 0 for candidate in candidates}
            for transaction in transactions:
                for candidate in candidates:
                    if candidate <= transaction:
                        counts[candidate] += 1
            return {c: n / total for c, n in counts.items() if n / total >= self.min_support}

        singles = {frozenset([a]) for transaction in transactions for a in transaction}
        frequent = support_of(singles)
        result = dict(frequent)
        size = 1
        while frequent and size < self.max_length:
            size += 1
            previous = list(frequent)
            candidates = set()
            for left, right in combinations(previous, 2):
                union = left | right
                if len(union) != size:
                    continue
                # Apriori property: all subsets must be frequent.
                if all(frozenset(sub) in frequent for sub in combinations(union, size - 1)):
                    candidates.add(union)
            frequent = support_of(candidates)
            result.update(frequent)
        return result
