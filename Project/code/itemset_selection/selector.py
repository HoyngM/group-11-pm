"""Common interface of the itemset selection strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from .event_log import EventLog


@dataclass
class SelectionResult:
    """The itemsets chosen by a strategy plus a table describing how.

    Attributes:
        strategy: name of the strategy that produced the result.
        itemsets: distinct activity sets (by activity name), single activities
            first, then by size and name.
        details: one row per selected pattern or itemset with its scores.
    """

    strategy: str
    itemsets: list[frozenset[str]]
    details: pd.DataFrame

    def sizes(self) -> dict[int, int]:
        """Number of itemsets per itemset size."""
        counts: dict[int, int] = {}
        for itemset in self.itemsets:
            counts[len(itemset)] = counts.get(len(itemset), 0) + 1
        return dict(sorted(counts.items()))


def sort_itemsets(itemsets) -> list[frozenset[str]]:
    """Deterministic order: by size first, then alphabetically."""
    return sorted(set(itemsets), key=lambda itemset: (len(itemset), sorted(itemset)))


class ItemsetSelector(ABC):
    """Chooses the activity sets whose location importance will be measured.

    Both strategies return plain activity sets, so the permutation-importance
    pipeline can consume either of them without changes.
    """

    name: str

    @abstractmethod
    def select(self, log: EventLog) -> SelectionResult:
        """Select itemsets from a (training) event log."""
