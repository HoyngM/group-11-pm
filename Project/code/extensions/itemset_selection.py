"""The two itemset selection strategies of the assignment, behind one interface.

* ``AprioriSelector`` (baseline of the 2024 paper): every single activity plus the ``top_k`` most
  frequent itemsets, using ``DataManager.frequent_activity_sets`` of ``tools.py``.
* ``ImpressedSelector`` (our variant): the activity sets of the Pareto-optimal IMPresseD patterns,
  using ``auto_stepwise_ppd`` of ``auto_impid.py``; the ordering relations are dropped.

Both return plain activity sets, so the permutation importance can use either of them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterable, Sequence

import pandas as pd

from auto_impid import auto_stepwise_ppd
from tools import DataManager


@dataclass
class SelectionResult:
    """The itemsets chosen by a strategy and a table that explains the choice.

    Attributes:
        strategy: name of the strategy that produced the result.
        itemsets: distinct activity sets, ordered by size and then alphabetically.
        details: one row per selected itemset or pattern, with its scores.
    """

    strategy: str
    itemsets: list[frozenset[str]]
    details: pd.DataFrame

    def sizes(self) -> dict[int, int]:
        """Number of itemsets per itemset size, e.g. ``{1: 18, 2: 88, 3: 48}``."""
        return pd.Series([len(itemset) for itemset in self.itemsets]).value_counts().sort_index().to_dict()


def sort_itemsets(itemsets: Iterable[frozenset[str]]) -> list[frozenset[str]]:
    """Remove duplicates and order deterministically: by size, then alphabetically."""
    return sorted(set(itemsets), key=lambda itemset: (len(itemset), sorted(itemset)))


class ItemsetSelector(ABC):
    """Chooses the activity sets whose location importance will be measured.

    Args:
        name: label of this configuration in the results and the output file names.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    # Every strategy must implement select(); ItemsetSelector itself cannot be used directly.
    # This shared shape is why main.py can treat Apriori and IMPresseD in the same way.
    @abstractmethod
    def select(self, data_manager: DataManager, case_ids: Sequence[str]) -> SelectionResult:
        """Select itemsets using only the cases in ``case_ids``."""


class AprioriSelector(ItemsetSelector):
    """Every single activity plus the ``top_k`` most frequent itemsets of two or more activities.

    Args:
        min_support: minimum share of cases that must contain an itemset.
        top_k: number of multi-activity itemsets to keep.
        max_length: largest itemset size (``None`` = no limit).
    """

    def __init__(self, name: str = "apriori", min_support: float = 0.5, top_k: int = 10,
                 max_length: int | None = None) -> None:
        super().__init__(name)  # the base class stores the name
        self.min_support = min_support
        self.top_k = top_k
        self.max_length = max_length

    def select(self, data_manager: DataManager, case_ids: Sequence[str]) -> SelectionResult:
        """All single activities and the top-k frequent itemsets of the cases in ``case_ids``."""
        _, frequent_itemsets = data_manager.frequent_activity_sets(self.min_support, self.top_k,
                                                                   self.max_length, case_ids)
        # Every single activity, frequent or not, with its support (share of cases that contain it).
        support = data_manager.activity_support(case_ids)
        singles = pd.DataFrame({"itemsets": [frozenset([a]) for a in support.index], "support": support.to_numpy()})

        # Details table: readable itemset ("a + b"), its size and its support.
        details = pd.concat([singles, frequent_itemsets], ignore_index=True)
        # pop removes the "itemsets" column and returns it, so it can be inserted again as text.
        details.insert(0, "itemset", details.pop("itemsets").apply(lambda items: " + ".join(sorted(items))))
        # Size = number of " + " separators + 1.
        details.insert(1, "size", details["itemset"].str.count(r" \+ ") + 1)
        itemsets = sort_itemsets([*singles["itemsets"], *frequent_itemsets["itemsets"]])
        return SelectionResult(self.name, itemsets, details)


class ImpressedSelector(ItemsetSelector):
    """Itemsets of the Pareto-optimal patterns found by the automatic mode of IMPresseD.

    The patterns on the fronts of all steps are used, and the ordering relations are dropped:
    ``a -> b``, ``b -> a`` and ``a => b`` all become the itemset {a, b}.

    Args:
        max_extension_step: extension steps after the single activities (2 = patterns up to length 3).
        max_gap_between_events: furthest an eventually-related event may be (in events).
    """

    def __init__(self, name: str = "impressed", max_extension_step: int = 2,
                 max_gap_between_events: int = 1000) -> None:
        super().__init__(name)
        self.max_extension_step = max_extension_step
        self.max_gap_between_events = max_gap_between_events

    def select(self, data_manager: DataManager, case_ids: Sequence[str]) -> SelectionResult:
        """Run the automatic discovery with ``case_ids`` as training cases; return the activity sets."""
        details = auto_stepwise_ppd(self.max_extension_step, self.max_gap_between_events, data_manager, case_ids)
        itemsets = sort_itemsets(frozenset(text.split(" + ")) for text in details["itemset"])
        return SelectionResult(self.name, itemsets, details)
