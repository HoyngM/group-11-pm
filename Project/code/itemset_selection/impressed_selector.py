"""Variant strategy: outcome-oriented pattern extraction of IMPresseD.

Vazifehdoostirani et al. (2023) discover patterns step by step: start with
single activities, keep the patterns on the Pareto front of the interest
functions, extend those patterns, and repeat. Here the discovery runs in the
fully automated setting and the ordering relations of the discovered patterns
are dropped, so that the result is a list of activity sets.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from .case_distance import CaseDistance
from .event_log import EventLog
from .interest import InterestCalculator
from .pareto import non_dominated
from .pattern import ExtensionRule, Pattern, extend_instance
from .selector import ItemsetSelector, SelectionResult, sort_itemsets

# Frequency and outcome are maximised, case distance is minimised.
MAXIMISE = (True, True, False)


class ImpressedItemsetSelector(ItemsetSelector):
    """Itemsets from the Pareto-optimal outcome-oriented patterns of IMPresseD.

    Args:
        max_pattern_length: longest pattern that is discovered (1, 2 or 3 in
            the assignment). Every extension step adds one activity.
        numeric_attributes: case attributes used (numerically) for case distance.
        categorical_attributes: case attributes used (categorically) for case distance.
        extend_eventual_patterns: whether patterns that contain an eventual
            relation may be extended again. The reference implementation of the
            authors does not extend them.
        rules: extension rules to apply; all four rules by default.
        name: label of this configuration in the results.
    """

    def __init__(
        self,
        max_pattern_length: int = 3,
        numeric_attributes: Sequence[str] = (),
        categorical_attributes: Sequence[str] = (),
        extend_eventual_patterns: bool = False,
        rules: Sequence[ExtensionRule] = tuple(ExtensionRule),
        name: str = "impressed",
    ) -> None:
        self.name = name
        self.max_pattern_length = max_pattern_length
        self.distance = CaseDistance(numeric_attributes, categorical_attributes)
        self.extend_eventual_patterns = extend_eventual_patterns
        self.rules = tuple(rules)

    def select(self, log: EventLog) -> SelectionResult:
        calculator = InterestCalculator(
            log.labels, self.distance.pairwise(log.case_attributes)
        )
        names = log.activity_names

        # Step 0: all single activities form the first generation of patterns.
        counts_per_activity = log.activity_counts_per_case()
        candidates = {
            Pattern((activity,)): counts_per_activity[:, activity]
            for activity in range(len(names))
            if counts_per_activity[:, activity].any()
        }

        selected_rows = []
        for length in range(1, self.max_pattern_length + 1):
            front = self._pareto_front(candidates, calculator, length, names)
            selected_rows.extend(front)
            if length == self.max_pattern_length:
                break
            parents = [
                row["pattern_object"]
                for row in front
                if self.extend_eventual_patterns
                or not row["pattern_object"].has_eventual_relation
            ]
            candidates = self._extend(parents, log)
            if not candidates:
                break

        details = pd.DataFrame(
            [{k: v for k, v in row.items() if k != "pattern_object"} for row in selected_rows]
        )
        itemsets = sort_itemsets(
            frozenset(names[a] for a in row["pattern_object"].itemset)
            for row in selected_rows
        )
        return SelectionResult(self.name, itemsets, details)

    def _pareto_front(self, candidates, calculator, length, names) -> list[dict]:
        """Score all candidates and return the rows of the non-dominated ones."""
        patterns = list(candidates)
        interests = [calculator.evaluate(candidates[pattern]) for pattern in patterns]
        objectives = np.array([[i.frequency, i.outcome, i.distance] for i in interests])
        mask = non_dominated(objectives, MAXIMISE)
        return [
            {
                "pattern": pattern.describe(names),
                "length": length,
                "itemset": " + ".join(sorted(names[a] for a in pattern.itemset)),
                "case_coverage": interest.frequency,
                "outcome_interest": interest.outcome,
                "case_distance": interest.distance,
                "pattern_object": pattern,
            }
            for pattern, interest, keep in zip(patterns, interests, mask)
            if keep
        ]

    def _extend(self, parents: Sequence[Pattern], log: EventLog) -> dict[Pattern, np.ndarray]:
        """Extend the parent patterns and count the instances of every new pattern per case."""
        instances: dict[Pattern, dict[int, set]] = defaultdict(lambda: defaultdict(set))
        for parent in parents:
            for case_index, trace in enumerate(log.traces):
                for instance in parent.find_instances(trace):
                    for new_pattern, new_instance in extend_instance(
                        parent, instance, trace, self.rules
                    ):
                        instances[new_pattern][case_index].add(new_instance)

        counts = {}
        for pattern, per_case in instances.items():
            vector = np.zeros(len(log), dtype=int)
            for case_index, found in per_case.items():
                vector[case_index] = len(found)
            counts[pattern] = vector
        return counts
