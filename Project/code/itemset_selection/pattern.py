"""Process patterns on totally ordered traces and their extension rules.

A pattern is a chain of activities. The relation between two neighbouring
activities is either *direct* (the second event comes immediately after the
first one) or *eventual* (it comes later, with at least one event in between).
This is the special case of the IMPresseD patterns for traces without
concurrency.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterator, Sequence

DIRECT = 0
EVENTUAL = 1


class ExtensionRule(Enum):
    """The four extension rules that apply to totally ordered traces."""

    DIRECTLY_FOLLOWING = "directly following"
    DIRECTLY_PRECEDING = "directly preceding"
    EVENTUALLY_FOLLOWING = "eventually following"
    EVENTUALLY_PRECEDING = "eventually preceding"


@dataclass(frozen=True)
class Pattern:
    """A chain of activity ids; ``relations[i]`` links activity ``i`` and ``i + 1``."""

    activities: tuple[int, ...]
    relations: tuple[int, ...] = ()

    @property
    def length(self) -> int:
        return len(self.activities)

    @property
    def itemset(self) -> frozenset[int]:
        """The activity set of the pattern; ordering relations are ignored."""
        return frozenset(self.activities)

    @property
    def has_eventual_relation(self) -> bool:
        return EVENTUAL in self.relations

    def describe(self, activity_names: Sequence[str]) -> str:
        """Readable form such as ``A -> B => C`` (``->`` direct, ``=>`` eventual)."""
        parts = [activity_names[self.activities[0]]]
        for relation, activity in zip(self.relations, self.activities[1:]):
            parts.append("->" if relation == DIRECT else "=>")
            parts.append(activity_names[activity])
        return " ".join(parts)

    def find_instances(self, trace: Sequence[int]) -> list[tuple[int, ...]]:
        """All positions tuples at which the pattern occurs in ``trace``."""
        first = self.activities[0]
        instances = [(position,) for position, act in enumerate(trace) if act == first]
        for activity, relation in zip(self.activities[1:], self.relations):
            extended = []
            for instance in instances:
                last = instance[-1]
                if relation == DIRECT:
                    if last + 1 < len(trace) and trace[last + 1] == activity:
                        extended.append(instance + (last + 1,))
                else:
                    extended.extend(
                        instance + (q,)
                        for q in range(last + 2, len(trace))
                        if trace[q] == activity
                    )
            instances = extended
        return instances


def extend_instance(
    pattern: Pattern,
    instance: tuple[int, ...],
    trace: Sequence[int],
    rules: Sequence[ExtensionRule],
) -> Iterator[tuple[Pattern, tuple[int, ...]]]:
    """Yield every (new pattern, new instance) obtained from one pattern instance."""
    first, last = instance[0], instance[-1]
    for rule in rules:
        if rule is ExtensionRule.DIRECTLY_FOLLOWING:
            positions, relation, at_end = ([last + 1], DIRECT, True)
        elif rule is ExtensionRule.EVENTUALLY_FOLLOWING:
            positions, relation, at_end = (range(last + 2, len(trace)), EVENTUAL, True)
        elif rule is ExtensionRule.DIRECTLY_PRECEDING:
            positions, relation, at_end = ([first - 1], DIRECT, False)
        else:
            positions, relation, at_end = (range(0, first - 1), EVENTUAL, False)

        for position in positions:
            if not 0 <= position < len(trace):
                continue
            activity = trace[position]
            if at_end:
                new_pattern = Pattern(
                    pattern.activities + (activity,), pattern.relations + (relation,)
                )
                new_instance = instance + (position,)
            else:
                new_pattern = Pattern(
                    (activity,) + pattern.activities, (relation,) + pattern.relations
                )
                new_instance = (position,) + instance
            yield new_pattern, new_instance
