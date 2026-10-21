"""Location permutation, ported from ``tools.py`` of the reference implementation.

Two procedures exist in the reference code:

* ``shuffle_activity`` (``trace_permutation_importance``): moves every occurrence of one
  activity to a random observed location.
* ``shuffle_itemset`` (``shuffle_sequence`` + ``find_itemset_indexes``): moves the first
  non-overlapping occurrences of a multi-activity itemset to random observed locations
  while keeping their relative order.

Locations are 1-based, like ``event_nr``.
"""

from __future__ import annotations

import itertools

import numpy as np


class ReferencePermuter:
    """Permutes the location of activities within one trace."""

    def __init__(self, allowed_locations: dict[str, list[int]], seed: int = 2023,
                 exact_tracking: bool = False) -> None:
        self.allowed_locations = allowed_locations
        self.rng = np.random.default_rng(seed)
        # The reference code keeps track of the moving events with a dictionary that is off by one
        # when an event is moved backwards; later events of the itemset can then be the wrong event
        # and the order within the itemset is not always kept. True tracks every event exactly.
        self.exact_tracking = exact_tracking

    def shuffle_activity(self, trace: list[str], activity: str, constrain: bool = True) -> list[str]:
        """Move each occurrence of ``activity`` to a random (observed, if ``constrain``) location."""
        trace = list(trace)
        for index in [i for i, a in enumerate(trace) if a == activity]:
            if constrain:
                allowed = [i for i in self.allowed_locations[activity] if i < len(trace) + 1]
                random_index = int(self.rng.choice(allowed)) - 1
            else:
                random_index = int(self.rng.integers(len(trace)))
            trace.insert(random_index, trace.pop(index))
        return trace

    def shuffle_itemset(self, sequence: list[str], itemset: frozenset[str]) -> list[str]:
        """Move the occurrences of ``itemset`` to random observed locations, keeping their order."""
        if self.exact_tracking:
            return self._shuffle_itemset_exact(sequence, itemset)
        shuffled = list(sequence)
        locations = {index: index for index in range(len(sequence))}
        for occurrence in self.find_itemset_indexes(sequence, itemset):
            subset = [sequence[i] for i in occurrence]
            max_possible = self._latest_locations(subset)
            if max_possible is None:  # no observed location keeps the order; leave this occurrence
                continue
            previous_index = -1
            for activity in subset:
                allowed = [i for i in self.allowed_locations[activity]
                           if previous_index < i <= max_possible[activity]]
                random_index = int(self.rng.choice(allowed))
                previous_index = random_index
                current = locations[occurrence[subset.index(activity)]]
                shuffled.insert(random_index - 1, shuffled.pop(current))
                locations[current] = min(random_index - 1, len(sequence) - 1)
                for location in locations:
                    if current < location < random_index - 1:
                        locations[location] -= 1
        return shuffled

    def _shuffle_itemset_exact(self, sequence: list[str], itemset: frozenset[str]) -> list[str]:
        """Same draws as the reference, but every moved event is found by identity."""
        events = list(enumerate(sequence))  # (original index, activity)
        for occurrence in self.find_itemset_indexes(sequence, itemset):
            subset = [sequence[i] for i in occurrence]
            max_possible = self._latest_locations(subset)
            if max_possible is None:
                continue
            previous_index = -1
            for original_index, activity in zip(occurrence, subset):
                allowed = [i for i in self.allowed_locations[activity]
                           if previous_index < i <= max_possible[activity]]
                random_index = int(self.rng.choice(allowed))
                previous_index = random_index
                current = next(p for p, (index, _) in enumerate(events) if index == original_index)
                events.insert(random_index - 1, events.pop(current))
        return [activity for _, activity in events]

    def _latest_locations(self, subset: list[str]) -> dict[str, int] | None:
        """Latest observed location of every activity such that the order of ``subset`` can be kept."""
        latest, bound = {}, 10000
        for activity in reversed(subset):
            candidates = [a for a in self.allowed_locations[activity] if a < bound]
            if not candidates:
                return None
            latest[activity] = bound = max(candidates)
        return latest

    @staticmethod
    def find_itemset_indexes(sequence: list[str], itemset: frozenset[str]) -> list[tuple[int, ...]]:
        """Non-overlapping index combinations of ``sequence`` that contain exactly the itemset."""
        found = [combination for combination in itertools.combinations(range(len(sequence)), len(itemset))
                 if {sequence[i] for i in combination} == set(itemset)]
        kept: list[tuple[int, ...]] = []
        for candidate in found:
            if all(not set(candidate) & set(other) for other in kept):
                kept.append(candidate)
        return kept
