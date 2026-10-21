"""Simulated event logs in which the position of activities determines the outcome.

Because we create the log ourselves, we know which activity sets matter for
which reason. This gold standard lets us test whether a selection strategy
puts the right itemsets forward for the location permutation importance.

Planted mechanisms (additive on the log-odds of a positive outcome):

* ``X`` (location effect): always at an early or at a late position, with equal
  probability. Early gives a higher chance of a positive outcome, late a
  lower one. Its mere presence says nothing about the outcome.
* ``P`` followed by ``Q`` (location effect of an activity pair): same idea for
  the block ``P Q``. The block occurs in a configurable share of the cases.
* ``Y`` (existence effect): presence raises the chance of a positive outcome,
  its position is random.
* ``N1`` .. ``Nk``: noise activities without any effect.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

START, END = "Register", "Close"


@dataclass
class SimulationConfig:
    """Parameters of a simulated log."""

    number_of_cases: int = 1000
    seed: int = 42
    min_length: int = 14
    max_length: int = 22
    number_of_noise_activities: int = 10
    share_with_x: float = 0.8
    share_with_pair: float = 0.3
    share_with_y: float = 0.6
    effect_strength: float = 2.0
    existence_strength: float = 2.0
    early_zone: tuple[int, int] = (1, 4)  # positions (inclusive) counted from the start
    late_zone_from_end: tuple[int, int] = (2, 5)  # positions counted back from the end
    extra: dict = field(default_factory=dict)


class SimulatedLogGenerator:
    """Creates a labelled event log with planted location and existence effects."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.noise_activities = [
            f"N{i}" for i in range(1, config.number_of_noise_activities + 1)
        ]

    def generate(self) -> pd.DataFrame:
        """Return the event log as a table with one row per event."""
        rng = np.random.default_rng(self.config.seed)
        rows = []
        for case in range(self.config.number_of_cases):
            trace, log_odds = self._simulate_case(rng)
            label = int(rng.random() < 1.0 / (1.0 + np.exp(-log_odds)))
            age = int(rng.integers(20, 81))
            category = str(rng.choice(["A", "B", "C"]))
            for event_nr, activity in enumerate(trace, start=1):
                rows.append(
                    {
                        "case:concept:name": case,
                        "concept:name": activity,
                        "event_nr": event_nr,
                        "label": label,
                        "Age": age,
                        "Category": category,
                    }
                )
        return pd.DataFrame(rows)

    def gold_standard(self) -> dict:
        """Which itemsets matter, and why."""
        return {
            "location_itemsets": [["X"], ["P", "Q"]],
            "existence_only_itemsets": [["Y"]],
            "planted_activities": ["X", "P", "Q", "Y"],
            "noise_activities": [START, END] + self.noise_activities,
        }

    def _simulate_case(self, rng: np.random.Generator) -> tuple[list[str], float]:
        config = self.config
        length = int(rng.integers(config.min_length, config.max_length + 1))
        slots: list[str | None] = [None] * length
        slots[0], slots[-1] = START, END
        log_odds = 0.0

        # Location effects: choose early or late zone with equal probability.
        for items, share in ((["X"], config.share_with_x), (["P", "Q"], config.share_with_pair)):
            if rng.random() >= share:
                continue
            early = bool(rng.random() < 0.5)
            start = self._free_start(rng, slots, length, early, len(items))
            for offset, item in enumerate(items):
                slots[start + offset] = item
            log_odds += config.effect_strength if early else -config.effect_strength

        # Existence effect: present or not, random position.
        if rng.random() < config.share_with_y:
            free = [i for i, s in enumerate(slots) if s is None]
            slots[int(rng.choice(free))] = "Y"
            log_odds += config.existence_strength / 2.0
        else:
            log_odds -= config.existence_strength / 2.0

        # Everything that is still empty becomes a noise activity.
        for position, slot in enumerate(slots):
            if slot is None:
                slots[position] = str(rng.choice(self.noise_activities))
        return slots, log_odds

    def _free_start(self, rng, slots, length, early: bool, width: int) -> int:
        """Pick a start position for a block of ``width`` events in the early or late zone."""
        low, high = self.config.early_zone
        if not early:
            near, far = self.config.late_zone_from_end
            low, high = length - far, length - near
        options = [
            start
            for start in range(low, high + 1)
            if start + width <= length - 1
            and all(slots[start + k] is None for k in range(width))
        ]
        return int(rng.choice(options))
