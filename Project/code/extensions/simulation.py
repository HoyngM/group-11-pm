"""Simulated event logs in which the position of activities determines the outcome.

Because we create the log ourselves, we know which activity sets matter and why. This gold
standard shows whether a selection strategy puts the right itemsets forward.

Planted mechanisms (added to the log-odds of a positive outcome):

* ``x`` (location effect): at an early or a late position with equal probability. Early
  raises the chance of a positive outcome, late lowers it; its mere presence says nothing.
* ``p`` followed by ``q`` (location effect of a pair): the same for the block ``p q``,
  which occurs in a configurable share of the cases.
* ``y`` (existence effect): its presence raises the chance of a positive outcome; its
  position is random.
* ``n1`` .. ``nk``: noise activities without any effect.

Activity names are lower case, as they are after the cleaning in ``DataManager._load_df``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

START, END = "register", "close"  # first and last activity of every simulated case


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
    effect_strength: float = 2.0  # change of the log-odds by an early (+) or late (-) location
    existence_strength: float = 2.0  # difference in log-odds between cases with and without y
    early_zone: tuple[int, int] = (1, 4)  # positions (inclusive) counted from the start
    late_zone_from_end: tuple[int, int] = (2, 5)  # positions counted back from the end


class SimulatedLogGenerator:
    """Creates a labelled event log with planted location and existence effects.

    Args:
        config: parameters of the simulated log.
    """

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.noise_activities = [f"n{i}" for i in range(1, config.number_of_noise_activities + 1)]

    def generate(self) -> pd.DataFrame:
        """The event log as a table with one row per event (case attributes repeated on every row)."""
        rng = np.random.default_rng(self.config.seed)
        rows = []
        for case in range(self.config.number_of_cases):
            trace, log_odds = self._simulate_case(rng)
            # The label is drawn with the logistic probability of the summed effects.
            label = int(rng.random() < 1.0 / (1.0 + np.exp(-log_odds)))
            # Case attributes without any effect, for the IMPresseD case distance.
            age, category = int(rng.integers(20, 81)), str(rng.choice(["A", "B", "C"]))
            rows += [{"case:concept:name": case, "concept:name": activity, "event_nr": event_nr,
                      "label": label, "Age": age, "Category": category}
                     for event_nr, activity in enumerate(trace, start=1)]
        return pd.DataFrame(rows)

    def gold_standard(self) -> dict:
        """Which itemsets matter, and why."""
        return {
            "location_itemsets": [["x"], ["p", "q"]],
            "existence_only_itemsets": [["y"]],
            "noise_activities": [START, END] + self.noise_activities,
        }

    def _simulate_case(self, rng: np.random.Generator) -> tuple[list[str], float]:
        """One trace and the log-odds of a positive outcome that its planted activities imply."""
        config = self.config
        length = int(rng.integers(config.min_length, config.max_length + 1))
        slots: list[str | None] = [None] * length
        slots[0], slots[-1] = START, END
        log_odds = 0.0

        # Location effects: place x and the block p q early or late, with equal probability.
        for items, share in ((["x"], config.share_with_x), (["p", "q"], config.share_with_pair)):
            if rng.random() >= share:
                continue
            early = bool(rng.random() < 0.5)
            start = self._free_start(rng, slots, length, early, len(items))
            for offset, item in enumerate(items):
                slots[start + offset] = item
            log_odds += config.effect_strength if early else -config.effect_strength

        # Existence effect: y present (anywhere) or absent.
        if rng.random() < config.share_with_y:
            free = [i for i, slot in enumerate(slots) if slot is None]
            slots[int(rng.choice(free))] = "y"
            log_odds += config.existence_strength / 2.0
        else:
            log_odds -= config.existence_strength / 2.0

        # Every slot that is still empty becomes a noise activity.
        for position, slot in enumerate(slots):
            if slot is None:
                slots[position] = str(rng.choice(self.noise_activities))
        return slots, log_odds

    def _free_start(self, rng, slots, length, early: bool, width: int) -> int:
        """Random start position of a block of ``width`` free events in the early or late zone."""
        low, high = self.config.early_zone
        if not early:
            # The late zone is defined from the end of the trace; convert it to positions.
            near, far = self.config.late_zone_from_end
            low, high = length - far, length - near
        options = [start for start in range(low, high + 1)
                   if start + width <= length - 1 and all(slots[start + k] is None for k in range(width))]
        return int(rng.choice(options))
