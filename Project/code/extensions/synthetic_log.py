"""Synthetic event log in which only the *location* of one activity determines the outcome.

Every case contains the same key activity ``B``. Only its position differs:
early in the trace (label 0) or late in the trace (label 1). Because ``B``
occurs in 100% of the cases, frequency-based methods such as Apriori cannot
see the effect, while a location-aware method can. The remaining activities
are noise in random order, so they carry no information about the outcome.

The column layout follows the BPIC11 files (``case:concept:name``, ``concept:name``,
``label``, ``event_nr``), so the output can be read without any adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# Column names of the generated log.
CASE, ACTIVITY, TIMESTAMP, ORDER = "case:concept:name", "concept:name", "time:timestamp", "event_nr"
LABEL = "label"
KEY_ACTIVITY = "B"  # the activity whose location decides the outcome


@dataclass
class GeneratorConfig:
    """Parameters of the synthetic log."""

    target_events: int | None = 100  # exact number of events (agreed with the lecturer); None = use number_of_cases
    number_of_cases: int = 15  # only used when target_events is None
    seed: int = 42
    share_late: float = 0.5  # share of cases in which B is late (label 1 before noise)
    label_noise: float = 0.1  # probability that the label is flipped, so the effect is not perfect
    other_activities: list[str] = field(default_factory=lambda: list("ACDEF"))
    min_others: int = 3  # other activities per case (excluding B, Register and Close)
    max_others: int = 4
    start_activity: str = "Register"
    end_activity: str = "Close"
    zone_size: int = 2  # B is placed within the first or last ``zone_size`` positions of the middle part


class SyntheticLogGenerator:
    """Creates a labelled event log with a planted location effect of ``B``.

    Args:
        config: parameters of the log; the defaults give 100 events.
    """

    def __init__(self, config: GeneratorConfig | None = None) -> None:
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> pd.DataFrame:
        """Return the event log, one row per event, case attributes repeated on every row."""
        rows = []
        for index, number_of_others in enumerate(self._plan_cases()):
            rows.extend(self._generate_case(index, number_of_others))
        return pd.DataFrame(rows)

    def _plan_cases(self) -> list[int]:
        """Number of other activities per case; with ``target_events`` the cases add up to exactly that many events."""
        config = self.config
        sizes = list(range(config.min_others, config.max_others + 1))
        if config.target_events is None:
            return [int(self.rng.choice(sizes)) for _ in range(config.number_of_cases)]

        # A case has Register, Close, B and k other activities. Find all ways to reach the target exactly.
        overhead = 3
        plans = [counts for counts in self._compositions(config.target_events, [k + overhead for k in sizes])]
        if not plans:
            raise ValueError(f"{config.target_events} events cannot be built from traces of length "
                             f"{[k + overhead for k in sizes]}")
        # Pick one of the possible plans at random and shuffle the order of the cases.
        counts = plans[int(self.rng.integers(len(plans)))]
        others = [k for k, count in zip(sizes, counts) for _ in range(count)]
        return [int(k) for k in self.rng.permutation(others)]

    @staticmethod
    def _compositions(total: int, lengths: list[int], prefix: tuple[int, ...] = ()):
        """All tuples of non-negative counts c with sum(c_i * lengths_i) == total (recursive generator)."""
        if len(prefix) == len(lengths) - 1:
            # Last length: the remaining events must be an exact multiple of it.
            remainder = total - sum(c * n for c, n in zip(prefix, lengths))
            if remainder >= 0 and remainder % lengths[-1] == 0:
                yield (*prefix, remainder // lengths[-1])
            return
        used = sum(c * n for c, n in zip(prefix, lengths))
        for count in range((total - used) // lengths[len(prefix)] + 1):
            yield from SyntheticLogGenerator._compositions(total, lengths, (*prefix, count))

    def _generate_case(self, index: int, number_of_others: int) -> list[dict]:
        """The event rows of one case: B early (label 0) or late (label 1), with label noise."""
        config = self.config
        late = self.rng.random() < config.share_late
        trace = self._build_trace(late, number_of_others)

        label = int(late)
        if self.rng.random() < config.label_noise:
            label = 1 - label

        # Case attributes are independent of the outcome (confounder-free baseline).
        case_attributes = {
            "Age": int(self.rng.integers(40, 86)),
            "Category": str(self.rng.choice(["X", "Y", "Z"])),
            LABEL: label,
        }
        # One case per day, one event per hour.
        start = pd.Timestamp("2026-01-01") + pd.Timedelta(days=index)
        return [
            {
                CASE: f"Case_{index + 1:04d}",
                ACTIVITY: activity,
                TIMESTAMP: start + pd.Timedelta(hours=position),
                ORDER: position,
                **case_attributes,
            }
            for position, activity in enumerate(trace, start=1)
        ]

    def _build_trace(self, late: bool, number_of_others: int) -> list[str]:
        """Random noise activities with ``B`` inserted in the early or the late zone."""
        config = self.config
        middle = list(self.rng.choice(config.other_activities, size=number_of_others, replace=False))
        offset = int(self.rng.integers(0, config.zone_size))
        insert_at = number_of_others - offset if late else offset
        middle.insert(insert_at, KEY_ACTIVITY)
        return [config.start_activity, *middle, config.end_activity]
