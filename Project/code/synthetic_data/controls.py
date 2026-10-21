"""Shuffled negative controls for an event log.

Each control destroys a different part of the structure that a pattern or
location-aware method looks for. A correct implementation should therefore
find (close to) nothing in the controls:

* ``within_case``:   activities are permuted inside each case, so order and
  location are gone but the activity set of every case is kept.
* ``outcomes``:      the case outcomes are permuted over the cases, so the
  link between traces and outcome is gone but the traces are untouched.
* ``across_traces``: the activity column is permuted over *all* events, so
  both the patterns and the activity sets of the cases are gone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from synthetic_data.generator import ACTIVITY, CASE, LABEL


class ShuffleControls:
    """Creates shuffled copies of an event log; the input is never modified."""

    def __init__(self, seed: int = 42, case_column: str = CASE,
                 activity_column: str = ACTIVITY, label_column: str = LABEL) -> None:
        self.rng = np.random.default_rng(seed)
        self.case_column = case_column
        self.activity_column = activity_column
        self.label_column = label_column

    def within_case(self, log: pd.DataFrame) -> pd.DataFrame:
        """Permute the activities inside every case; timestamps and event numbers stay in place."""
        shuffled = log.copy()
        shuffled[self.activity_column] = shuffled.groupby(self.case_column)[self.activity_column].transform(
            lambda activities: self.rng.permutation(activities.to_numpy())
        )
        return shuffled

    def outcomes(self, log: pd.DataFrame) -> pd.DataFrame:
        """Permute the case-level outcomes over the cases."""
        shuffled = log.copy()
        outcome_per_case = shuffled.groupby(self.case_column)[self.label_column].first()
        permuted = dict(zip(outcome_per_case.index, self.rng.permutation(outcome_per_case.to_numpy())))
        shuffled[self.label_column] = shuffled[self.case_column].map(permuted)
        return shuffled

    def across_traces(self, log: pd.DataFrame) -> pd.DataFrame:
        """Permute the activity column over all events of the log."""
        shuffled = log.copy()
        shuffled[self.activity_column] = self.rng.permutation(shuffled[self.activity_column].to_numpy())
        return shuffled

    def all_controls(self, log: pd.DataFrame) -> dict[str, pd.DataFrame]:
        """Return every control, keyed by name."""
        return {
            "shuffled_within_case": self.within_case(log),
            "shuffled_outcomes": self.outcomes(log),
            "shuffled_across_traces": self.across_traces(log),
        }
