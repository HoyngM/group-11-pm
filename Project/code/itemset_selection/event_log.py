"""Event log container used by all itemset selection strategies."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class EventLog:
    """A labelled event log, reduced to what the selection strategies need.

    Only the control-flow perspective (activity names in execution order), the
    binary case outcome and a handful of case-level attributes are kept.
    Activities are stored as integer ids; ``activity_names[i]`` maps id ``i``
    back to its name.
    """

    def __init__(
        self,
        case_ids: Sequence,
        activity_names: Sequence[str],
        traces: Sequence[Sequence[int]],
        labels: Sequence[int],
        case_attributes: pd.DataFrame,
    ) -> None:
        self.case_ids = list(case_ids)
        self.activity_names = list(activity_names)
        self.traces = [list(trace) for trace in traces]
        self.labels = np.asarray(labels, dtype=int)
        self.case_attributes = case_attributes.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.traces)

    @property
    def number_of_events(self) -> int:
        return sum(len(trace) for trace in self.traces)

    @classmethod
    def from_csv(
        cls,
        path: str,
        case_column: str = "case:concept:name",
        activity_column: str = "concept:name",
        label_column: str = "label",
        order_column: str | None = "event_nr",
        numeric_attributes: Sequence[str] = (),
        categorical_attributes: Sequence[str] = (),
    ) -> "EventLog":
        """Read a labelled event log from a CSV file (one row per event).

        Events are ordered per case by ``order_column``; if that column is
        missing the file order is kept. Case attributes and the label are taken
        from the first event of each case.
        """
        frame = pd.read_csv(path)
        if order_column and order_column in frame.columns:
            frame = frame.sort_values([case_column, order_column], kind="stable")

        activity_names = sorted(frame[activity_column].astype(str).unique())
        activity_ids = {name: index for index, name in enumerate(activity_names)}

        grouped = frame.groupby(case_column, sort=False)
        case_ids = list(grouped.groups.keys())
        traces = [
            [activity_ids[name] for name in group[activity_column].astype(str)]
            for _, group in grouped
        ]
        first_events = grouped.first()
        labels = first_events[label_column].astype(int).to_numpy()

        attribute_columns = list(numeric_attributes) + list(categorical_attributes)
        case_attributes = first_events[attribute_columns].copy()
        for column in categorical_attributes:
            case_attributes[column] = case_attributes[column].astype(str)
        return cls(case_ids, activity_names, traces, labels, case_attributes)

    def subset(self, indices: Sequence[int]) -> "EventLog":
        """Return a new log with only the cases at ``indices``."""
        indices = list(indices)
        return EventLog(
            [self.case_ids[i] for i in indices],
            self.activity_names,
            [self.traces[i] for i in indices],
            self.labels[indices],
            self.case_attributes.iloc[indices],
        )

    def stratified_split(self, test_size: float, seed: int) -> tuple["EventLog", "EventLog"]:
        """Split into train and test logs, keeping the label ratio."""
        train_idx, test_idx = train_test_split(
            np.arange(len(self)), test_size=test_size, random_state=seed, stratify=self.labels
        )
        return self.subset(sorted(train_idx)), self.subset(sorted(test_idx))

    def activity_counts_per_case(self) -> np.ndarray:
        """Matrix (cases x activities) with how often each activity occurs per case."""
        counts = np.zeros((len(self), len(self.activity_names)), dtype=int)
        for row, trace in enumerate(self.traces):
            counts[row] = np.bincount(trace, minlength=len(self.activity_names))
        return counts
