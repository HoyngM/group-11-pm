"""Event log preparation and index-based encoding, as in ``DataManager`` of the authors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from itemset_selection import EventLog


class PreparedLog:
    """A cleaned event log with traces, labels, observed locations and the encoding.

    Cleaning follows ``DataManager._load_df`` of the reference implementation:
    activity names are lower-cased and stripped of spaces, ``-`` and ``_``, and
    cases that contain an activity with fewer than ``frequency_threshold`` events
    in the whole log are removed. Locations are the (1-based) ``event_nr`` values.
    """

    PADDING = "0"  # the reference one-hot encodes the padding of short traces as activity "0"

    def __init__(self, frame: pd.DataFrame, case_column: str = "case:concept:name",
                 activity_column: str = "concept:name", label_column: str = "label",
                 order_column: str = "event_nr", frequency_threshold: int = 2) -> None:
        self.case_column, self.activity_column = case_column, activity_column
        self.label_column, self.order_column = label_column, order_column

        frame = frame.copy()
        frame[case_column] = frame[case_column].astype(str)
        frame[activity_column] = frame[activity_column].astype(str).map(self.clean_activity_name)
        frame = frame.sort_values([case_column, order_column], kind="stable")

        events_per_activity = frame.groupby(activity_column)[case_column].count()
        rare = events_per_activity[events_per_activity < frequency_threshold].index
        rare_cases = frame.loc[frame[activity_column].isin(rare), case_column].unique()
        self.frame = frame[~frame[case_column].isin(rare_cases)].reset_index(drop=True)

        grouped = self.frame.groupby(case_column, sort=False)
        self.case_ids = list(grouped.groups.keys())
        self.traces = {case: group[activity_column].tolist() for case, group in grouped}
        self.labels = {case: int(group[label_column].iloc[0]) for case, group in grouped}
        self.activities = sorted(self.frame[activity_column].unique())
        self.max_length = int(self.frame.groupby(case_column).size().max())
        self.allowed_locations = {
            activity: sorted(group[order_column].unique().tolist())
            for activity, group in self.frame.groupby(activity_column)
        }

        self._column_of = {activity: index for index, activity in enumerate([self.PADDING, *self.activities])}

    @staticmethod
    def clean_activity_name(name: str) -> str:
        """Activity name as the reference code writes it: lower case, without spaces, ``-`` and ``_``.

        Also used to write the gold standard of the simulated logs in the same spelling.
        """
        return name.lower().replace(" ", "").replace("-", "").replace("_", "")

    @classmethod
    def from_csv(cls, path: str, **kwargs) -> "PreparedLog":
        return cls(pd.read_csv(path), **kwargs)

    def encode(self, traces: list[list[str]]) -> np.ndarray:
        """Index-based one-hot encoding: one column per (position, activity), 0/1 values."""
        width = len(self._column_of)
        encoded = np.zeros((len(traces), self.max_length, width), dtype=np.float32)
        for row, trace in enumerate(traces):
            for position in range(self.max_length):
                activity = trace[position] if position < len(trace) else self.PADDING
                encoded[row, position, self._column_of[activity]] = 1
        return encoded.reshape(len(traces), -1)

    def to_event_log(self, numeric: list[str], categorical: list[str]) -> EventLog:
        """The cleaned log as ``EventLog`` for the itemset selection strategies."""
        activity_ids = {name: index for index, name in enumerate(self.activities)}
        first_events = self.frame.groupby(self.case_column, sort=False).first()
        attributes = first_events[numeric + categorical].copy()
        for column in categorical:
            attributes[column] = attributes[column].astype(str)
        return EventLog(
            self.case_ids,
            self.activities,
            [[activity_ids[a] for a in self.traces[case]] for case in self.case_ids],
            [self.labels[case] for case in self.case_ids],
            attributes,
        )
