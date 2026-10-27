"""``DataManager``: event log, encoding, Apriori itemsets and location permutation.

Ported from ``tools.py`` of https://github.com/MozhganVD/PermutationLocationImportance
(Vazifehdoostirani et al., 2024); see DIFFERENCES.md. The event log is loaded and queried with pm4py.

Vocabulary used throughout the code:

* a *case* is one process instance (e.g. one patient), identified by its case id;
* an *event* is one row of the CSV: one activity executed in one case;
* a *trace* is the list of activity names of one case, in execution order;
* the *outcome* (label) is the 0/1 result of a case that the model has to predict;
* a *location* is the position of an event in its trace (1 = first event), like ``event_nr``;
* *permuting the location* of an activity means: take it out of the trace and put it back at
  another position; the other events shift to make room (``a b c d`` -> move c to 1 -> ``c a b d``).
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import pm4py
from mlxtend.frequent_patterns import apriori
from mlxtend.preprocessing import TransactionEncoder
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold, train_test_split

MIN_SHUFFLED_CASES = 3  # single activities: with fewer shuffled cases the importance counts as 0


class DataManager:
    """A labelled event log with everything the location permutation importance needs.

    Cases are kept in sorted order of their id, so row ``i`` of
    ``case_table`` and of every encoding belongs to ``case_ids[i]``.

    Args:
        address: path of the CSV event log (one row per event), or an already loaded DataFrame.
        columns: maps our standard names (case, activity, label, order) to the file's column names.
        numeric_attributes: numeric case attributes (used by the IMPresseD case distance).
        categorical_attributes: categorical case attributes (idem).
        frq_threshold: cases with an activity that occurs fewer times in the log are removed.
    """

    PADDING = "0"  # padding symbol for short traces in the one-hot encoding

    def __init__(self, address: str | Path | pd.DataFrame, columns: dict | None = None,
                 numeric_attributes: Sequence[str] = (), categorical_attributes: Sequence[str] = (),
                 frq_threshold: int = 2) -> None:
        # Standard (XES) column names; every log is renamed to these.
        self.case_id = "case:concept:name"
        self.activity = "concept:name"
        self.time_col = "time:timestamp"
        self.outcome = "label"
        self.order = "event_nr"
        self.frq_threshold = frq_threshold
        self.numeric_attributes = list(numeric_attributes)
        self.categorical_attributes = list(categorical_attributes)
        self.data = self._load_df(address, columns or {})

        # Control-flow view: one list of activity names per case, in event order.
        # After _load_df the rows are sorted by case id, so unique() lists the cases in sorted order.
        self.case_ids = list(self.data[self.case_id].unique())
        # pm4py returns the traces in that same case order, e.g. [["register", "b", "close"], ...].
        traces = pm4py.project_on_event_attribute(self.data, self.activity)
        self.traces = dict(zip(self.case_ids, traces))  # case id -> trace
        self.activities = sorted(self.data[self.activity].unique())  # every distinct activity name
        self.max_length = max(len(trace) for trace in traces)  # length of the longest trace
        # Observed 1-based locations per activity: the feasible targets of a permutation.
        # Example: b is the 2nd event in one case and the 5th in another -> b: [2, 5].
        # pm4py counts positions from 0, event_nr from 1, hence the + 1.
        self.allowed_locations = {
            activity: sorted(position + 1 for position in pm4py.get_activity_position_summary(self.data, activity))
            for activity in self.activities
        }
        self.fixed_location_activities = self._find_fixed_location_activities()

        # Case-level view: outcome and case attributes, taken from the first event of each case
        # (in the CSV these values are repeated on every event of the case).
        first_events = self.data.groupby(self.case_id, sort=False).first()
        # reset_index turns the case id from the index back into a normal column.
        self.case_table = first_events[[self.outcome, *self.numeric_attributes,
                                        *self.categorical_attributes]].reset_index()
        self.labels = self.case_table[self.outcome].to_numpy()  # numpy array of 0/1, one value per case
        # Column of every activity in the one-hot encoding; column 0 is the padding symbol.
        self._column_of = {activity: i for i, activity in enumerate([self.PADDING, *self.activities])}
        self._random = np.random.RandomState(2023)  # replaced at the start of every importance run

    # --- loading -------------------------------------------------------------------------

    def _load_df(self, address: str | Path | pd.DataFrame, columns: dict) -> pd.DataFrame:
        """Read the log, clean names and labels, order the events, remove cases with rare activities."""
        frame = address.copy() if isinstance(address, pd.DataFrame) else pd.read_csv(address)
        # Rename the file's columns to the standard names; e.g. {"case": "CaseID"} renames
        # "CaseID" to "case:concept:name". Without a mapping the names stay the same.
        frame = frame.rename(columns={
            columns.get("case", self.case_id): self.case_id, columns.get("activity", self.activity): self.activity,
            columns.get("label", self.outcome): self.outcome, columns.get("order", self.order): self.order})
        frame[self.case_id] = frame[self.case_id].astype(str)
        # "Check-In_Desk 2" becomes "checkindesk2".
        frame[self.activity] = (frame[self.activity].astype(str).str.lower()
                                .str.replace(" ", "").str.replace("-", "").str.replace("_", ""))
        # Some logs use text labels instead of 0/1.
        frame[self.outcome] = frame[self.outcome].replace({"deviant": 1, "regular": 0}).astype(int)
        # The traces are treated as totally ordered: the event number is used as the time, so
        # pm4py orders the events by event_nr and no two events of a case are simultaneous.
        # (event 1 -> 2000-01-01 00:00:01, event 2 -> 00:00:02, ...; the real BPIC11 timestamps
        # only contain a date, so most events of a case would otherwise get the same time.)
        frame[self.time_col] = pd.Timestamp("2000-01-01") + pd.to_timedelta(frame[self.order], unit="s")
        # format_dataframe checks the columns, converts the time column and sorts by case and time.
        frame = pm4py.format_dataframe(frame, case_id=self.case_id, activity_key=self.activity,
                                       timestamp_key=self.time_col)

        # Remove cases that contain very low frequent activities.
        act_frequency = pm4py.get_event_attribute_values(frame, self.activity)  # {activity: number of events}
        low_frequent_acts = [act for act, count in act_frequency.items() if count < self.frq_threshold]
        if low_frequent_acts:
            # level="case" + retain=False: drop whole cases that contain one of the rare activities.
            frame = pm4py.filter_event_attribute_values(frame, self.activity, low_frequent_acts,
                                                        level="case", retain=False)
        return frame

    def _find_fixed_location_activities(self) -> list[str]:
        """Activities whose location never varies, so that their location cannot be important.

        An activity has a fixed location if every occurrence in the log is at the same position
        counted from the start (e.g. a start activity, always event 1) or at the same position
        counted from the end (e.g. an end activity, always the last event, whatever the trace length).
        """
        from_start, from_end = {}, {}
        for trace in self.traces.values():
            for position, activity in enumerate(trace, start=1):
                from_start.setdefault(activity, set()).add(position)
                from_end.setdefault(activity, set()).add(len(trace) - position + 1)  # 1 = last event
        return [activity for activity in self.activities
                if len(from_start[activity]) == 1 or len(from_end[activity]) == 1]

    def __len__(self) -> int:
        """Number of cases, so that ``len(data_manager)`` works."""
        return len(self.case_ids)

    @property
    def number_of_events(self) -> int:
        """Total number of events over all cases."""
        return sum(len(trace) for trace in self.traces.values())

    # --- itemsets and splits ---------------------------------------------------------------

    def frequent_activity_sets(self, min_support: float, top_k: int, max_length: int | None = None,
                               case_list: Sequence[str] | None = None) -> tuple[list[list[str]], pd.DataFrame]:
        """The ``top_k`` most frequent itemsets with 2+ activities, found with Apriori.

        An *itemset* is a set of activities, e.g. {a, b}; its *support* is the share of cases whose
        trace contains all of them. Returns the itemsets as lists and the Apriori table
        (columns ``support``, ``itemsets``). ``max_length`` limits the itemset size (``None`` = no
        limit); ``case_list`` defaults to all cases.
        """
        # Every case is one transaction: the set of activities in its trace.
        transactions = [self.traces[case] for case in (case_list or self.case_ids)]
        # TransactionEncoder makes a True/False table: one row per case, one column per activity,
        # True if the activity occurs in that case (how often does not matter).
        encoder = TransactionEncoder()
        one_hot = pd.DataFrame(encoder.fit(transactions).transform(transactions), columns=encoder.columns_)
        frequent_itemsets = apriori(one_hot, min_support=min_support, use_colnames=True, max_len=max_length)
        frequent_itemsets = frequent_itemsets.sort_values("support", ascending=False)
        selected_itemsets = frequent_itemsets[frequent_itemsets["itemsets"].apply(len) > 1].head(top_k)
        frequent_activities_list = selected_itemsets["itemsets"].apply(list).to_list()
        return frequent_activities_list, selected_itemsets

    def activity_support(self, case_list: Sequence[str] | None = None) -> pd.Series:
        """Share of cases that contain each activity (the support of every single activity).

        Only activities that occur in at least one of the cases are listed, alphabetically.
        """
        cases = case_list or self.case_ids
        support = pd.Series({activity: sum(activity in self.traces[c] for c in cases) / len(cases)
                             for activity in self.activities})
        return support[support > 0]

    def split_test_train(self, test_size: float, seed: int) -> list[str]:
        """Case ids of the training part of a stratified train/test split.

        *Stratified* means that the share of label 1 is the same in both parts.
        """
        train, _ = train_test_split(self.case_ids, test_size=test_size, random_state=seed, stratify=self.labels)
        return sorted(train)  # the test part (_) is not needed by the callers

    def cross_split_test_train(self, k_fold: int, seed: int) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
        """Stratified k-fold split: per fold the row numbers of the training and the test cases.

        Every fold has about the same share of label 1 as the whole log; ``seed`` makes the
        folds reproducible.
        """
        splitter = StratifiedKFold(k_fold, shuffle=True, random_state=seed)
        train_dict, test_dict = {}, {}
        for i, (train_index, test_index) in enumerate(splitter.split(self.case_ids, self.labels)):
            train_dict[i], test_dict[i] = list(train_index), list(test_index)
        return train_dict, test_dict

    # --- encoding --------------------------------------------------------------------------

    def index_encoding(self, traces: Sequence[Sequence[str]]) -> np.ndarray:
        """Index-based one-hot encoding: one 0/1 column per (position, activity).

        Traces shorter than the longest trace are padded with ``PADDING``. Example with activities
        a, b and max_length 2: the trace ["b"] becomes position 1: [pad=0, a=0, b=1],
        position 2: [pad=1, a=0, b=0] -> row [0, 0, 1, 1, 0, 0]. The model can therefore see
        *where* an activity happens, which is what the location permutation changes.
        """
        # 3D array of zeros: traces x positions x (padding + activities).
        encoded = np.zeros((len(traces), self.max_length, len(self._column_of)), dtype=np.float32)
        for row, trace in enumerate(traces):
            for position in range(self.max_length):
                activity = trace[position] if position < len(trace) else self.PADDING
                encoded[row, position, self._column_of[activity]] = 1
        # Flatten (position, activity) into one feature axis for the classifier.
        return encoded.reshape(len(traces), -1)

    # --- location permutation importance ---------------------------------------------------

    def trace_permutation_importance(self, model, x: np.ndarray, y: np.ndarray, case_list: Sequence[str],
                                     activities: Sequence[str], constrain: bool = True, n_repeats: int = 10,
                                     random_state: int = 2023) -> list[dict]:
        """Location importance of single activities.

        ``x`` and ``y`` are the encoding and labels of the cases in ``case_list`` (the held-out test
        cases of one fold). Each activity is moved ``n_repeats`` times in every case that contains
        it; the importance is the drop in weighted F1. Every repetition starts from the original
        traces, so the result for one activity does not depend on the others. Returns one row per
        activity and repetition.
        """
        self._random = np.random.RandomState(random_state)  # fresh random draws for every call
        predicted = model.predict(x)
        baseline = f1_score(y, predicted, average="weighted")
        permutation_importance = []
        for act in activities:
            for r in range(n_repeats):
                importance = 0.0
                if len(self.allowed_locations[act]) >= 2:  # otherwise there is no alternative location
                    # Fresh working copy of the original traces for every repetition.
                    sub_data = {case: list(self.traces[case]) for case in case_list}
                    shuffled_cases = []
                    for row, case in enumerate(case_list):
                        if act in sub_data[case]:
                            shuffled_cases.append(row)
                            sub_data[case] = self._shuffle_activity(sub_data[case], act, constrain)
                    # With too few changed cases the importance counts as 0.
                    if len(shuffled_cases) >= MIN_SHUFFLED_CASES:
                        importance = self._f1_drop(model, predicted, baseline, y, sub_data, case_list, shuffled_cases)
                permutation_importance.append({"itemset": act, "size": 1, "iter": r, "importance": importance})
        return permutation_importance

    def itemset_permutation_importance(self, model, x: np.ndarray, y: np.ndarray, case_list: Sequence[str],
                                       frequent_itemsets: Sequence[frozenset[str]], n_repeats: int = 10,
                                       random_state: int = 2023) -> list[dict]:
        """Location importance of multi-activity itemsets.

        The same as ``trace_permutation_importance``, but every occurrence of the itemset is moved
        with ``shuffle_sequence``, which keeps the order of its activities. Cases that consist of
        nothing but the itemset are skipped.
        """
        self._random = np.random.RandomState(random_state)  # fresh random draws for every call
        predicted = model.predict(x)
        baseline = f1_score(y, predicted, average="weighted")
        permutation_importance = []
        for itemset in frequent_itemsets:
            for r in range(n_repeats):
                # Fresh working copy of the original traces for every repetition.
                sub_data = {case: list(self.traces[case]) for case in case_list}
                shuffled_cases = []
                for row, case in enumerate(case_list):
                    trace = sub_data[case]
                    # issubset: the trace contains every activity of the itemset (at least once).
                    if itemset.issubset(trace) and len(itemset) != len(trace):
                        shuffled_cases.append(row)
                        sub_data[case] = self.shuffle_sequence(trace, itemset)
                importance = (self._f1_drop(model, predicted, baseline, y, sub_data, case_list, shuffled_cases)
                              if shuffled_cases else 0.0)
                permutation_importance.append({"itemset": ", ".join(sorted(itemset)), "size": len(itemset),
                                               "iter": r, "importance": importance})
        return permutation_importance

    def _f1_drop(self, model, predicted, baseline, y, sub_data, case_list, shuffled_cases) -> float:
        """Baseline F1 minus the F1 after permuting; only the shuffled cases get a new encoding.

        The other cases keep their original prediction.
        """
        new_predicted = predicted.copy()
        new_predicted[shuffled_cases] = model.predict(
            self.index_encoding([sub_data[case_list[row]] for row in shuffled_cases]))
        return baseline - f1_score(y, new_predicted, average="weighted")

    def _shuffle_activity(self, trace: list[str], act: str, constrain: bool = True) -> list[str]:
        """Move each occurrence of ``act`` to a random (observed, if ``constrain``) location.

        Returns a new trace; the other events shift to make room, as with ``list.insert``.
        """
        trace = list(trace)  # work on a copy
        # Positions of every occurrence, found before anything moves.
        for index in [i for i, x in enumerate(trace) if x == act]:
            if constrain:
                # Observed locations that exist in this trace; convert 1-based to a 0-based index.
                adjusted_allowed = [i for i in self.allowed_locations[act] if i < len(trace) + 1]
                random_index = self._random.choice(adjusted_allowed, 1)[0] - 1
            else:
                random_index = self._random.choice(len(trace), 1)[0]  # any position of the trace
            # pop removes the event at ``index``; insert puts it back at ``random_index``.
            trace.insert(random_index, trace.pop(index))
        return trace

    def shuffle_sequence(self, sequence: list[str], item_set: frozenset[str]) -> list[str]:
        """Move the occurrences of ``item_set`` to random observed locations, keeping their order.

        Returns a new trace with the same activities. Example: for {a, b} in ``a x b y`` the events
        a and b are moved one after the other, b to a location after the new location of a.
        """
        # Every event keeps its original position as a label, so it can always be found again
        # after other events have moved: [(0, "a"), (1, "x"), (2, "b"), (3, "y")].
        events = list(enumerate(sequence))
        # Every non-overlapping occurrence, e.g. [(0, 2)] for {a, b} in "a x b y".
        for idx in self.find_itemset_indexes(sequence, item_set):
            subset = [sequence[i] for i in idx]  # the activities of this occurrence, in order
            max_possibles = self._max_possible_locations(subset)
            if max_possibles is None:  # no valid location: skip this occurrence
                continue
            # Place the activities one by one, each at a location after the previous one.
            previous_index = -1
            for original_index, act in zip(idx, subset):
                # Observed locations after the previous activity and not too late for the next ones.
                adjusted_allowed = [index for index in self.allowed_locations[act]
                                    if previous_index < index <= max_possibles[act]]
                if not adjusted_allowed:  # no valid location: stop with this occurrence
                    break
                random_index = self._random.choice(adjusted_allowed, 1)[0]  # 1-based target location
                previous_index = random_index
                # Find this event by its original position and move it to the target (0-based).
                # pop + insert never changes the order of the other events, so every activity of
                # the itemset ends up after the ones placed before it.
                current = next(p for p, (index, _) in enumerate(events) if index == original_index)
                events.insert(random_index - 1, events.pop(current))
        return [act for _, act in events]

    def _max_possible_locations(self, subset: list[str]) -> dict[str, int] | None:
        """Latest observed location of every activity at which the order of ``subset`` can still be kept.

        Returns ``None`` if some activity has no observed location before the next one.
        Example: for [a, b] with b observed at {2, 5} and a at {1, 6}: b at most 5, so a at most 1.
        """
        max_possibles, max_location = {}, 10000  # 10000 = "no limit" for the last activity
        # Work backwards: each activity must be placed before the latest location of the next one.
        for act in reversed(subset):
            candidates = [a for a in self.allowed_locations[act] if a < max_location]
            if not candidates:
                return None
            max_possibles[act] = max_location = max(candidates)  # also the bound for the activity before
        return max_possibles

    @staticmethod
    def find_itemset_indexes(sequence: list[str], itemset: frozenset[str]) -> list[tuple[int, ...]]:
        """Non-overlapping index combinations of ``sequence`` whose activities are exactly the itemset.

        Example: {a, b} in ["a", "b", "a", "b"] gives [(0, 1), (2, 3)]; (0, 3) and (1, 2) are
        skipped because they share an event with an occurrence that was kept before.
        """
        # Try every combination of positions of the right size (e.g. all pairs for 2 activities).
        indexes = [combination for combination in itertools.combinations(range(len(sequence)), len(itemset))
                   if {sequence[i] for i in combination} == set(itemset)]
        # Keep a combination only if it shares no event with one that was kept before.
        remaining_indexes: list[tuple[int, ...]] = []
        for t in indexes:
            if all(not set(t) & set(r_idx) for r_idx in remaining_indexes):
                remaining_indexes.append(t)
        return remaining_indexes
