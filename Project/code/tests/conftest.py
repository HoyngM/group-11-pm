"""Shared helpers for the tests."""

import numpy as np
import pandas as pd

from tools import DataManager


def make_log(traces, labels, ages=None, frq_threshold: int = 1) -> DataManager:
    """Small event log from lists of activity names, with an ``Age`` attribute per case.

    ``frq_threshold`` defaults to 1, so that no case of a tiny test log is removed.
    """
    ages = ages if ages is not None else range(len(traces))
    # One row per event, in the same column layout as the BPIC11 files.
    rows = [{"case:concept:name": f"c{case:04d}", "concept:name": activity, "event_nr": position,
             "label": label, "Age": age}
            for case, (trace, label, age) in enumerate(zip(traces, labels, ages))
            for position, activity in enumerate(trace, start=1)]
    return DataManager(pd.DataFrame(rows), numeric_attributes=["Age"], frq_threshold=frq_threshold)


def make_permuter(allowed_locations: dict, seed: int = 0) -> DataManager:
    """A DataManager whose permutation uses the given observed locations and random seed.

    The permutation methods read ``allowed_locations`` and ``_random``; the importance methods
    normally set ``_random`` themselves.
    """
    data_manager = make_log([["a"]], [0])
    data_manager.allowed_locations = allowed_locations
    data_manager._random = np.random.RandomState(seed)
    return data_manager
