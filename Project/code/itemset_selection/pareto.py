"""Pareto front (non-dominated set) selection."""

from __future__ import annotations

from typing import Sequence

import numpy as np


def non_dominated(values: np.ndarray, maximise: Sequence[bool]) -> np.ndarray:
    """Return a boolean mask of the rows on the Pareto front.

    Row ``l`` dominates row ``j`` if it is no worse on every objective and
    strictly better on at least one. Of several rows with identical objective
    values only the first one is kept.

    Args:
        values: array (candidates x objectives).
        maximise: per objective, True if larger is better, False if smaller is.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return np.zeros(len(values), dtype=bool)

    # Turn every objective into "smaller is better".
    signs = np.where(np.asarray(maximise, dtype=bool), -1.0, 1.0)
    costs = values * signs

    # Keep the first row of every set of identical rows.
    _, first_of_each = np.unique(costs, axis=0, return_index=True)
    candidates = np.sort(first_of_each)

    keep = np.zeros(len(costs), dtype=bool)
    candidate_costs = costs[candidates]
    for position, row in zip(candidates, candidate_costs):
        no_worse = np.all(candidate_costs <= row, axis=1)
        strictly_better = np.any(candidate_costs < row, axis=1)
        keep[position] = not np.any(no_worse & strictly_better)
    return keep
