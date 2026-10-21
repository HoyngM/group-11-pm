"""The three interest functions of IMPresseD for outcome-oriented patterns."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import mutual_info_score


@dataclass(frozen=True)
class Interest:
    """Interest of one pattern: frequency and outcome (maximise), distance (minimise)."""

    frequency: float
    outcome: float
    distance: float


class InterestCalculator:
    """Computes the interest of a pattern from its per-case instance counts.

    Args:
        labels: binary outcome per case.
        distances: symmetric (cases x cases) matrix of case distances.
    """

    def __init__(self, labels: np.ndarray, distances: np.ndarray) -> None:
        self.labels = np.asarray(labels)
        self.distances = distances

    def evaluate(self, counts: np.ndarray) -> Interest:
        """Interest of a pattern given how often it occurs in every case."""
        occurs = counts > 0
        return Interest(
            frequency=float(occurs.mean()),
            outcome=self._information_gain(counts),
            distance=self._case_distance(occurs),
        )

    def _information_gain(self, counts: np.ndarray) -> float:
        """Information gain of the label from the pattern's frequency vector."""
        return float(mutual_info_score(self.labels, counts))

    def _case_distance(self, occurs: np.ndarray) -> float:
        """Mean distance between cases with and without the pattern.

        Patterns that occur in all or none of the cases cannot be compared and
        get an infinite (worst) distance.
        """
        inside, outside = np.flatnonzero(occurs), np.flatnonzero(~occurs)
        if len(inside) == 0 or len(outside) == 0:
            return float("inf")
        return float(self.distances[np.ix_(inside, outside)].mean())
