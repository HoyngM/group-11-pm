"""Pairwise distance between cases, based on their case-level attributes."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class CaseDistance:
    """Distance between cases, used for the case-distance interest function.

    Following Vazifehdoostirani et al. (2023), the distance of two cases is
    ``(normalised_euclidean + m * jaccard) / (m + 1)`` where ``m`` is the number
    of categorical attributes. The Jaccard distance is computed on the sets of
    (attribute, value) pairs of both cases.
    """

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str]) -> None:
        self.numeric = list(numeric)
        self.categorical = list(categorical)

    def pairwise(self, case_attributes: pd.DataFrame) -> np.ndarray:
        """Return the symmetric (cases x cases) distance matrix."""
        size = len(case_attributes)
        if not self.numeric and not self.categorical:
            return np.zeros((size, size))

        numeric_part = self._normalised_euclidean(case_attributes)
        categorical_part = self._jaccard(case_attributes)
        m = len(self.categorical)

        if numeric_part is not None and categorical_part is not None:
            return (numeric_part + m * categorical_part) / (m + 1)
        return numeric_part if numeric_part is not None else categorical_part

    def _normalised_euclidean(self, case_attributes: pd.DataFrame) -> np.ndarray | None:
        if not self.numeric:
            return None
        values = case_attributes[self.numeric].to_numpy(dtype=float)
        distances = cdist(values, values, metric="euclidean")
        largest = distances.max()
        return distances / largest if largest > 0 else distances

    def _jaccard(self, case_attributes: pd.DataFrame) -> np.ndarray | None:
        if not self.categorical:
            return None
        m = len(self.categorical)
        codes = np.column_stack(
            [pd.factorize(case_attributes[column])[0] for column in self.categorical]
        )
        # Number of attributes on which two cases agree: |A intersect B|.
        agreeing = (codes[:, None, :] == codes[None, :, :]).sum(axis=2)
        # Both sets contain m pairs, so |A union B| = 2m - |A intersect B|.
        return 1.0 - agreeing / (2 * m - agreeing)
