"""Cross-validated location permutation importance.

The idea: train a model that predicts the outcome from *where* each activity happens (the index
encoding). Then move an itemset to other locations in the traces and predict again. If the
predictions get worse (lower F1 score), the model relied on the location of that itemset, so
its location is *important*. Importance = F1 before - F1 after the permutation.

Ported from ``CrossValidation_ProcessPermutation.py`` of the permutation repository (see
DIFFERENCES.md). Per fold an XGBoost model with default parameters is trained on the index
encoding of the training cases, and the importance is the drop in weighted F1 *on the held-out
test cases of that fold*, repeated ``n_repeats`` times. Measuring on cases the model has not
seen avoids that it simply recognises memorised training cases. Single activities use
``trace_permutation_importance``, larger itemsets ``itemset_permutation_importance`` (both in
``tools.py``).

*Cross-validation* (``k_fold`` folds): the cases are split into k parts. Each part is the test set
once, and the model of that fold is trained on the other k - 1 parts. Every itemset therefore gets
``k_fold x n_repeats`` importance values (5 x 10 = 50 by default).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import pandas as pd
from sklearn.metrics import f1_score
from xgboost import XGBClassifier

from tools import DataManager


@dataclass
class ImportanceResult:
    """Output of ``CrossValidationProcessPermutation.run``."""

    importance: pd.DataFrame  # itemset, size, fold, iter, importance
    fold_scores: pd.DataFrame  # fold, train_f1, test_f1 (weighted)
    excluded_itemsets: list[frozenset[str]]  # itemsets with a fixed-location activity (not permuted)


class CrossValidationProcessPermutation:
    """Location permutation importance of itemsets over ``k_fold`` folds and ``n_repeats`` repetitions.

    Args:
        k_fold: number of cross-validation folds.
        n_repeats: permutations per itemset and fold.
        constrain: only move activities to locations at which they were observed in the log.
        seed: seed of the fold split.
        random_state: seed of the permutations.
        exclude_fixed_locations: leave out itemsets with an activity whose location never varies
            (``DataManager.fixed_location_activities``, e.g. a start or end activity).
    """

    def __init__(self, k_fold: int = 5, n_repeats: int = 10, constrain: bool = True,
                 seed: int = 42, random_state: int = 2023, exclude_fixed_locations: bool = True) -> None:
        self.k_fold, self.n_repeats, self.constrain = k_fold, n_repeats, constrain
        self.seed, self.random_state = seed, random_state
        self.exclude_fixed_locations = exclude_fixed_locations

    def run(self, data_manager: DataManager, itemsets: Sequence[frozenset[str]]) -> ImportanceResult:
        """Importance of every itemset in every fold and repetition, plus the model's F1 per fold."""
        # An activity that is always at the same place cannot have an important location, and moving
        # it would create unrealistic traces (e.g. "close" in the middle). Such itemsets are left out.
        fixed = set(data_manager.fixed_location_activities) if self.exclude_fixed_locations else set()
        excluded_itemsets = [itemset for itemset in itemsets if itemset & fixed]
        itemsets = [itemset for itemset in itemsets if not itemset & fixed]
        # Single activities and larger itemsets are permuted by different procedures.
        activities = [next(iter(itemset)) for itemset in itemsets if len(itemset) == 1]
        candidate_itemsets = [itemset for itemset in itemsets if len(itemset) > 1]
        # Per fold the row numbers of the training cases and of the test cases.
        train_list, test_list = data_manager.cross_split_test_train(self.k_fold, self.seed)

        permutation_importance_test_all, scores = [], []
        for i in range(self.k_fold):
            # Train the model of this fold on the encoded training cases.
            train_cases = [data_manager.case_ids[row] for row in train_list[i]]
            test_cases = [data_manager.case_ids[row] for row in test_list[i]]
            train_x = data_manager.index_encoding([data_manager.traces[c] for c in train_cases])
            train_y = data_manager.labels[train_list[i]]
            test_x = data_manager.index_encoding([data_manager.traces[c] for c in test_cases])
            test_y = data_manager.labels[test_list[i]]
            model = XGBClassifier()  # default parameters
            model.fit(train_x, train_y)

            # How good is the model? Weighted F1 = the F1 of each class, averaged with the number
            # of cases per class as weight. The test F1 is the baseline of the importance.
            scores.append({"fold": i,
                           "train_f1": f1_score(train_y, model.predict(train_x), average="weighted"),
                           "test_f1": f1_score(test_y, model.predict(test_x), average="weighted")})

            # Importance on the held-out test cases of this fold: first the single activities,
            # then the larger itemsets.
            fold_rows = data_manager.trace_permutation_importance(
                model, test_x, test_y, test_cases, activities, constrain=self.constrain,
                n_repeats=self.n_repeats, random_state=self.random_state)
            fold_rows += data_manager.itemset_permutation_importance(
                model, test_x, test_y, test_cases, candidate_itemsets,
                n_repeats=self.n_repeats, random_state=self.random_state)
            permutation_importance_test_all += [{**row, "fold": i} for row in fold_rows]

        importance = pd.DataFrame(permutation_importance_test_all,
                                  columns=["itemset", "size", "fold", "iter", "importance"])
        return ImportanceResult(importance, pd.DataFrame(scores), excluded_itemsets)
