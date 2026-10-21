"""5-fold cross-validated location permutation importance (``CrossValidation_ProcessPermutation.py``).

Per fold an XGBoost model (default parameters) is trained on the index-based one-hot
encoding of the training cases. The importance of an itemset is the drop in weighted F1
after permuting the location of the itemset, repeated ``repetitions`` times. The reference
code measures this on the *training* cases of the fold (``evaluate_on="train"``); the paper
text describes an evaluation by cross-validation, which corresponds to ``evaluate_on="test"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from location_permutation.log import PreparedLog
from location_permutation.permuter import ReferencePermuter

MIN_SHUFFLED_CASES = 3  # single activities: fewer shuffled cases count as importance 0


@dataclass
class ImportanceResult:
    importance: pd.DataFrame  # itemset, size, fold, repetition, importance
    fold_scores: pd.DataFrame  # fold, train_f1, test_f1 (weighted)


class CrossValidatedLocationImportance:
    """Location permutation importance of itemsets, over k folds and several repetitions."""

    def __init__(self, folds: int = 5, repetitions: int = 10, constrain: bool = True, seed: int = 42,
                 permutation_seed: int = 2023, accumulate_permutations: bool = False,
                 evaluate_on: str = "train", exact_tracking: bool = False) -> None:
        if evaluate_on not in {"train", "test"}:
            raise ValueError("evaluate_on must be 'train' or 'test'")
        self.evaluate_on, self.exact_tracking = evaluate_on, exact_tracking
        self.folds, self.repetitions, self.constrain = folds, repetitions, constrain
        self.seed, self.permutation_seed = seed, permutation_seed
        # The reference code never resets the permuted traces, so permutations pile up over
        # itemsets and repetitions. False starts every repetition from the original traces.
        self.accumulate_permutations = accumulate_permutations

    def run(self, log: PreparedLog, itemsets: Sequence[frozenset[str]]) -> ImportanceResult:
        labels = np.array([log.labels[case] for case in log.case_ids])
        splitter = StratifiedKFold(self.folds, shuffle=True, random_state=self.seed)

        rows, scores = [], []
        for fold, (train_index, test_index) in enumerate(splitter.split(log.case_ids, labels)):
            train_cases = [log.case_ids[i] for i in train_index]
            test_cases = [log.case_ids[i] for i in test_index]
            model = XGBClassifier(random_state=self.seed)
            model.fit(log.encode([log.traces[c] for c in train_cases]), labels[train_index])

            train_prediction = model.predict(log.encode([log.traces[c] for c in train_cases]))
            test_prediction = model.predict(log.encode([log.traces[c] for c in test_cases]))
            train_f1 = f1_score(labels[train_index], train_prediction, average="weighted")
            test_f1 = f1_score(labels[test_index], test_prediction, average="weighted")
            scores.append({"fold": fold, "train_f1": train_f1, "test_f1": test_f1})

            if self.evaluate_on == "train":
                cases, case_labels, baseline_prediction, baseline = train_cases, labels[train_index], train_prediction, train_f1
            else:
                cases, case_labels, baseline_prediction, baseline = test_cases, labels[test_index], test_prediction, test_f1
            rows.extend(self._fold_importance(log, model, cases, case_labels,
                                              baseline_prediction, baseline, itemsets, fold))
        return ImportanceResult(pd.DataFrame(rows), pd.DataFrame(scores))

    def _fold_importance(self, log, model, cases, labels, baseline_prediction, baseline, itemsets, fold):
        permuter = ReferencePermuter(log.allowed_locations, seed=self.permutation_seed,
                                     exact_tracking=self.exact_tracking)
        permuted = {case: list(log.traces[case]) for case in cases}
        rows = []
        for itemset in itemsets:
            for repetition in range(self.repetitions):
                if not self.accumulate_permutations:
                    permuted = {case: list(log.traces[case]) for case in cases}
                changed = self._permute(permuter, log, permuted, cases, itemset)
                if changed:
                    prediction = baseline_prediction.copy()
                    prediction[changed] = model.predict(log.encode([permuted[cases[i]] for i in changed]))
                    importance = baseline - f1_score(labels, prediction, average="weighted")
                else:
                    importance = 0.0
                rows.append({"itemset": ", ".join(sorted(itemset)), "size": len(itemset), "fold": fold,
                             "repetition": repetition, "importance": importance})
        return rows

    def _permute(self, permuter, log, permuted, cases, itemset) -> list[int]:
        """Permute ``itemset`` in the traces that contain it; return the row numbers that changed."""
        changed = []
        if len(itemset) == 1:
            (activity,) = itemset
            if len(log.allowed_locations[activity]) < 2:
                return []  # no alternative location
            for row, case in enumerate(cases):
                if activity in permuted[case]:
                    changed.append(row)
                    permuted[case] = permuter.shuffle_activity(permuted[case], activity, self.constrain)
            return changed if len(changed) >= MIN_SHUFFLED_CASES else []

        for row, case in enumerate(cases):
            trace = permuted[case]
            if itemset.issubset(trace) and len(itemset) != len(trace):
                changed.append(row)
                permuted[case] = permuter.shuffle_itemset(trace, itemset)
        return changed
