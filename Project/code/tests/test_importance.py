"""End-to-end: the planted location effect of b in the synthetic log and its shuffled controls."""

import pandas as pd
import pytest
from scipy.stats import spearmanr

from cross_validation_process_permutation import CrossValidationProcessPermutation
from extensions.shuffle_controls import ShuffleControls
from extensions.synthetic_log import ACTIVITY, CASE, GeneratorConfig, KEY_ACTIVITY, LABEL, ORDER, SyntheticLogGenerator
from tools import DataManager

# The planted activity, a noise activity and a noise pair (names as after cleaning).
ITEMSETS = [frozenset({"b"}), frozenset({"a"}), frozenset({"a", "c"})]


@pytest.fixture(scope="module")
def logs() -> dict[str, pd.DataFrame]:
    """A synthetic log of 200 cases and its two shuffled controls."""
    synthetic = SyntheticLogGenerator(GeneratorConfig(target_events=None, number_of_cases=200)).generate()
    return {"synthetic": synthetic, **ShuffleControls(seed=1).all_controls(synthetic)}


@pytest.fixture(scope="module")
def results(logs):
    """Location importance of ``ITEMSETS`` in every log (3 folds, 3 repetitions, to keep it fast)."""
    return {name: CrossValidationProcessPermutation(k_fold=3, n_repeats=3).run(DataManager(frame), ITEMSETS)
            for name, frame in logs.items()}


def position_label_correlation(frame: pd.DataFrame) -> float:
    """Spearman correlation between the relative position of b in a case (0 = start, 1 = end) and the label."""
    frame = frame.assign(relative=(frame[ORDER] - 1) / (frame.groupby(CASE)[ORDER].transform("max") - 1))
    key = frame[frame[ACTIVITY] == KEY_ACTIVITY].groupby(CASE).first()
    return spearmanr(key["relative"], key[LABEL])[0]


def test_planted_effect_exists_only_in_the_synthetic_log(logs):
    """The position of b predicts the label in the original log, not in the controls."""
    assert position_label_correlation(logs["synthetic"]) > 0.6
    assert abs(position_label_correlation(logs["shuffled_within_case"])) < 0.2
    assert abs(position_label_correlation(logs["shuffled_outcomes"])) < 0.2


def test_model_predicts_well_only_when_the_effect_is_present(results):
    """The model generalises (high test F1) only on the log with the planted effect."""
    assert results["synthetic"].fold_scores["test_f1"].mean() > 0.85
    for control in ("shuffled_within_case", "shuffled_outcomes"):
        assert results[control].fold_scores["test_f1"].mean() < 0.7


def test_location_of_b_is_the_most_important(results):
    """Moving b clearly lowers the test F1, much more than moving the noise activity a."""
    means = results["synthetic"].importance.groupby("itemset")["importance"].mean()
    assert means.idxmax() == "b"
    assert means["b"] > 0.3
    assert means["b"] > 3 * abs(means["a"])


def test_itemsets_with_a_fixed_location_activity_are_excluded(logs):
    """'register' (always first) and 'close' (always last) are not permuted, nor any itemset containing them."""
    itemsets = [frozenset({"b"}), frozenset({"register"}), frozenset({"b", "close"})]
    result = CrossValidationProcessPermutation(k_fold=3, n_repeats=1).run(DataManager(logs["synthetic"]), itemsets)
    assert set(result.excluded_itemsets) == {frozenset({"register"}), frozenset({"b", "close"})}
    assert set(result.importance["itemset"]) == {"b"}


def test_nothing_is_important_in_the_controls(results):
    """Without a location effect, no itemset has a mean F1 drop of note (measured on test cases)."""
    for control in ("shuffled_within_case", "shuffled_outcomes"):
        means = results[control].importance.groupby("itemset")["importance"].mean()
        assert means.abs().max() < 0.1
