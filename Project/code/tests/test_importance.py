"""End-to-end: location of B matters in the synthetic log and not in a label-shuffled copy."""

import pandas as pd
import pytest

from itemset_selection import EventLog  # noqa: F401  (ensures the group's package is importable)
from location_permutation import CrossValidatedLocationImportance, PreparedLog
from synthetic_data import ShuffleControls, SyntheticLogGenerator
from synthetic_data.generator import GeneratorConfig


@pytest.fixture(scope="module")
def synthetic() -> pd.DataFrame:
    return SyntheticLogGenerator(GeneratorConfig(target_events=None, number_of_cases=200)).generate()


def run(frame: pd.DataFrame) -> pd.Series:
    log = PreparedLog(frame)
    result = CrossValidatedLocationImportance(folds=3, repetitions=3).run(
        log, [frozenset({"b"}), frozenset({"a"})])
    return result.importance.groupby("itemset")["importance"].mean()


def test_location_of_b_is_important_in_the_synthetic_log(synthetic):
    means = run(synthetic)
    assert means["b"] > 0.2
    assert means["b"] > 3 * abs(means["a"])


def test_nothing_is_important_when_the_labels_are_shuffled(synthetic):
    shuffled = ShuffleControls(seed=1).outcomes(synthetic)
    assert run(shuffled).abs().max() < run(synthetic)["b"]


def test_encoding_has_one_one_per_position():
    frame = SyntheticLogGenerator(GeneratorConfig(target_events=None, number_of_cases=5)).generate()
    log = PreparedLog(frame)
    encoded = log.encode([log.traces[c] for c in log.case_ids])
    assert (encoded.reshape(len(log.case_ids), log.max_length, -1).sum(axis=2) == 1).all()
