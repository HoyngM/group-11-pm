"""Invariants of the generator and the shuffled controls."""

import pandas as pd
import pytest

from synthetic_data import ShuffleControls, SyntheticLogGenerator
from synthetic_data.generator import ACTIVITY, CASE, GeneratorConfig, LABEL


@pytest.fixture(scope="module")
def log() -> pd.DataFrame:
    return SyntheticLogGenerator(GeneratorConfig(target_events=None, number_of_cases=100)).generate()


def activity_sets(frame: pd.DataFrame) -> dict:
    return frame.groupby(CASE)[ACTIVITY].apply(lambda s: sorted(s)).to_dict()


def test_every_case_contains_b_once(log):
    assert (log[log[ACTIVITY] == "B"].groupby(CASE).size() == 1).all()
    assert log[log[ACTIVITY] == "B"][CASE].nunique() == log[CASE].nunique()


def test_generator_is_deterministic():
    config = GeneratorConfig(target_events=None, number_of_cases=20)
    pd.testing.assert_frame_equal(SyntheticLogGenerator(config).generate(), SyntheticLogGenerator(config).generate())


def test_within_case_keeps_activity_sets_but_changes_order(log):
    shuffled = ShuffleControls().within_case(log)
    assert activity_sets(shuffled) == activity_sets(log)
    assert not shuffled[ACTIVITY].equals(log[ACTIVITY])


def test_outcome_shuffle_keeps_traces_and_label_distribution(log):
    shuffled = ShuffleControls().outcomes(log)
    assert shuffled[ACTIVITY].equals(log[ACTIVITY])
    assert shuffled.groupby(CASE)[LABEL].nunique().max() == 1  # label still constant per case
    assert shuffled.groupby(CASE)[LABEL].first().sum() == log.groupby(CASE)[LABEL].first().sum()
    assert not shuffled[LABEL].equals(log[LABEL])


def test_across_traces_keeps_global_activity_counts(log):
    shuffled = ShuffleControls().across_traces(log)
    assert shuffled[ACTIVITY].value_counts().to_dict() == log[ACTIVITY].value_counts().to_dict()
    assert activity_sets(shuffled) != activity_sets(log)


def test_input_log_is_not_modified(log):
    before = log.copy()
    ShuffleControls().all_controls(log)
    pd.testing.assert_frame_equal(log, before)


def test_default_log_has_exactly_100_events():
    assert len(SyntheticLogGenerator().generate()) == 100
