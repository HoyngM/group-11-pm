"""Invariants of the synthetic log generator and the shuffled controls."""

import pandas as pd
import pytest

from extensions.shuffle_controls import ShuffleControls
from extensions.synthetic_log import ACTIVITY, CASE, GeneratorConfig, LABEL, SyntheticLogGenerator


@pytest.fixture(scope="module")
def log() -> pd.DataFrame:
    """A synthetic log of 100 cases."""
    return SyntheticLogGenerator(GeneratorConfig(target_events=None, number_of_cases=100)).generate()


def activity_sets(frame: pd.DataFrame) -> dict:
    """Per case the sorted list of its activities (order ignored)."""
    return frame.groupby(CASE)[ACTIVITY].apply(sorted).to_dict()


def test_default_log_has_exactly_100_events_and_b_once_per_case():
    """The default log has the agreed 100 events, and every case contains B exactly once."""
    log = SyntheticLogGenerator().generate()
    assert len(log) == 100
    assert (log[log[ACTIVITY] == "B"].groupby(CASE).size() == 1).all()
    assert log[log[ACTIVITY] == "B"][CASE].nunique() == log[CASE].nunique()


def test_generator_is_deterministic():
    """The same settings give exactly the same log."""
    config = GeneratorConfig(target_events=None, number_of_cases=20)
    pd.testing.assert_frame_equal(SyntheticLogGenerator(config).generate(), SyntheticLogGenerator(config).generate())


def test_within_case_keeps_activity_sets_but_changes_order(log):
    """Shuffling within cases changes the order, not which activities a case has."""
    shuffled = ShuffleControls().within_case(log)
    assert activity_sets(shuffled) == activity_sets(log)
    assert not shuffled[ACTIVITY].equals(log[ACTIVITY])


def test_outcome_shuffle_keeps_traces_and_label_distribution(log):
    """Shuffling the outcomes keeps the traces, one label per case and the number of positive cases."""
    shuffled = ShuffleControls().outcomes(log)
    assert shuffled[ACTIVITY].equals(log[ACTIVITY])
    assert shuffled.groupby(CASE)[LABEL].nunique().max() == 1
    assert shuffled.groupby(CASE)[LABEL].first().sum() == log.groupby(CASE)[LABEL].first().sum()
    assert not shuffled[LABEL].equals(log[LABEL])


def test_input_log_is_not_modified(log):
    """The controls work on copies of the log."""
    before = log.copy()
    ShuffleControls().all_controls(log)
    pd.testing.assert_frame_equal(log, before)
