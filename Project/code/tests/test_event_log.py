"""Loading, cleaning, splitting and encoding of the event log (``DataManager`` in tools.py)."""

import pytest

from conftest import make_log


def test_activity_names_are_cleaned():
    """Names are lower-cased and stripped of spaces, '-' and '_'."""
    data_manager = make_log([["Start Case", "check-in", "Close_Case"]] * 2, [1, 0])
    assert data_manager.activities == ["checkin", "closecase", "startcase"]


def test_cases_with_rare_activities_are_removed():
    """A case with an activity that occurs fewer than twice in the log is dropped."""
    data_manager = make_log([["a", "b"], ["a", "b"], ["a", "rare"]], [1, 0, 1], frq_threshold=2)
    assert len(data_manager) == 2 and "rare" not in data_manager.activities


def test_allowed_locations_are_one_based_observed_positions():
    """The locations from pm4py are converted to 1-based event numbers, like ``event_nr``."""
    data_manager = make_log([["a", "b", "c"], ["b", "a", "c"]], [1, 0])
    assert data_manager.allowed_locations == {"a": [1, 2], "b": [1, 2], "c": [3]}


def test_fixed_location_activities_are_found_from_the_start_and_from_the_end():
    """'start' is always first and 'end' always last (traces differ in length); 'a' and 'b' move."""
    data_manager = make_log([["start", "a", "b", "end"], ["start", "b", "a", "a", "end"]], [1, 0])
    assert data_manager.fixed_location_activities == ["end", "start"]


def test_index_encoding_has_exactly_one_activity_per_position():
    """Every position of every trace (including padding) has exactly one 1 in the one-hot encoding."""
    data_manager = make_log([["a", "b", "c"], ["b", "a"]], [1, 0])
    encoded = data_manager.index_encoding([data_manager.traces[case] for case in data_manager.case_ids])
    assert (encoded.reshape(len(data_manager), data_manager.max_length, -1).sum(axis=2) == 1).all()


def test_split_test_train_keeps_the_label_ratio():
    """The training part has 80% of the cases and the same share of positive labels."""
    data_manager = make_log([["a"]] * 100, [1] * 30 + [0] * 70)
    train = data_manager.split_test_train(0.2, seed=1)
    assert len(train) == 80
    labels = data_manager.case_table.set_index("case:concept:name").loc[train, "label"]
    assert labels.mean() == pytest.approx(0.3, abs=0.02)


def test_cross_split_test_train_uses_every_case_once_as_test_case():
    """Over the folds, every case is in exactly one test part."""
    data_manager = make_log([["a"]] * 50, [1] * 20 + [0] * 30)
    _, test_list = data_manager.cross_split_test_train(5, seed=0)
    assert sorted(row for rows in test_list.values() for row in rows) == list(range(50))
