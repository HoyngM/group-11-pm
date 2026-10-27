"""Properties of the location permutation (``shuffle_sequence`` and the activity shuffle)."""

from conftest import make_permuter
from tools import DataManager

# Observed (1-based) locations per activity, and a trace to permute.
ALLOWED = {"a": [1, 2, 3, 4], "b": [2, 3, 4, 5], "c": [1, 2, 3, 4, 5], "d": [3, 4, 5, 6]}
TRACE = ["a", "b", "c", "d", "c", "a"]


def test_find_itemset_indexes_returns_non_overlapping_occurrences():
    """In 'a b a b' the itemset {a, b} occurs twice without sharing an event."""
    found = DataManager.find_itemset_indexes(["a", "b", "a", "b"], frozenset({"a", "b"}))
    assert found == [(0, 1), (2, 3)]


def test_shuffles_keep_the_multiset_of_activities():
    """A permutation only moves events; it never adds or removes one."""
    for seed in range(20):
        data_manager = make_permuter(ALLOWED, seed)
        assert sorted(data_manager._shuffle_activity(TRACE, "c")) == sorted(TRACE)
        assert sorted(data_manager.shuffle_sequence(TRACE, frozenset({"a", "b"}))) == sorted(TRACE)


def test_shuffle_moves_something_for_some_seeds():
    """For at least one seed the itemset really changes location."""
    assert any(make_permuter(ALLOWED, s).shuffle_sequence(TRACE, frozenset({"a", "b"})) != TRACE for s in range(20))


def test_input_trace_is_not_modified():
    """The shuffles return a new trace and leave their input alone."""
    trace = list(TRACE)
    make_permuter(ALLOWED).shuffle_sequence(trace, frozenset({"a", "b"}))
    make_permuter(ALLOWED)._shuffle_activity(trace, "c")
    assert trace == TRACE


def test_unconstrained_shuffle_uses_any_position():
    """Without the feasibility constraint every position of the trace can be chosen."""
    positions = {make_permuter(ALLOWED, s)._shuffle_activity(["x"] * 5 + ["y"], "y", constrain=False).index("y")
                 for s in range(100)}
    assert positions == set(range(6))


def test_itemset_shuffle_keeps_the_order_of_the_itemset():
    """After moving {a, b}, a still comes before b, for every seed."""
    trace = ["a", "b", "c", "d", "c"]  # a and b occur once, so their order is unambiguous
    results = [make_permuter(ALLOWED, s).shuffle_sequence(trace, frozenset({"a", "b"})) for s in range(200)]
    assert all(r.index("a") < r.index("b") for r in results)
    assert any(r != trace for r in results)  # and the itemset really moves
