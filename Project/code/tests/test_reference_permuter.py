"""Properties of the ported reference permutation."""

import numpy as np

from location_permutation import ReferencePermuter

ALLOWED = {"a": [1, 2, 3, 4], "b": [2, 3, 4, 5], "c": [1, 2, 3, 4, 5], "d": [3, 4, 5, 6]}
TRACE = ["a", "b", "c", "d", "c", "a"]


def permuter(seed=0, exact=True):
    return ReferencePermuter(ALLOWED, seed=seed, exact_tracking=exact)


def test_find_itemset_indexes_returns_non_overlapping_occurrences():
    found = ReferencePermuter.find_itemset_indexes(["a", "b", "a", "b"], frozenset({"a", "b"}))
    assert found == [(0, 1), (2, 3)]


def test_activity_shuffle_keeps_the_multiset_of_activities():
    for seed in range(20):
        assert sorted(permuter(seed).shuffle_activity(TRACE, "c")) == sorted(TRACE)


def test_itemset_shuffle_keeps_the_multiset_and_the_relative_order():
    trace = ["a", "b", "c", "d", "c"]  # a and b occur once, so their order is unambiguous
    for seed in range(50):
        result = permuter(seed).shuffle_itemset(trace, frozenset({"a", "b"}))
        assert sorted(result) == sorted(trace)
        assert result.index("a") < result.index("b")


def test_shuffle_moves_something_for_some_seeds():
    assert any(permuter(seed).shuffle_itemset(TRACE, frozenset({"a", "b"})) != TRACE for seed in range(20))


def test_input_trace_is_not_modified():
    trace = list(TRACE)
    permuter().shuffle_itemset(trace, frozenset({"a", "b"}))
    permuter().shuffle_activity(trace, "c")
    assert trace == TRACE


def test_unconstrained_shuffle_uses_any_position():
    positions = {permuter(seed).shuffle_activity(["x"] * 5 + ["y"], "y", constrain=False).index("y")
                 for seed in range(100)}
    assert positions == set(range(6))


def test_literal_port_keeps_the_multiset_but_not_always_the_order():
    """The reference bookkeeping is off by one: the order of the itemset is sometimes broken."""
    trace = ["a", "b", "c", "d", "c"]
    results = [permuter(seed, exact=False).shuffle_itemset(trace, frozenset({"a", "b"})) for seed in range(200)]
    assert all(sorted(r) == sorted(trace) for r in results)
    assert any(r.index("a") > r.index("b") for r in results)
