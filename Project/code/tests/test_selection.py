"""Unit tests for the itemset selection strategies."""

import numpy as np
import pandas as pd
import pytest

from itemset_selection import AprioriItemsetSelector, EventLog, ImpressedItemsetSelector
from itemset_selection.case_distance import CaseDistance
from itemset_selection.pareto import non_dominated
from itemset_selection.pattern import DIRECT, EVENTUAL, ExtensionRule, Pattern, extend_instance
from itemset_selection.simulation import SimulatedLogGenerator, SimulationConfig


def make_log(traces, labels):
    """Small log from lists of activity names."""
    names = sorted({a for trace in traces for a in trace})
    ids = {name: i for i, name in enumerate(names)}
    attributes = pd.DataFrame({"Age": range(len(traces))})
    return EventLog(
        range(len(traces)), names, [[ids[a] for a in t] for t in traces], labels, attributes
    )


def test_pareto_keeps_only_non_dominated_rows():
    values = np.array([[1.0, 1.0], [2.0, 2.0], [2.0, 1.0], [0.5, 3.0]])
    mask = non_dominated(values, maximise=[True, True])
    assert mask.tolist() == [False, True, False, True]


def test_pareto_respects_minimisation_and_keeps_one_of_duplicates():
    # Rows 0 and 1 are identical; row 2 is worse on the first but better on the second objective.
    values = np.array([[1.0, 5.0], [1.0, 5.0], [0.5, 4.0], [0.5, 6.0]])
    mask = non_dominated(values, maximise=[True, False])
    assert mask.tolist() == [True, False, True, False]


def test_direct_and_eventual_instances():
    trace = [0, 1, 2, 1, 2]  # A B C B C
    direct = Pattern((1, 2), (DIRECT,))
    eventual = Pattern((0, 2), (EVENTUAL,))
    assert direct.find_instances(trace) == [(1, 2), (3, 4)]
    assert eventual.find_instances(trace) == [(0, 2), (0, 4)]


def test_extension_rules_generate_expected_patterns():
    trace = [0, 1, 2, 3]
    pattern = Pattern((1,))
    produced = {
        (p.activities, p.relations)
        for p, _ in extend_instance(pattern, (1,), trace, list(ExtensionRule))
    }
    assert produced == {
        ((1, 2), (DIRECT,)),
        ((1, 3), (EVENTUAL,)),
        ((0, 1), (DIRECT,)),
    }


def test_itemset_ignores_order_and_relations():
    assert Pattern((2, 1), (EVENTUAL,)).itemset == frozenset({1, 2})


def test_apriori_selects_most_frequent_multi_activity_itemsets():
    traces = [["A", "B", "C"], ["A", "B"], ["A", "B", "D"], ["C", "D"]]
    log = make_log(traces, [1, 0, 1, 0])
    result = AprioriItemsetSelector(min_support=0.5, top_k=1, max_length=3).select(log)
    multi = [i for i in result.itemsets if len(i) > 1]
    assert multi == [frozenset({"A", "B"})]
    assert len([i for i in result.itemsets if len(i) == 1]) == 4  # all singletons kept


def test_apriori_support_is_a_fraction_of_cases():
    log = make_log([["A", "B"], ["A"], ["A"], ["B"]], [1, 0, 1, 0])
    result = AprioriItemsetSelector(min_support=0.5, top_k=5, singletons="frequent").select(log)
    assert {frozenset({"A"}), frozenset({"B"})} == set(result.itemsets)


def test_impressed_selects_the_activity_that_predicts_the_outcome():
    rng = np.random.default_rng(0)
    traces, labels = [], []
    for _ in range(200):
        positive = int(rng.random() < 0.5)
        trace = ["Start"] + (["Treat"] if positive else []) + list(
            rng.choice(["N1", "N2", "N3"], size=3)
        ) + ["End"]
        traces.append(trace)
        labels.append(positive)
    log = make_log(traces, labels)
    result = ImpressedItemsetSelector(
        max_pattern_length=2, numeric_attributes=["Age"]
    ).select(log)
    assert frozenset({"Treat"}) in result.itemsets
    # Outcome interest of the predictive activity is clearly the highest of the singles.
    singles = result.details[result.details["length"] == 1].set_index("pattern")
    assert singles["outcome_interest"].idxmax() == "Treat"


def test_impressed_returns_itemsets_up_to_the_maximum_length():
    rng = np.random.default_rng(1)
    traces = [["A", "B", "C", str(rng.choice(["D", "E"]))] for _ in range(100)]
    labels = [int(t[-1] == "D") for t in traces]
    log = make_log(traces, labels)
    result = ImpressedItemsetSelector(max_pattern_length=3, numeric_attributes=["Age"]).select(log)
    assert max(len(i) for i in result.itemsets) <= 3
    assert max(result.details["length"]) <= 3


def test_case_distance_is_symmetric_with_zero_diagonal():
    frame = pd.DataFrame({"Age": [20, 40, 60], "Sex": ["m", "f", "m"]})
    matrix = CaseDistance(["Age"], ["Sex"]).pairwise(frame)
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert matrix[0, 2] < matrix[0, 1]  # same sex, larger age gap vs. other sex


def test_simulation_is_reproducible_and_plants_location_effect():
    config = SimulationConfig(number_of_cases=2000, seed=7)
    first = SimulatedLogGenerator(config).generate()
    second = SimulatedLogGenerator(config).generate()
    pd.testing.assert_frame_equal(first, second)

    # Among cases containing X, an early X gives a much higher positive rate than a late X.
    rows = first[first["concept:name"] == "X"].merge(
        first.groupby("case:concept:name").size().rename("length"), on="case:concept:name"
    )
    early = rows[rows["event_nr"] <= 6]
    late = rows[rows["event_nr"] > 6]
    assert early["label"].mean() - late["label"].mean() > 0.4


def test_simulation_existence_of_x_is_uninformative():
    log = SimulatedLogGenerator(SimulationConfig(number_of_cases=4000, seed=3)).generate()
    per_case = log.groupby("case:concept:name").agg(
        label=("label", "first"), has_x=("concept:name", lambda s: "X" in set(s))
    )
    gap = per_case[per_case.has_x]["label"].mean() - per_case[~per_case.has_x]["label"].mean()
    assert abs(gap) < 0.1


def test_event_log_stratified_split_keeps_label_ratio():
    log = make_log([["A"]] * 100, [1] * 30 + [0] * 70)
    train, test = log.stratified_split(0.2, seed=1)
    assert len(train) == 80 and len(test) == 20
    assert train.labels.mean() == pytest.approx(0.3, abs=0.02)
