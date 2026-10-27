"""The itemset selection strategies, the IMPresseD functions and the simulated logs."""

import networkx as nx
import numpy as np
import pandas as pd

from conftest import make_log
from extensions.itemset_selection import AprioriSelector, ImpressedSelector
from extensions.simulation import SimulatedLogGenerator, SimulationConfig
from imipd import PatternsDictionary, calculate_pairwise_case_distance, update_pattern_dict


def test_apriori_keeps_all_single_activities_and_the_top_k_larger_itemsets():
    """{a, b} is in 3 of 4 cases, so it is the most frequent itemset; all 4 activities stay as singles."""
    data_manager = make_log([["a", "b", "c"], ["a", "b"], ["a", "b", "d"], ["c", "d"]], [1, 0, 1, 0])
    result = AprioriSelector(min_support=0.5, top_k=1).select(data_manager, data_manager.case_ids)
    assert [i for i in result.itemsets if len(i) > 1] == [frozenset({"a", "b"})]
    assert len([i for i in result.itemsets if len(i) == 1]) == 4


def test_impressed_selects_the_activity_that_predicts_the_outcome():
    """'treat' occurs exactly in the positive cases, so it has the highest outcome interest."""
    rng = np.random.default_rng(0)
    labels = [int(rng.random() < 0.5) for _ in range(200)]
    traces = [["start", *(["treat"] if label else []), *rng.choice(["n1", "n2", "n3"], size=3), "end"]
              for label in labels]
    data_manager = make_log(traces, labels)
    result = ImpressedSelector(max_extension_step=1).select(data_manager, data_manager.case_ids)
    singles = result.details[result.details["length"] == 1].set_index("pattern")
    assert frozenset({"treat"}) in result.itemsets
    assert singles["Outcome_Interest"].idxmax() == "treat"


def test_impressed_respects_the_maximum_extension_step():
    """With the direct context rule off, step k gives patterns of at most k + 1 activities."""
    rng = np.random.default_rng(1)
    traces = [["a", "b", "c", str(rng.choice(["d", "e"]))] for _ in range(100)]
    data_manager = make_log(traces, [int(t[-1] == "d") for t in traces])
    for step in (0, 1, 2):
        result = ImpressedSelector(max_extension_step=step).select(data_manager, data_manager.case_ids)
        assert max(len(i) for i in result.itemsets) <= step + 1
        assert result.details["length"].max() <= step + 1


def test_update_pattern_dict_separates_direct_and_eventual_relations():
    """'a -> b' and 'a => b' are different patterns; a repeated instance is added to the first one."""
    def pair(eventually):
        """Pattern 'a -> b' (eventually=False) or 'a => b' (eventually=True)."""
        graph = nx.DiGraph()
        graph.add_node(0, value="a")
        graph.add_node(2, value="b")
        graph.add_edge(0, 2, eventually=eventually)
        return graph

    patterns_dictionary, new_patterns_for_core = PatternsDictionary(), []
    for eventually in (False, True, False):
        update_pattern_dict(patterns_dictionary, pair(eventually), None, "case", "a", new_patterns_for_core)
    assert new_patterns_for_core == ["a_1", "a_2"]
    assert patterns_dictionary["a_1"]["Instances"]["case"] == ["case", "case"]


def test_case_distance_is_symmetric_with_zero_diagonal():
    """The distance matrix is a proper distance: symmetric and 0 between a case and itself."""
    x_features = pd.DataFrame({"Age": [20, 40, 60], "Sex": ["m", "f", "m"]})
    matrix = calculate_pairwise_case_distance(x_features, ["Age"])
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0.0)


def test_simulation_is_reproducible_and_plants_a_location_effect():
    """Same seed, same log; an early x gives a much higher share of positive cases than a late x."""
    config = SimulationConfig(number_of_cases=2000, seed=7)
    first, second = SimulatedLogGenerator(config).generate(), SimulatedLogGenerator(config).generate()
    pd.testing.assert_frame_equal(first, second)
    x_events = first[first["concept:name"] == "x"]
    early, late = x_events[x_events["event_nr"] <= 6], x_events[x_events["event_nr"] > 6]
    assert early["label"].mean() - late["label"].mean() > 0.4


def test_simulation_presence_of_x_says_nothing_about_the_outcome():
    """x only matters through its location: cases with and without x have about the same positive rate."""
    log = SimulatedLogGenerator(SimulationConfig(number_of_cases=4000, seed=3)).generate()
    per_case = log.groupby("case:concept:name").agg(
        label=("label", "first"), has_x=("concept:name", lambda s: "x" in set(s)))
    gap = per_case[per_case.has_x]["label"].mean() - per_case[~per_case.has_x]["label"].mean()
    assert abs(gap) < 0.1
