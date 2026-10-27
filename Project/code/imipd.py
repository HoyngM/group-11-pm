"""IMPresseD pattern discovery: interest functions and pattern extension.

Ported from ``IMIPD.py`` and ``tools.py`` of https://github.com/MozhganVD/InteractivePatternDetection
(Vazifehdoostirani et al., 2023); see DIFFERENCES.md. ``auto_impid.py`` uses these functions for the
automatic discovery.

Main ideas:

* A *pattern* is a small ``networkx`` graph whose nodes carry the activity (``value``) and whose
  edges say whether the next event follows directly (``eventually=False``) or later (``True``).
  Written as text: ``a -> b`` (b directly after a) and ``a => b`` (b later, events in between).
* Every pattern is scored on three *interest functions*: outcome interest (how much it tells us
  about the label; higher is better), frequency interest (share of cases with it; higher is
  better) and case distance interest (how different the cases with it are from the others, based
  on case attributes such as age; lower is better).
* ``patient_data`` has one row per case with its outcome, its case attributes and one column per
  activity and per discovered pattern, holding the number of instances.

Traces are totally ordered, so every trace graph is a chain of events, and every extension step
adds exactly one event.
"""

from __future__ import annotations

from collections import Counter
from typing import Sequence

import networkx as nx
import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from tools import DataManager


# --- case data and distances ------------------------------------------------------------------

def create_patient_data(data_manager: DataManager) -> pd.DataFrame:
    """Per case its id, outcome, attributes and activity counts.

    Row ``i`` belongs to ``data_manager.case_ids[i]``. Example of the count columns: a case with
    trace a b a gets a=2, b=1 and 0 for every other activity.
    """
    patient_data = data_manager.case_table.copy()
    # value_counts gives, per case, a Series like {"a": 2, "b": 1}; together they form a table.
    counts = pd.DataFrame([pd.Series(data_manager.traces[case]).value_counts() for case in data_manager.case_ids])
    # Fix the column order and give activities that do not occur in a case a count of 0 instead of NaN.
    counts = counts.reindex(columns=data_manager.activities).fillna(0).astype(int).reset_index(drop=True)
    return pd.concat([patient_data, counts], axis=1)


def calculate_pairwise_case_distance(x_features: pd.DataFrame, num_col: Sequence[str]) -> np.ndarray:
    """Pairwise case distance as a square matrix (entry ``[i, j]``: case i vs case j).

    ``x_features`` holds the case attributes; every column not in ``num_col`` is categorical.
    Distance = ``(m * jaccard + minmax(euclidean)) / (m + 1)`` for ``m`` categorical attributes;
    0 means identical attributes.
    """
    size = len(x_features)
    cat_col = [c for c in x_features.columns if c not in num_col]
    if not cat_col and not num_col:
        return np.zeros((size, size))  # no attributes: all cases are equally similar
    parts, weights = [], []
    if cat_col:
        # Label-encode every attribute (text -> integer code), then the Jaccard
        # distance of the codes: roughly the share of attributes on which two cases differ.
        codes = np.column_stack([LabelEncoder().fit_transform(x_features[c].astype(str)) for c in cat_col])
        parts.append(pdist(codes, "jaccard"))
        weights.append(len(cat_col))  # each categorical attribute counts once
    if num_col:
        # Euclidean distance, rescaled to [0, 1] over all pairs of cases.
        numeric_dist = pdist(x_features[list(num_col)].to_numpy(float), "euclidean").reshape(-1, 1)
        parts.append(MinMaxScaler().fit_transform(numeric_dist).ravel())
        weights.append(1)  # all numeric attributes together count once
    # pdist returns a "condensed" list with one value per pair of cases; combine the parts as a
    # weighted average and turn the list into a symmetric cases x cases matrix.
    combined_dist = sum(w * p for w, p in zip(weights, parts)) / sum(weights)
    return squareform(combined_dist)


# --- interest functions -----------------------------------------------------------------------

def create_pattern_frame(pattern_list: Sequence[str]) -> pd.DataFrame:
    """Empty score table with one row per pattern (column ``patterns``)."""
    return pd.DataFrame({"patterns": list(pattern_list)})


def frequency_measuring_patterns(patterns_data: pd.DataFrame, pattern_list: Sequence[str],
                                 patient_data: pd.DataFrame) -> pd.DataFrame:
    """Frequency interest: the share of cases with at least one instance of the pattern."""
    patterns_data["Frequency_Interest"] = (patient_data[list(pattern_list)] > 0).sum().to_numpy() / len(patient_data)
    return patterns_data


def predictive_measuring_patterns(patterns_data: pd.DataFrame, patient_data: pd.DataFrame,
                                  label_class: str) -> pd.DataFrame:
    """Outcome interest: information gain (mutual information) of the label from the instance counts.

    0 means that the counts of the pattern say nothing about the outcome.
    """
    x = patient_data[list(patterns_data["patterns"])]
    patterns_data["Outcome_Interest"] = mutual_info_classif(x, patient_data[label_class], discrete_features=True)
    return patterns_data


def similarity_measuring_patterns(patterns_data: pd.DataFrame, patient_data: pd.DataFrame,
                                  pairwise_distances_array: np.ndarray) -> pd.DataFrame:
    """Case distance interest: mean distance between the cases with and without the pattern.

    The index of ``patient_data`` holds the row numbers of the cases in the distance matrix.
    """
    distances = []
    for pattern in patterns_data["patterns"]:
        in_pattern_cases = patient_data.index[patient_data[pattern] > 0].to_numpy()
        out_pattern_cases = patient_data.index[patient_data[pattern] == 0].to_numpy()
        if len(in_pattern_cases) == 0 or len(out_pattern_cases) == 0:
            distances.append(np.nan)  # no cases to compare with
        else:
            # np.ix_ selects the block of the matrix with rows "in" and columns "out": every pair of
            # one case with the pattern and one case without it.
            distances.append(float(pairwise_distances_array[np.ix_(in_pattern_cases, out_pattern_cases)].mean()))
    patterns_data["Case_Distance_Interest"] = distances
    return patterns_data


def create_pattern_attributes(patient_data: pd.DataFrame, label_class: str, pattern_list: Sequence[str],
                              pairwise_distances_array: np.ndarray) -> pd.DataFrame:
    """The three interests of every pattern in ``pattern_list``, computed on ``patient_data``.

    In the automatic mode ``patient_data`` holds only the training cases.
    """
    patterns_data = create_pattern_frame(pattern_list)
    patterns_data = predictive_measuring_patterns(patterns_data, patient_data, label_class)
    patterns_data = frequency_measuring_patterns(patterns_data, pattern_list, patient_data)
    patterns_data = similarity_measuring_patterns(patterns_data, patient_data, pairwise_distances_array)
    return patterns_data


# --- trace graphs and patterns ----------------------------------------------------------------

def trace_graph_generator(trace: Sequence[str]) -> nx.DiGraph:
    """Trace graph of a totally ordered trace: a chain of events.

    Example: ["a", "b", "c"] becomes nodes 0, 1, 2 (values a, b, c) with edges 0 -> 1 -> 2.
    Node numbers are positions in the trace, so different events of the same activity stay apart.
    """
    trace_graph = nx.DiGraph()
    for i, act in enumerate(trace):
        trace_graph.add_node(i, value=act)
    for i in range(len(trace) - 1):
        trace_graph.add_edge(i, i + 1, eventually=False)
    return trace_graph


def create_embedded_pattern_in_trace(inside_pattern_nodes: set[int], trace_graph: nx.DiGraph) -> nx.DiGraph:
    """The trace with the pattern's events merged into one node called ``"pattern"``.

    Example: trace 0 -> 1 -> 2 -> 3 with the pattern on nodes {1, 2} becomes 0 -> "pattern" -> 3.
    The neighbours of ``"pattern"`` (here 0 and 3) can extend the pattern directly in the next
    step; the other nodes can extend it eventually.
    """
    embedded_trace_graph = trace_graph.copy()
    embedded_trace_graph.add_node("pattern", value="pattern")
    for node in trace_graph.pred[min(inside_pattern_nodes)]:  # the event just before the pattern
        embedded_trace_graph.add_edge(node, "pattern", eventually=False)
    for node in trace_graph.succ[max(inside_pattern_nodes)]:  # the event just after the pattern
        embedded_trace_graph.add_edge("pattern", node, eventually=False)
    embedded_trace_graph.remove_nodes_from(inside_pattern_nodes)
    return embedded_trace_graph


def chain_order(pattern: nx.DiGraph) -> list:
    """Nodes of a chain pattern from first to last event (a topological sort follows the arrows)."""
    return list(nx.topological_sort(pattern))


def describe_pattern(pattern: nx.DiGraph) -> str:
    """Readable form, e.g. ``a -> b => c`` (``->`` directly, ``=>`` eventually following)."""
    order = chain_order(pattern)
    text = pattern.nodes[order[0]]["value"]
    for left, right in zip(order, order[1:]):  # every pair of neighbouring events in the chain
        text += (" => " if pattern.edges[left, right]["eventually"] else " -> ") + pattern.nodes[right]["value"]
    return text


def has_eventual_relation(pattern: nx.DiGraph) -> bool:
    """True if at least one pair of neighbouring events is related eventually (not directly)."""
    return any(nx.get_edge_attributes(pattern, "eventually").values())


class PatternsDictionary(dict):
    """Discovered patterns: pattern id -> ``{"pattern": graph, "Instances": {...}}``.

    ``Instances`` holds, per instance (one place where the pattern occurs), its ``case`` and its
    embedded trace graph (``emb_trace``), which is needed to extend the pattern later. Besides
    the entries this dictionary keeps ``key_index`` (pattern key -> id) to find a pattern quickly.
    """

    def __init__(self) -> None:
        """Empty dictionary of patterns."""
        super().__init__()
        self.key_index: dict[tuple, str] = {}


def update_pattern_dict(patterns_dictionary: PatternsDictionary, pattern: nx.DiGraph,
                        embedded_trace_graph: nx.DiGraph | None, case: str, core: str,
                        new_patterns_for_core: list[str]) -> None:
    """Add one instance of ``pattern``; a pattern seen for the first time gets the id ``<core>_<number>``.

    New ids are also appended to ``new_patterns_for_core``, the patterns found for the current core.
    """
    # Two chain patterns are equal (isomorphic) exactly when their activities and their relation
    # types match in order, so this key replaces a graph isomorphism test.
    # Example: "a -> b" has key (("a", "b"), (False,)); "a => b" has (("a", "b"), (True,)).
    order = chain_order(pattern)
    key = (tuple(pattern.nodes[n]["value"] for n in order),
           tuple(pattern.edges[a, b]["eventually"] for a, b in zip(order, order[1:])))
    if key not in patterns_dictionary.key_index:
        # One more than the highest number used for this core,
        # e.g. "b_1", "b_2", ... for patterns grown from activity "b".
        pattern_ids = max((int(s.split("_")[-1]) for s in new_patterns_for_core), default=0)
        new_pattern_id = f"{core}_{pattern_ids + 1}"
        patterns_dictionary.key_index[key] = new_pattern_id
        patterns_dictionary[new_pattern_id] = {"pattern": pattern, "Instances": {"case": [], "emb_trace": []}}
        new_patterns_for_core.append(new_pattern_id)
    # Record this instance under the (new or existing) pattern.
    instances = patterns_dictionary[patterns_dictionary.key_index[key]]["Instances"]
    instances["case"].append(case)
    instances["emb_trace"].append(embedded_trace_graph)


def eventual_pattern(trace_graph: nx.DiGraph, first: int, second: int) -> nx.DiGraph:
    """Pattern of two events of ``trace_graph`` in which ``second`` eventually follows ``first``."""
    pattern = nx.DiGraph()
    pattern.add_node(first, value=trace_graph.nodes[first]["value"])
    pattern.add_node(second, value=trace_graph.nodes[second]["value"])
    pattern.add_edge(first, second, eventually=True)
    return pattern


def pattern_extension(case: str, trace_graph: nx.DiGraph, core_activity: str,
                      patterns_dictionary: PatternsDictionary, max_gap_between_events: int,
                      new_patterns_for_core: list[str]) -> None:
    """Every pattern of two events around each occurrence of the core.

    For the trace a b c d and core activity c this adds: b -> c (directly preceding),
    c -> d (directly following) and a => c (eventually preceding).
    """
    for n, value in trace_graph.nodes(data="value"):  # every event of the trace
        if value != core_activity:
            continue
        # The events directly before and after this occurrence (at most one each in a chain).
        in_pattern_nodes, out_pattern_nodes = set(trace_graph.pred[n]), set(trace_graph.succ[n])
        for neighbours in (in_pattern_nodes, out_pattern_nodes):  # directly preceding, directly following
            if neighbours:
                nodes = neighbours | {n}
                update_pattern_dict(patterns_dictionary, trace_graph.subgraph(nodes).copy(),
                                    create_embedded_pattern_in_trace(nodes, trace_graph),
                                    case, core_activity, new_patterns_for_core)
        # Events that are not direct neighbours can be related eventually (within the gap).
        eventual_relations_nodes = set(trace_graph.nodes) - in_pattern_nodes - out_pattern_nodes - {n}
        if out_pattern_nodes:  # eventually following: events after the direct successor
            for node in sorted(o for o in eventual_relations_nodes
                               if max(out_pattern_nodes) < o <= max(out_pattern_nodes) + max_gap_between_events):
                update_pattern_dict(patterns_dictionary, eventual_pattern(trace_graph, n, node), None,
                                    case, core_activity, new_patterns_for_core)
        if in_pattern_nodes:  # eventually preceding: events before the direct predecessor
            for node in sorted(o for o in eventual_relations_nodes
                               if min(in_pattern_nodes) - max_gap_between_events <= o < min(in_pattern_nodes)):
                update_pattern_dict(patterns_dictionary, eventual_pattern(trace_graph, node, n), None,
                                    case, core_activity, new_patterns_for_core)


def single_pattern_extender(all_extension_list: dict, chosen_pattern_id: str, patient_data: pd.DataFrame,
                            event_log_graphs: dict[str, nx.DiGraph], max_gap_between_events: int,
                            case_ids: Sequence[str]) -> tuple[dict, PatternsDictionary, pd.DataFrame]:
    """Add one event before or after every instance of a pattern.

    For the trace a b c d e and an instance of b -> c this adds: a -> b -> c, b -> c -> d and
    b -> c => e. Returns the updated list of all patterns, the new patterns of this step and
    ``patient_data`` with one count column per new pattern.
    """
    extended_patterns_at_stage = PatternsDictionary()
    new_patterns_for_core: list[str] = []
    instances = all_extension_list[chosen_pattern_id]["Instances"]
    for case, embedded_trace_graph in zip(instances["case"], instances["emb_trace"]):
        # Recover the events of this instance: the trace nodes that the embedding merged away.
        trace_graph = event_log_graphs[case]
        inside_pattern_nodes = set(trace_graph.nodes) - set(embedded_trace_graph.nodes)
        chosen_pattern = trace_graph.subgraph(inside_pattern_nodes).copy()
        starting_node, ending_node = min(inside_pattern_nodes), max(inside_pattern_nodes)
        # The direct neighbours of the instance, read from the embedding.
        in_pattern_nodes = set(embedded_trace_graph.pred["pattern"])
        out_pattern_nodes = set(embedded_trace_graph.succ["pattern"])

        # Directly preceding and directly following extensions: (new event, new edge).
        direct = ([(n, (n, starting_node)) for n in in_pattern_nodes]
                  + [(n, (ending_node, n)) for n in out_pattern_nodes])
        for extra, edge in direct:
            extended_pattern = chosen_pattern.copy()
            extended_pattern.add_node(extra, value=trace_graph.nodes[extra]["value"])
            extended_pattern.add_edge(*edge, eventually=False)
            update_pattern_dict(extended_patterns_at_stage, extended_pattern,
                                create_embedded_pattern_in_trace(set(extended_pattern.nodes), trace_graph),
                                case, chosen_pattern_id, new_patterns_for_core)

        # Eventually following and eventually preceding extensions (within the gap).
        eventual_relations_nodes = set(embedded_trace_graph.nodes) - in_pattern_nodes - out_pattern_nodes - {"pattern"}
        eventual = []
        if out_pattern_nodes:
            eventual += [(o, (ending_node, o)) for o in sorted(eventual_relations_nodes)
                         if max(out_pattern_nodes) < o <= max(out_pattern_nodes) + max_gap_between_events]
        if in_pattern_nodes:
            eventual += [(o, (o, starting_node)) for o in sorted(eventual_relations_nodes)
                         if min(in_pattern_nodes) - max_gap_between_events <= o < min(in_pattern_nodes)]
        for extra, edge in eventual:
            extended_pattern = chosen_pattern.copy()
            extended_pattern.add_node(extra, value=trace_graph.nodes[extra]["value"])
            extended_pattern.add_edge(*edge, eventually=True)
            # No embedding: patterns with an eventual relation are never extended again.
            update_pattern_dict(extended_patterns_at_stage, extended_pattern, None,
                                case, chosen_pattern_id, new_patterns_for_core)

    patient_data = add_pattern_counts(patient_data, extended_patterns_at_stage, new_patterns_for_core, case_ids)
    all_extension_list.update(extended_patterns_at_stage)
    return all_extension_list, extended_patterns_at_stage, patient_data


def add_pattern_counts(patient_data: pd.DataFrame, patterns_dictionary: dict, pattern_ids: Sequence[str],
                       case_ids: Sequence[str]) -> pd.DataFrame:
    """Add one column per new pattern to ``patient_data``: its number of instances in every case."""
    position = {case: row for row, case in enumerate(case_ids)}  # case id -> row
    columns = {}
    for pattern_id in pattern_ids:
        column = np.zeros(len(case_ids), dtype=int)
        # Counter counts how often each case appears in the instance list, e.g. {"c1": 2, "c7": 1}.
        for case, count in Counter(patterns_dictionary[pattern_id]["Instances"]["case"]).items():
            column[position[case]] = count
        columns[pattern_id] = column
    if not columns:
        return patient_data
    # Add all new columns at once (side by side).
    return pd.concat([patient_data, pd.DataFrame(columns)], axis=1)
