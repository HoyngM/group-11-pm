"""Automatic IMPresseD pattern discovery.

Ported from ``AutoStepWise_PPD`` in ``Auto_IMPID.py`` of
https://github.com/MozhganVD/InteractivePatternDetection; see DIFFERENCES.md. How it works, in short:

1. Score every single activity on the three interest functions (see ``imipd.py``).
2. Keep the *Pareto front*: the activities for which no other activity is at least as good on
   all three interests and strictly better on one. No thresholds or weights are needed.
3. Extend each activity on the front by one neighbouring event (``pattern_extension``), score the
   new 2-event patterns and keep their Pareto front.
4. Extend those patterns again (``single_pattern_extender``), up to ``max_extension_step`` steps.

It returns the patterns on the fronts, from which ``extensions/itemset_selection.py`` takes the
activity sets.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from paretoset import paretoset

from imipd import (PatternsDictionary, add_pattern_counts, calculate_pairwise_case_distance,
                   create_pattern_attributes, create_patient_data, describe_pattern, has_eventual_relation,
                   pattern_extension, single_pattern_extender, trace_graph_generator)
from tools import DataManager

PARETO_FEATURES = ["Outcome_Interest", "Frequency_Interest", "Case_Distance_Interest"]
PARETO_SENSE = ["max", "max", "min"]  # the Pareto directions of the three interests


def auto_stepwise_ppd(max_extension_step: int, max_gap_between_events: int, data_manager: DataManager,
                      train_ids: Sequence[str]) -> pd.DataFrame:
    """Discover patterns step by step; return the patterns on the Pareto front of every step.

    ``max_extension_step``: 1 gives patterns up to length 2, 2 up to length 3, and 0 only single
    activities. Instances are counted in all cases, but the
    interests and so the Pareto fronts only use the training cases ``train_ids``.

    Returns one row per selected pattern: ``pattern`` (e.g. ``a -> b``), ``length``, ``itemset``
    (e.g. ``a + b``) and the three interests.
    """
    case_ids = data_manager.case_ids
    patient_data = create_patient_data(data_manager)
    attributes = data_manager.numeric_attributes + data_manager.categorical_attributes
    pairwise_distances_array = calculate_pairwise_case_distance(data_manager.case_table[attributes],
                                                                data_manager.numeric_attributes)
    # Row numbers (positions in case_ids) of the training cases.
    train_rows = np.flatnonzero(np.isin(case_ids, list(train_ids)))
    event_log_graphs = {case: trace_graph_generator(data_manager.traces[case]) for case in case_ids}

    def pareto_front(data: pd.DataFrame, pattern_list: Sequence[str]) -> pd.DataFrame:
        """Interests of the patterns on the training cases, keeping only the Pareto front."""
        pattern_attributes = create_pattern_attributes(data.iloc[train_rows], data_manager.outcome,
                                                       pattern_list, pairwise_distances_array)
        # paretoset returns True for every pattern that no other pattern beats on all three.
        mask = paretoset(pattern_attributes[PARETO_FEATURES], sense=PARETO_SENSE)
        return pattern_attributes[mask].set_index("patterns")

    # Step 1: single activities, in order of first appearance.
    # dict.fromkeys keeps the first occurrence of every name and removes duplicates.
    activities = list(dict.fromkeys(a for case in case_ids for a in data_manager.traces[case]))
    paretoset_patterns = pareto_front(patient_data, activities)
    selected = [paretoset_patterns.assign(pattern=paretoset_patterns.index, length=1,
                                          itemset=paretoset_patterns.index)]

    # Step 2: extend every activity on the front by one event.
    # This gives patterns of length 2, e.g. "a -> b", "b -> a", "a => c".
    extended_patterns_at_stage, new_patterns = PatternsDictionary(), []
    for core_activity in paretoset_patterns.index if max_extension_step >= 1 else []:
        new_patterns_for_core: list[str] = []
        for case in case_ids:
            if core_activity in data_manager.traces[case]:
                pattern_extension(case, event_log_graphs[case], core_activity, extended_patterns_at_stage,
                                  max_gap_between_events, new_patterns_for_core)
        # Counted per core, right after its extension.
        patient_data = add_pattern_counts(patient_data, extended_patterns_at_stage, new_patterns_for_core, case_ids)
        new_patterns += new_patterns_for_core
    all_extended_patterns_dict = dict(extended_patterns_at_stage)  # every pattern discovered so far, by id

    # Further steps: score the newest patterns, keep their front and extend that front again.
    # With max_extension_step 2 this loop runs for length 2 (score the 2-event patterns and extend
    # them) and for length 3 (score and stop).
    for length in range(2, max_extension_step + 2):
        if not new_patterns:
            break  # nothing could be extended any more
        paretoset_patterns = pareto_front(patient_data, new_patterns)
        selected.append(paretoset_patterns.assign(
            pattern=[describe_pattern(all_extended_patterns_dict[p]["pattern"]) for p in paretoset_patterns.index],
            length=length,
            itemset=[" + ".join(sorted({v for _, v in all_extended_patterns_dict[p]["pattern"].nodes(data="value")}))
                     for p in paretoset_patterns.index]))
        if length == max_extension_step + 1:
            break  # the longest patterns are scored; do not extend them further
        new_patterns = []
        for chosen_pattern_id in paretoset_patterns.index:
            if has_eventual_relation(all_extended_patterns_dict[chosen_pattern_id]["pattern"]):
                continue  # patterns with an eventual relation are not extended
            all_extended_patterns_dict, extended_patterns_at_stage, patient_data = single_pattern_extender(
                all_extended_patterns_dict, chosen_pattern_id, patient_data, event_log_graphs,
                max_gap_between_events, case_ids)
            new_patterns += list(extended_patterns_at_stage)

    # The union of the fronts of all steps.
    details = pd.concat(selected).reset_index(drop=True)
    return details[["pattern", "length", "itemset", *PARETO_FEATURES]]
