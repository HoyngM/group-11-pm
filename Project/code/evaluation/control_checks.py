"""Check that the planted location effect of B exists in the synthetic log and not in the controls.

This check looks at the data only, without any model: it confirms that the ground truth we
planted is really there before we ask the location permutation importance to find it.

For every log, the relative position of the first ``B`` in a case (0 = start, 1 = end) is
correlated with the case outcome (Spearman). Expected result: a strong correlation for
``synthetic`` and a correlation close to zero (large p-value) for the three shuffled controls.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

from synthetic_data.generator import ACTIVITY, CASE, KEY_ACTIVITY, LABEL, ORDER

LOG_NAMES = ["synthetic", "shuffled_within_case", "shuffled_outcomes", "shuffled_across_traces"]


def relative_position_of_key_activity(log: pd.DataFrame) -> pd.DataFrame:
    """Per case that contains ``B``: relative position of its first ``B`` and the case label."""
    log = log.sort_values([CASE, ORDER])
    last = log.groupby(CASE)[ORDER].transform("max")
    log = log.assign(relative_position=(log[ORDER] - 1) / (last - 1))
    first_key_event = log[log[ACTIVITY] == KEY_ACTIVITY].groupby(CASE).first()
    return first_key_event[["relative_position", LABEL]]


def summarise(name: str, log: pd.DataFrame) -> dict:
    """One row of the check: how many cases contain ``B`` and how strongly its position predicts the label."""
    cases = relative_position_of_key_activity(log)
    correlation, p_value = spearmanr(cases["relative_position"], cases[LABEL])
    return {
        "log": name,
        "cases_with_B": f"{len(cases)}/{log[CASE].nunique()}",
        "spearman_position_vs_label": round(correlation, 3),
        "p_value": f"{p_value:.2g}",
    }


def check_controls(data_directory: Path) -> pd.DataFrame:
    """Run the check on the synthetic log and its controls in ``data_directory``."""
    rows = [summarise(name, pd.read_csv(data_directory / f"{name}.csv")) for name in LOG_NAMES]
    return pd.DataFrame(rows)
