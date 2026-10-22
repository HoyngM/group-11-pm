"""End-to-end: the steps of main.py on the small synthetic log, with fast settings."""

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline import Pipeline

CONFIG = Path(__file__).resolve().parents[1] / "config.yaml"


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory) -> Pipeline:
    pipeline = Pipeline.from_file(CONFIG)
    pipeline.output_directory = tmp_path_factory.mktemp("results")
    pipeline.config["importance"].update(folds=2, repetitions=2, evaluate_on=["test"])
    pipeline.config["comparison"]["datasets"] = ["synthetic"]
    pipeline.run(steps=["check", "select", "importance", "compare"], only=["synthetic"])
    return pipeline


def test_every_step_writes_its_output(pipeline):
    output = pipeline.output_directory
    assert (output / "synthetic_checks.csv").exists()
    assert (output / "selection" / "summary.csv").exists()
    for strategy in ("apriori", "impressed"):
        assert (output / "importance" / "synthetic" / "evaluated_on_test" / f"{strategy}_summary.csv").exists()
    assert (output / "comparison" / "comparison_metrics.csv").exists()
    assert (output / "comparison" / "comparison_test.png").exists()


def test_importance_uses_exactly_the_selected_itemsets(pipeline):
    output = pipeline.output_directory
    for strategy in ("apriori", "impressed"):
        with open(output / "selection" / "synthetic" / f"{strategy}_itemsets.json", encoding="utf-8") as handle:
            selected = {", ".join(sorted(itemset)) for itemset in json.load(handle)}
        summary = pd.read_csv(output / "importance" / "synthetic" / "evaluated_on_test" / f"{strategy}_summary.csv")
        assert set(summary.itemset) == selected


def test_strategies_without_importance_are_only_selected(pipeline):
    output = pipeline.output_directory
    assert (output / "selection" / "synthetic" / "apriori_support0.2_top25_itemsets.json").exists()
    assert not (output / "importance" / "synthetic" / "evaluated_on_test"
                / "apriori_support0.2_top25_summary.csv").exists()


def test_importance_without_selection_gives_a_clear_error(tmp_path):
    pipeline = Pipeline.from_file(CONFIG)
    pipeline.output_directory = tmp_path
    with pytest.raises(FileNotFoundError, match="select"):
        pipeline.importance(only=["synthetic"])
