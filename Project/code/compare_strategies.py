"""Compares the location importance of the Apriori and the IMPresseD itemsets.

This answers the main question of the assignment: does the importance of activities and
their locations change when IMPresseD instead of Apriori chooses the itemsets?

Run it after ``run_location_importance.py`` has written its results for both evaluation
settings (``--evaluate-on train`` and ``--evaluate-on test``). It only reads those results,
so it takes a few seconds. For every dataset and evaluation setting it reports how many
itemsets both strategies selected, how much their most important itemsets and activities
overlap, and the Spearman rank correlation of the importance of the itemsets that both
selected. It writes ``comparison_metrics.csv``, one top-k table per dataset and one figure
per evaluation setting to ``results/importance/comparison/``. The metrics are explained in
``location_permutation/comparison.py``.

Usage:
    python compare_strategies.py                                   # BPIC11 f1-f3, train and test
    python compare_strategies.py --evaluate-on test --top-k 5      # other settings
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from location_permutation.comparison import ComparisonReport

BASE = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, default=BASE / "importance_config.yaml",
                        help="config of run_location_importance.py (for the results folder)")
    parser.add_argument("--datasets", nargs="+", default=["BPIC11_f1", "BPIC11_f2", "BPIC11_f3"])
    parser.add_argument("--evaluate-on", nargs="+", choices=["train", "test"], default=["train", "test"])
    parser.add_argument("--strategies", nargs=2, default=["apriori", "impressed"],
                        metavar=("FIRST", "SECOND"), help="the two strategies to compare")
    parser.add_argument("--top-k", type=int, default=10, help="most important itemsets that are compared")
    parser.add_argument("--plot-top", type=int, default=8, help="itemsets per strategy in the figure")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_directory = config_path.parent / config["output_directory"]

    report = ComparisonReport(results_directory, *arguments.strategies,
                              top_k=arguments.top_k, plot_top=arguments.plot_top)
    output_directory = results_directory / "comparison"
    metrics = report.run(arguments.datasets, arguments.evaluate_on, output_directory)

    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(metrics.set_index(["dataset", "evaluated_on"]).T.round(3).to_string())
    print(f"\nResults: {output_directory}")


if __name__ == "__main__":
    main()
