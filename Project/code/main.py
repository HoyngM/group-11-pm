"""Entry point: compares Apriori and IMPresseD itemset selection.

Usage:
    python main.py                       # run everything in config.yaml
    python main.py --datasets BPIC11_f1  # run selected datasets only
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from itemset_selection.experiment import ExperimentRunner


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "config.yaml",
        help="YAML file with datasets and parameters (default: config.yaml next to main.py)",
    )
    parser.add_argument("--output-directory", help="folder for the results (default: from the config)")
    parser.add_argument("--datasets", nargs="+", help="names of the datasets or scenarios to run")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    runner = ExperimentRunner.from_file(arguments.config, arguments.output_directory)
    summary = runner.run(only=arguments.datasets)
    shown = summary.drop(columns=["scenario", "seed"], errors="ignore")
    shown = shown[~shown["dataset"].str.contains("_seed")]  # simulated runs: see summary files
    with pd.option_context("display.max_columns", None, "display.width", 200, "display.max_rows", None):
        print(shown.set_index("dataset").dropna(axis=1, how="all").T.to_string())
    print(f"\nAll runs: {runner.output_directory / 'summary.csv'}")
    print(f"Simulated scenarios (mean over repetitions): {runner.output_directory / 'summary_simulated_mean.csv'}")


if __name__ == "__main__":
    main()
