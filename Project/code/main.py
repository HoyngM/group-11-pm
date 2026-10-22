"""Entry point of the project: runs the whole experiment, or some of its steps.

Steps (in this order, see pipeline/runner.py):
    generate    synthetic log with a planted location effect + shuffled controls -> data/synthetic/
    check       is the planted effect present in the synthetic log and absent in the controls?
    select      itemsets of every strategy (Apriori, IMPresseD) on every dataset -> results/selection/
    importance  location permutation importance of those itemsets -> results/importance/
    compare     importance of the Apriori and the IMPresseD itemsets side by side -> results/comparison/

Usage:
    python main.py                                         # everything (about 20 minutes)
    python main.py --steps select importance compare       # only these steps
    python main.py --datasets synthetic --evaluate-on test # one dataset, one setting (seconds)
    python main.py --datasets BPIC11_f1 --repetitions 2 --max-itemsets 30   # quick look

All parameters are in config.yaml; the options below only override some of them for one run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from pipeline import STEPS, Pipeline


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml",
                        help="YAML file with all parameters (default: config.yaml next to main.py)")
    parser.add_argument("--steps", nargs="+", choices=STEPS, help="steps to run (default: all)")
    parser.add_argument("--datasets", nargs="+",
                        help="names of the datasets or simulation scenarios (default: all in the config)")
    parser.add_argument("--output-directory", help="folder for the results (default: from the config)")
    parser.add_argument("--evaluate-on", nargs="+", choices=["train", "test"],
                        help="measure the F1 drop on the training and/or the test cases of each fold")
    parser.add_argument("--repetitions", type=int, help="permutations per itemset and fold")
    parser.add_argument("--folds", type=int, help="cross-validation folds")
    parser.add_argument("--max-itemsets", type=int, help="only the first N itemsets per strategy (quick look)")
    return parser.parse_args()


def apply_overrides(pipeline: Pipeline, arguments: argparse.Namespace) -> None:
    """Replace the config values for which a command line option was given."""
    importance = pipeline.config["importance"]
    for option, key in (("evaluate_on", "evaluate_on"), ("repetitions", "repetitions"),
                        ("folds", "folds"), ("max_itemsets", "max_itemsets")):
        if getattr(arguments, option) is not None:
            importance[key] = getattr(arguments, option)
    if arguments.output_directory:
        pipeline.output_directory = pipeline.base_directory / arguments.output_directory


def main() -> None:
    arguments = parse_arguments()
    pipeline = Pipeline.from_file(arguments.config)
    apply_overrides(pipeline, arguments)
    pipeline.run(steps=arguments.steps, only=arguments.datasets)
    print(f"\nResults: {pipeline.output_directory}")


if __name__ == "__main__":
    main()
