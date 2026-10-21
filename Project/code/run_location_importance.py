"""Itemset selection (Apriori, IMPresseD) followed by the location permutation importance.

For every dataset and strategy: select itemsets, run the 5-fold cross-validated location
permutation importance of Vazifehdoostirani et al. (2024) and write tables and a box plot.

Usage:
    python run_location_importance.py                              # everything in the config
    python run_location_importance.py --datasets synthetic         # selected datasets only
    python run_location_importance.py --datasets BPIC11_f1 --repetitions 2 --max-itemsets 20   # quick look
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import yaml

from itemset_selection import AprioriItemsetSelector, ImpressedItemsetSelector
from location_permutation import CrossValidatedLocationImportance, PreparedLog
from location_permutation.reporting import box_plot, summarise, to_wide

BASE = Path(__file__).resolve().parent


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, default=BASE / "importance_config.yaml")
    parser.add_argument("--datasets", nargs="+", help="names of the datasets to run")
    parser.add_argument("--strategies", nargs="+", help="names of the strategies to run")
    parser.add_argument("--repetitions", type=int, help="override the number of permutation repetitions")
    parser.add_argument("--folds", type=int, help="override the number of cross-validation folds")
    parser.add_argument("--evaluate-on", choices=["train", "test"],
                        help="measure the F1 drop on the training or the test cases of each fold")
    parser.add_argument("--max-itemsets", type=int, help="only the first N itemsets per strategy (quick look)")
    parser.add_argument("--top", type=int, default=8, help="itemsets printed per strategy (default: 8)")
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config_path = arguments.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    base = config_path.parent

    columns = config["columns"]
    evaluate_on = arguments.evaluate_on or config["evaluate_on"]
    analysis = CrossValidatedLocationImportance(
        folds=arguments.folds or config["folds"],
        repetitions=arguments.repetitions or config["repetitions"],
        constrain=config["constrain"], seed=config["seed"],
        accumulate_permutations=config["accumulate_permutations"],
        evaluate_on=evaluate_on,
        exact_tracking=config["exact_tracking"],
    )

    for dataset in config["datasets"]:
        if arguments.datasets and dataset["name"] not in arguments.datasets:
            continue
        attributes = dataset["case_attributes"]
        log = PreparedLog.from_csv(
            str(base / dataset["path"]), case_column=columns["case"], activity_column=columns["activity"],
            label_column=columns["label"], order_column=columns["order"])
        event_log = log.to_event_log(attributes["numeric"], attributes["categorical"])
        training_log, _ = event_log.stratified_split(config["test_size"], config["seed"])
        print(f"\n=== {dataset['name']}: {len(log.case_ids)} cases, {len(log.activities)} activities, "
              f"max trace length {log.max_length}")

        for entry in config["strategies"]:
            if arguments.strategies and entry["name"] not in arguments.strategies:
                continue
            parameters = {k: v for k, v in entry.items() if k not in {"type", "select_on"}}
            if entry["type"] == "apriori":
                selector = AprioriItemsetSelector(**parameters)
            else:
                selector = ImpressedItemsetSelector(
                    numeric_attributes=attributes["numeric"], categorical_attributes=attributes["categorical"],
                    **parameters)
            selection_log = event_log if entry["select_on"] == "all" else training_log
            itemsets = selector.select(selection_log).itemsets
            if arguments.max_itemsets:
                itemsets = itemsets[: arguments.max_itemsets]

            started = time.time()
            result = analysis.run(log, itemsets)
            summary = summarise(result.importance)

            folder = base / config["output_directory"] / dataset["name"] / f"evaluated_on_{evaluate_on}"
            folder.mkdir(parents=True, exist_ok=True)
            stem = f"{entry['name']}"
            to_wide(result.importance).to_csv(folder / f"{stem}_location_importance.csv", index=False)
            summary.to_csv(folder / f"{stem}_summary.csv", index=False)
            result.fold_scores.to_csv(folder / f"{stem}_fold_scores.csv", index=False)
            box_plot(result.importance, folder / f"{stem}_boxplot.png",
                     f"Location permutation importance: {dataset['name']}, {entry['name']}")

            print(f"-- {entry['name']}: {len(itemsets)} itemsets ({time.time() - started:.0f} s), "
                  f"train F1 {result.fold_scores.train_f1.mean():.3f}, test F1 {result.fold_scores.test_f1.mean():.3f}")
            print(summary.head(arguments.top).round(4).to_string(index=False))
    print(f"\nResults: {base / config['output_directory']}")


if __name__ == "__main__":
    main()
