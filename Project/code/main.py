"""Entry point: itemset selection (Apriori, IMPresseD) followed by the location permutation importance.

Usage:
    python main.py                                         # everything in config.yaml
    python main.py --datasets synthetic                    # selected datasets or simulated scenarios
    python main.py --datasets BPIC11_f1 --n-repeats 2 --max-itemsets 20   # quick look

Order of the work (``Experiment.run``):

1. for every dataset in the configuration (``_run_dataset``):
   a. load the log (``DataManager`` in tools.py),
   b. let every strategy choose its itemsets (``_select``, extensions/itemset_selection.py),
   c. per strategy: location importance (cross_validation_process_permutation.py) and output files
      (plotting_results.py);
2. write ``results/summary.csv`` with one row per dataset and strategy;
3. for every simulated scenario (``_run_simulations``): selection only, scored against the
   itemsets that were planted in the simulated logs (extensions/simulation.py, evaluation.py).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import yaml

from cross_validation_process_permutation import CrossValidationProcessPermutation
from extensions.evaluation import compare_strategies, evaluate_against_gold
from extensions.itemset_selection import AprioriSelector, ImpressedSelector, ItemsetSelector, SelectionResult
from extensions.simulation import SimulatedLogGenerator, SimulationConfig
from plotting_results import box_plot, summarise, to_wide
from tools import DataManager

# Strategy classes by the ``type`` used in config.yaml.
SELECTORS = {"apriori": AprioriSelector, "impressed": ImpressedSelector}


class Experiment:
    """Reads the configuration and runs the experiments; the first strategy is the reference.

    Args:
        config: the parsed configuration (see config.yaml).
        base_directory: folder that the paths in the configuration are relative to.
        output_directory: folder for all results.
    """

    def __init__(self, config: dict, base_directory: Path, output_directory: Path) -> None:
        self.config = config
        self.base_directory = base_directory
        self.output_directory = output_directory

    @classmethod
    def from_file(cls, config_path: Path, output_directory: str | None = None) -> "Experiment":
        """Experiment for a YAML file; the output folder defaults to the one in the file."""
        config_path = Path(config_path).resolve()  # absolute path, so the folder below is known
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))  # YAML -> nested dicts/lists
        base = config_path.parent  # data paths in the config are relative to this folder
        return cls(config, base, base / (output_directory or config["output_directory"]))

    def run(self, datasets: list[str] | None = None, strategies: list[str] | None = None,
            max_itemsets: int | None = None, **importance_overrides) -> None:
        """Run the selected (default: all) datasets, strategies and simulated scenarios.

        ``importance_overrides`` (``k_fold``, ``n_repeats``) replace the values of the
        ``importance`` section of the configuration when they are not ``None``.
        """
        # Settings of the importance step: the config values, overwritten by the command line.
        settings = {**self.config["importance"],
                    **{key: value for key, value in importance_overrides.items() if value is not None}}
        self.importance = CrossValidationProcessPermutation(seed=self.config["seed"], **settings)
        # The strategies to run (all of them, unless --strategies named some).
        self.strategy_entries = [entry for entry in self.config["strategies"]
                                 if not strategies or entry["name"] in strategies]
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Real and synthetic logs: selection + location importance.
        rows = []
        for dataset in self.config["datasets"]:
            if not datasets or dataset["name"] in datasets:
                rows += self._run_dataset(dataset, max_itemsets)
        if rows:
            pd.DataFrame(rows).to_csv(self.output_directory / "summary.csv", index=False)

        # Simulated logs: selection only, compared with the planted itemsets.
        simulation = self.config.get("simulation", {})
        scenarios = [s for s in simulation.get("scenarios", []) if not datasets or s["name"] in datasets]
        if scenarios:
            self._run_simulations(simulation, scenarios)
        print(f"\nResults: {self.output_directory}")

    def _selectors(self) -> list[tuple[ItemsetSelector, str]]:
        """The configured strategies, each with the cases it selects on (``all`` or ``training_split``)."""
        selectors = []
        for entry in self.strategy_entries:
            # Everything except type and select_on is a constructor argument of the strategy, e.g.
            # {"name": "apriori", "min_support": 0.5, ...} -> AprioriSelector(name="apriori", min_support=0.5, ...).
            parameters = {k: v for k, v in entry.items() if k not in {"type", "select_on"}}
            selectors.append((SELECTORS[entry["type"]](**parameters), entry["select_on"]))
        return selectors

    def _read_log(self, path: Path, attributes: dict) -> DataManager:
        """Load a CSV log with the configured column names and case attributes."""
        return DataManager(path, columns=self.config["columns"],
                           numeric_attributes=attributes.get("numeric", []),
                           categorical_attributes=attributes.get("categorical", []),
                           frq_threshold=self.config["frq_threshold"])

    def _select(self, data_manager: DataManager, folder: Path) -> list[SelectionResult]:
        """Run every strategy and save the itemsets and the details of the selection to ``folder``."""
        # The training part of an 80/20 split; strategies that use the labels (IMPresseD) select
        # on these cases only, so that the test cases do not influence which itemsets are chosen.
        training_cases = data_manager.split_test_train(self.config["test_size"], self.config["seed"])
        folder.mkdir(parents=True, exist_ok=True)
        results = []
        for selector, select_on in self._selectors():
            result = selector.select(data_manager, data_manager.case_ids if select_on == "all" else training_cases)
            result.details.to_csv(folder / f"{result.strategy}_details.csv", index=False)
            # JSON cannot store sets, so every itemset is written as a sorted list.
            (folder / f"{result.strategy}_itemsets.json").write_text(
                json.dumps([sorted(itemset) for itemset in result.itemsets], indent=1), encoding="utf-8")
            results.append(result)
        return results

    def _run_dataset(self, dataset: dict, max_itemsets: int | None) -> list[dict]:
        """Selection and location importance on one dataset; returns one summary row per strategy."""
        # a. Load the log.
        data_manager = self._read_log(self.base_directory / dataset["path"], dataset.get("case_attributes", {}))
        print(f"\n=== {dataset['name']}: {len(data_manager)} cases, {len(data_manager.activities)} activities, "
              f"max trace length {data_manager.max_length}")
        print(f"fixed-location activities: {', '.join(data_manager.fixed_location_activities) or 'none'}")
        folder = self.output_directory / dataset["name"]
        # b. Every strategy chooses its itemsets.
        results = self._select(data_manager, folder)

        # c. Location importance of the itemsets of every strategy.
        rows = []
        for result in results:
            started = time.time()
            itemsets = result.itemsets[:max_itemsets] if max_itemsets else result.itemsets
            importance = self.importance.run(data_manager, itemsets)
            summary = summarise(importance.importance)  # mean and std per itemset
            tested = len(itemsets) - len(importance.excluded_itemsets)

            # Output files of this strategy (see the README for their content).
            (folder / f"{result.strategy}_excluded_itemsets.json").write_text(
                json.dumps([sorted(itemset) for itemset in importance.excluded_itemsets], indent=1), encoding="utf-8")
            importance.fold_scores.to_csv(folder / f"{result.strategy}_fold_scores.csv", index=False)
            summary.to_csv(folder / f"{result.strategy}_summary.csv", index=False)
            if tested:  # nothing to tabulate or plot if every itemset was excluded
                to_wide(importance.importance).to_csv(folder / f"{result.strategy}_location_importance.csv",
                                                      index=False)
                box_plot(importance.importance, folder / f"{result.strategy}_boxplot.png",
                         f"Location permutation importance: {dataset['name']}, {result.strategy}")

            # Progress on screen: how good the model is, and the five most important itemsets.
            train_f1, test_f1 = importance.fold_scores[["train_f1", "test_f1"]].mean()
            print(f"-- {result.strategy}: {tested} itemsets tested, {len(importance.excluded_itemsets)} excluded "
                  f"(fixed location) ({time.time() - started:.0f} s), train F1 {train_f1:.3f}, test F1 {test_f1:.3f}")
            print(summary.head(5).round(4).to_string(index=False))
            # One row of results/summary.csv. The "**{...}" parts add a variable number of
            # columns: the itemset sizes (size1, size2, ...) and the overlap with the first strategy.
            rows.append({
                "dataset": dataset["name"], "strategy": result.strategy, "cases": len(data_manager),
                "events": data_manager.number_of_events, "activities": len(data_manager.activities),
                "positive_rate": round(float(data_manager.labels.mean()), 3), "itemsets": len(itemsets),
                "excluded_fixed_location": len(importance.excluded_itemsets),
                **{f"size{size}": count for size, count in result.sizes().items()},
                **{f"{key}_vs_{results[0].strategy}": round(value, 4)
                   for key, value in compare_strategies(results[0], result).items()},
                "train_f1": round(train_f1, 4), "test_f1": round(test_f1, 4),
                "top_itemset": summary["itemset"].iloc[0] if tested else None,
                "top_importance": round(summary["mean"].iloc[0], 4) if tested else None,
            })
        return rows

    def _run_simulations(self, simulation: dict, scenarios: list[dict]) -> None:
        """Selection on simulated logs, scored against the itemsets that were planted."""
        folder = self.output_directory / "simulated"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        for scenario in scenarios:
            # Every field except the name overrides a field of SimulationConfig.
            parameters = {k: v for k, v in scenario.items() if k != "name"}
            # Several logs per scenario, each with its own seed, to average out randomness.
            for repetition in range(simulation["repetitions"]):
                seed = self.config["seed"] + repetition
                generator = SimulatedLogGenerator(SimulationConfig(seed=seed, **parameters))
                name = f"{scenario['name']}_seed{seed}"
                # Save the log and read it back, so it goes through the same loading as the real logs.
                generator.generate().to_csv(folder / f"{name}.csv", index=False)
                data_manager = self._read_log(folder / f"{name}.csv", simulation["case_attributes"])
                for result in self._select(data_manager, folder / name):
                    rows.append({"scenario": scenario["name"], "seed": seed, "strategy": result.strategy,
                                 "itemsets": len(result.itemsets),
                                 **evaluate_against_gold(result, generator.gold_standard())})
        # All runs, and the mean per scenario and strategy over the repetitions.
        runs = pd.DataFrame(rows)
        runs.to_csv(folder / "summary_simulated.csv", index=False)
        mean = runs.drop(columns="seed").groupby(["scenario", "strategy"]).mean().round(3)
        mean.to_csv(folder / "summary_simulated_mean.csv")
        print("\n=== Simulated logs (mean over repetitions)")
        print(mean.to_string())


def parse_arguments() -> argparse.Namespace:
    """Command-line options; all of them are optional overrides of config.yaml."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config.yaml",
                        help="YAML file with datasets and parameters (default: config.yaml next to main.py)")
    parser.add_argument("--output-directory", help="folder for the results (default: from the config)")
    parser.add_argument("--datasets", nargs="+", help="names of the datasets or simulated scenarios to run")
    parser.add_argument("--strategies", nargs="+", help="names of the strategies to run")
    parser.add_argument("--k-fold", type=int, help="override the number of cross-validation folds")
    parser.add_argument("--n-repeats", type=int, help="override the number of permutation repetitions")
    parser.add_argument("--max-itemsets", type=int, help="only the first N itemsets per strategy (quick look)")
    return parser.parse_args()


def main() -> None:
    """Read the options and run the configured experiment."""
    arguments = parse_arguments()
    experiment = Experiment.from_file(arguments.config, arguments.output_directory)
    experiment.run(datasets=arguments.datasets, strategies=arguments.strategies,
                   max_itemsets=arguments.max_itemsets, k_fold=arguments.k_fold, n_repeats=arguments.n_repeats)


if __name__ == "__main__":
    main()
