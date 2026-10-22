"""The complete experiment as one pipeline of five steps.

1. ``generate``    write the synthetic log with a planted location effect and its three
                   shuffled controls to ``data/synthetic/`` (the ground truth).
2. ``check``       confirm, without any model, that the planted effect is present in the
                   synthetic log and absent in the controls.
3. ``select``      choose the itemsets with every strategy (Apriori, IMPresseD, ...) on every
                   dataset, and check the strategies against the gold standard of simulated logs.
4. ``importance``  measure the location permutation importance of the selected itemsets
                   (Vazifehdoostirani et al., 2024): train XGBoost per fold, permute the location
                   of each itemset and record the drop in F1.
5. ``compare``     put the importance of the Apriori and the IMPresseD itemsets side by side.

Every step reads its input from disk and writes its output to disk, so a step can be rerun
on its own (``python main.py --steps compare``). Step 4 reads the itemsets that step 3
wrote, so both steps always use exactly the same itemsets.

All event logs go through ``PreparedLog`` (the cleaning of the reference code: lower-case
activity names, cases with very rare activities removed), so the selection and the importance
see the same cases and the same activity names.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd
import yaml

from evaluation.control_checks import check_controls
from evaluation.selection_metrics import compare_selections, evaluate_against_gold
from evaluation.strategy_comparison import ComparisonReport
from itemset_selection import AprioriItemsetSelector, EventLog, ImpressedItemsetSelector
from itemset_selection.selector import ItemsetSelector, SelectionResult
from itemset_selection.simulation import SimulatedLogGenerator, SimulationConfig
from location_permutation import CrossValidatedLocationImportance, PreparedLog
from location_permutation.reporting import box_plot, summarise, to_wide
from synthetic_data import ShuffleControls, SyntheticLogGenerator
from synthetic_data.generator import GeneratorConfig

STEPS = ("generate", "check", "select", "importance", "compare")
DATASET_STEPS = {"select", "importance", "compare"}  # steps that can be limited to some datasets

# Keys of a strategy entry in config.yaml that are read by the pipeline, not by the selector.
PIPELINE_KEYS = {"type", "select_on", "importance"}


class Pipeline:
    """Runs the steps of the experiment with the parameters of ``config.yaml``.

    ``config`` is the parsed YAML file and ``base_directory`` the folder that its relative
    paths start from (the folder of the config file).
    """

    def __init__(self, config: dict, base_directory: Path) -> None:
        self.config = config
        self.base_directory = base_directory
        self.output_directory = base_directory / config["output_directory"]
        self.seed = config["seed"]
        self.test_size = config["test_size"]

    @classmethod
    def from_file(cls, config_path: Path) -> "Pipeline":
        config_path = Path(config_path).resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        return cls(config, config_path.parent)

    def run(self, steps: list[str] | None = None, only: list[str] | None = None) -> None:
        """Run the given steps (default: all) in the fixed order of ``STEPS``.

        ``only`` restricts the steps ``select``, ``importance`` and ``compare`` to the named
        datasets or simulation scenarios.
        """
        for step in STEPS:
            if steps and step not in steps:
                continue
            print(f"\n##### Step {STEPS.index(step) + 1}: {step}")
            started = time.time()
            if step in DATASET_STEPS:
                getattr(self, step)(only)
            else:
                getattr(self, step)()
            print(f"##### {step} done in {time.time() - started:.0f} s")

    # --- step 1: synthetic data ------------------------------------------------------

    def generate(self) -> None:
        """Write the synthetic log and its shuffled controls (``synthetic_data`` in the config)."""
        settings = self.config["synthetic_data"]
        generator_config = GeneratorConfig(target_events=settings["events"], seed=self.seed,
                                           label_noise=settings["label_noise"])
        original = SyntheticLogGenerator(generator_config).generate()
        logs = {"synthetic": original, **ShuffleControls(seed=self.seed).all_controls(original)}

        directory = self.base_directory / settings["output_directory"]
        directory.mkdir(parents=True, exist_ok=True)
        for name, log in logs.items():
            log.to_csv(directory / f"{name}.csv", index=False)
            print(f"{name}.csv: {len(log)} events, {log[self.columns['case']].nunique()} cases")

    # --- step 2: check the planted effect --------------------------------------------

    def check(self) -> pd.DataFrame:
        """Spearman correlation of the position of B with the label, per synthetic log."""
        result = check_controls(self.base_directory / self.config["synthetic_data"]["output_directory"])
        self.output_directory.mkdir(parents=True, exist_ok=True)
        result.to_csv(self.output_directory / "synthetic_checks.csv", index=False)
        print(result.to_string(index=False))
        return result

    # --- step 3: itemset selection ---------------------------------------------------

    def select(self, only: list[str] | None = None) -> pd.DataFrame:
        """Select itemsets with every strategy on every dataset and on the simulated logs.

        Writes ``results/selection/<dataset>/<strategy>_itemsets.json`` (input of step 4),
        ``<strategy>_details.csv`` (why the itemsets were chosen) and ``summary.csv``.
        """
        rows = []
        for dataset in self._datasets(only):
            log = self._load(dataset["path"])
            event_log = self._event_log(log, dataset["case_attributes"])
            results = self._select_all(event_log, dataset["name"], dataset["case_attributes"])
            rows.append(self._selection_row({"dataset": dataset["name"]}, event_log, results, gold=None))
            print(f"{dataset['name']}: " + ", ".join(f"{r.strategy} {len(r.itemsets)}" for r in results))

        simulation = self.config.get("simulation", {})
        for scenario in simulation.get("scenarios", []):
            if only and scenario["name"] not in only:
                continue
            for repetition in range(simulation["repetitions"]):
                rows.append(self._select_simulated(scenario, repetition, simulation["case_attributes"]))
            print(f"{scenario['name']}: {simulation['repetitions']} simulated logs")

        summary = pd.DataFrame(rows)
        directory = self.output_directory / "selection"
        directory.mkdir(parents=True, exist_ok=True)
        summary.to_csv(directory / "summary.csv", index=False)
        if "scenario" in summary.columns:  # mean over the repetitions of each simulated scenario
            simulated = summary[summary["scenario"].notna()]
            simulated.groupby("scenario").mean(numeric_only=True).drop(columns=["seed"]).to_csv(
                directory / "summary_simulated_mean.csv")
        return summary

    def _select_all(self, event_log: EventLog, name: str, attributes: dict) -> list[SelectionResult]:
        """Run every strategy on one log and write its itemsets and details."""
        directory = self.output_directory / "selection" / name
        directory.mkdir(parents=True, exist_ok=True)
        training_log, _ = event_log.stratified_split(self.test_size, self.seed)
        results = []
        for entry in self.config["strategies"]:
            selection_log = event_log if entry["select_on"] == "all" else training_log
            result = self._selector(entry, attributes).select(selection_log)
            result.details.to_csv(directory / f"{result.strategy}_details.csv", index=False)
            with open(directory / f"{result.strategy}_itemsets.json", "w", encoding="utf-8") as handle:
                json.dump([sorted(itemset) for itemset in result.itemsets], handle, indent=1)
            results.append(result)
        return results

    def _select_simulated(self, scenario: dict, repetition: int, attributes: dict) -> dict:
        """Generate one simulated log, select itemsets on it and score them against its gold standard."""
        seed = self.seed + repetition
        parameters = {key: value for key, value in scenario.items() if key != "name"}
        generator = SimulatedLogGenerator(SimulationConfig(seed=seed, **parameters))
        name = f"{scenario['name']}_seed{seed}"

        directory = self.output_directory / "selection" / "simulated"
        directory.mkdir(parents=True, exist_ok=True)
        generator.generate().to_csv(directory / f"{name}.csv", index=False)
        # Write the gold standard in the spelling that PreparedLog gives the activities.
        gold = {key: [self._clean(value) for value in values]
                for key, values in generator.gold_standard().items()}
        with open(directory / f"{name}_gold.json", "w", encoding="utf-8") as handle:
            json.dump(gold, handle, indent=2)

        event_log = self._event_log(self._load(directory / f"{name}.csv"), attributes)
        results = self._select_all(event_log, name, attributes)
        identity = {"dataset": name, "scenario": scenario["name"], "seed": seed}
        return self._selection_row(identity, event_log, results, gold=gold)

    @staticmethod
    def _clean(value):
        """Clean an activity name, or every name in a list of names."""
        if isinstance(value, list):
            return [PreparedLog.clean_activity_name(item) for item in value]
        return PreparedLog.clean_activity_name(value)

    @staticmethod
    def _selection_row(identity: dict, log: EventLog, results: list[SelectionResult], gold) -> dict:
        """One row of ``summary.csv``: size of the log, of each selection and their overlap."""
        row = dict(identity)
        row.update({"cases": len(log), "events": log.number_of_events,
                    "activities": len(log.activity_names), "positive_rate": round(float(log.labels.mean()), 3)})
        reference = results[0]  # the first strategy in the config is the reference
        for result in results:
            row[f"{result.strategy}__itemsets"] = len(result.itemsets)
            for size, count in result.sizes().items():
                row[f"{result.strategy}__size{size}"] = count
            if result is not reference:
                for key, value in compare_selections(reference, result).items():
                    row[f"{result.strategy}__{key}_vs_{reference.strategy}"] = round(value, 4)
            if gold is not None:
                evaluation = evaluate_against_gold(result, gold)
                row[f"{result.strategy}__location_recall"] = evaluation["location_recall"]
                row[f"{result.strategy}__existence_recall"] = evaluation["existence_recall"]
                row[f"{result.strategy}__noise_only_share"] = round(evaluation["noise_only_share"], 3)
        return row

    # --- step 4: location permutation importance -------------------------------------

    def importance(self, only: list[str] | None = None) -> None:
        """Location importance of the itemsets of every strategy with ``importance: true``.

        Writes, per dataset, strategy and evaluation setting, to
        ``results/importance/<dataset>/evaluated_on_<train|test>/``: the F1 drop of every
        itemset in every fold and repetition, a summary per itemset, the F1 of the models and
        a box plot.
        """
        settings = self.config["importance"]
        strategies = [entry["name"] for entry in self.config["strategies"] if entry["importance"]]
        for dataset in self._datasets(only):
            log = self._load(dataset["path"])
            print(f"\n=== {dataset['name']}: {len(log.case_ids)} cases, {len(log.activities)} activities, "
                  f"max trace length {log.max_length}")
            for evaluate_on in settings["evaluate_on"]:
                analysis = CrossValidatedLocationImportance(
                    folds=settings["folds"], repetitions=settings["repetitions"],
                    constrain=settings["constrain"], seed=self.seed,
                    accumulate_permutations=settings["accumulate_permutations"],
                    evaluate_on=evaluate_on, exact_tracking=settings["exact_tracking"])
                folder = self.output_directory / "importance" / dataset["name"] / f"evaluated_on_{evaluate_on}"
                folder.mkdir(parents=True, exist_ok=True)
                for strategy in strategies:
                    self._importance_of(log, dataset["name"], strategy, analysis, folder, evaluate_on)

    def _importance_of(self, log: PreparedLog, dataset: str, strategy: str,
                       analysis: CrossValidatedLocationImportance, folder: Path, evaluate_on: str) -> None:
        """Run the importance for one strategy and write its four output files."""
        itemsets = self._read_itemsets(dataset, strategy)
        max_itemsets = self.config["importance"].get("max_itemsets")
        if max_itemsets:
            itemsets = itemsets[:max_itemsets]

        started = time.time()
        result = analysis.run(log, itemsets)
        summary = summarise(result.importance)
        to_wide(result.importance).to_csv(folder / f"{strategy}_location_importance.csv", index=False)
        summary.to_csv(folder / f"{strategy}_summary.csv", index=False)
        result.fold_scores.to_csv(folder / f"{strategy}_fold_scores.csv", index=False)
        box_plot(result.importance, folder / f"{strategy}_boxplot.png",
                 f"Location permutation importance: {dataset}, {strategy}")

        best = summary.iloc[0]
        print(f"-- {strategy}, evaluated on {evaluate_on}: {len(itemsets)} itemsets "
              f"({time.time() - started:.0f} s), test F1 {result.fold_scores.test_f1.mean():.3f}, "
              f"most important: {best['itemset']} ({best['mean']:.4f})")

    def _read_itemsets(self, dataset: str, strategy: str) -> list[frozenset[str]]:
        """The itemsets that step 3 selected, in the same order."""
        path = self.output_directory / "selection" / dataset / f"{strategy}_itemsets.json"
        if not path.exists():
            raise FileNotFoundError(f"{path} not found: run the step 'select' first")
        with open(path, encoding="utf-8") as handle:
            return [frozenset(itemset) for itemset in json.load(handle)]

    # --- step 5: comparison of the strategies ----------------------------------------

    def compare(self, only: list[str] | None = None) -> pd.DataFrame | None:
        """Compare the importance of the reference and the variant strategy (``comparison`` in the config)."""
        settings = self.config["comparison"]
        datasets = [name for name in settings["datasets"] if not only or name in only]
        if not datasets:
            print("No dataset of the comparison selected; nothing to compare.")
            return None
        report = ComparisonReport(self.output_directory / "importance", settings["reference"], settings["variant"],
                                  top_k=settings["top_k"], plot_top=settings["plot_top"])
        metrics = report.run(datasets, self.config["importance"]["evaluate_on"], self.output_directory / "comparison")
        with pd.option_context("display.width", 200, "display.max_columns", None):
            print(metrics.set_index(["dataset", "evaluated_on"]).T.round(3).to_string())
        return metrics

    # --- shared helpers ----------------------------------------------------------------

    @property
    def columns(self) -> dict:
        return self.config["columns"]

    def _datasets(self, only: list[str] | None) -> list[dict]:
        return [dataset for dataset in self.config["datasets"] if not only or dataset["name"] in only]

    def _load(self, path: str | Path) -> PreparedLog:
        """Read and clean an event log (paths are relative to the config file)."""
        return PreparedLog.from_csv(
            str(self.base_directory / path), case_column=self.columns["case"],
            activity_column=self.columns["activity"], label_column=self.columns["label"],
            order_column=self.columns["order"])

    @staticmethod
    def _event_log(log: PreparedLog, attributes: dict) -> EventLog:
        """The cleaned log in the form the selection strategies need."""
        return log.to_event_log(attributes["numeric"], attributes["categorical"])

    @staticmethod
    def _selector(entry: dict, attributes: dict) -> ItemsetSelector:
        """Create the strategy described by one entry under ``strategies`` in the config."""
        parameters = {key: value for key, value in entry.items() if key not in PIPELINE_KEYS}
        if entry["type"] == "apriori":
            return AprioriItemsetSelector(**parameters)
        if entry["type"] == "impressed":
            return ImpressedItemsetSelector(numeric_attributes=attributes["numeric"],
                                            categorical_attributes=attributes["categorical"], **parameters)
        raise ValueError(f"Unknown strategy type: {entry['type']}")
