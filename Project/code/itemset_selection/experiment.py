"""Runs the selection strategies on the configured datasets and stores the results."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from .apriori_selector import AprioriItemsetSelector
from .evaluation import compare_strategies, evaluate_against_gold
from .event_log import EventLog
from .impressed_selector import ImpressedItemsetSelector
from .selector import ItemsetSelector, SelectionResult
from .simulation import SimulatedLogGenerator, SimulationConfig


class ExperimentRunner:
    """Reads the YAML configuration and executes all experiments.

    All paths in the configuration are relative to ``base_directory``. The first
    strategy in the configuration is the reference that all others are compared
    with.
    """

    def __init__(self, config: dict, base_directory: Path, output_directory: Path) -> None:
        self.config = config
        self.base_directory = base_directory
        self.output_directory = output_directory

    @classmethod
    def from_file(cls, config_path: Path, output_directory: str | None = None) -> "ExperimentRunner":
        config_path = Path(config_path)
        with open(config_path, encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        base = config_path.resolve().parent
        output = base / (output_directory or config.get("output_directory", "results"))
        return cls(config, base, output)

    # --- running ------------------------------------------------------------------

    def run(self, only: list[str] | None = None) -> pd.DataFrame:
        """Run all (or the named) datasets; return one summary row per dataset and seed."""
        self.output_directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for dataset in self.config.get("datasets", []):
            if not only or dataset["name"] in only:
                rows.append(self._run_real_dataset(dataset))
        simulation = self.config.get("simulation", {})
        for scenario in simulation.get("scenarios", []):
            if only and scenario["name"] not in only:
                continue
            for repetition in range(simulation.get("repetitions", 1)):
                rows.append(self._run_simulated_dataset(scenario, repetition))

        summary = pd.DataFrame(rows)
        summary.to_csv(self.output_directory / "summary.csv", index=False)
        self._aggregate_simulations(summary).to_csv(
            self.output_directory / "summary_simulated_mean.csv"
        )
        return summary

    @staticmethod
    def _aggregate_simulations(summary: pd.DataFrame) -> pd.DataFrame:
        """Mean of every numeric column over the repetitions of each simulated scenario."""
        if "scenario" not in summary.columns:
            return pd.DataFrame()
        simulated = summary[summary["scenario"].notna()]
        return simulated.groupby("scenario").mean(numeric_only=True).drop(columns=["seed"], errors="ignore")

    # --- datasets -----------------------------------------------------------------

    def _attributes(self, simulated: bool) -> tuple[list[str], list[str]]:
        """Case attributes for the case distance; simulated logs have their own."""
        section = self.config.get("simulation", {}) if simulated else self.config
        attributes = section.get("case_attributes", {})
        return attributes.get("numeric", []), attributes.get("categorical", [])

    def _read_log(self, path: Path, simulated: bool = False) -> EventLog:
        columns = self.config.get("columns", {})
        numeric, categorical = self._attributes(simulated)
        return EventLog.from_csv(
            str(path),
            case_column=columns.get("case", "case:concept:name"),
            activity_column=columns.get("activity", "concept:name"),
            label_column=columns.get("label", "label"),
            order_column=columns.get("order", "event_nr"),
            numeric_attributes=numeric,
            categorical_attributes=categorical,
        )

    def _run_real_dataset(self, dataset: dict) -> dict:
        log = self._read_log(self.base_directory / dataset["path"])
        results = self._select_on_training_split(log, dataset["name"], simulated=False)
        return self._summarise({"dataset": dataset["name"]}, log, results, gold=None)

    def _run_simulated_dataset(self, scenario: dict, repetition: int) -> dict:
        parameters = {k: v for k, v in scenario.items() if k != "name"}
        seed = self.config.get("seed", 42) + repetition
        generator = SimulatedLogGenerator(SimulationConfig(seed=seed, **parameters))
        name = f"{scenario['name']}_seed{seed}"

        simulated_directory = self.output_directory / "simulated"
        simulated_directory.mkdir(parents=True, exist_ok=True)
        csv_path = simulated_directory / f"{name}.csv"
        generator.generate().to_csv(csv_path, index=False)
        gold = generator.gold_standard()
        with open(simulated_directory / f"{name}_gold.json", "w", encoding="utf-8") as handle:
            json.dump(gold, handle, indent=2)

        log = self._read_log(csv_path, simulated=True)
        results = self._select_on_training_split(log, name, simulated=True)
        identity = {"dataset": name, "scenario": scenario["name"], "seed": seed}
        return self._summarise(identity, log, results, gold=gold)

    # --- strategies ---------------------------------------------------------------

    def _build_selectors(self, simulated: bool) -> list[ItemsetSelector]:
        """Create the strategies listed under ``strategies`` in the configuration."""
        numeric, categorical = self._attributes(simulated)
        selectors = []
        for entry in self.config["strategies"]:
            parameters = {k: v for k, v in entry.items() if k != "type"}
            if entry["type"] == "apriori":
                selectors.append(AprioriItemsetSelector(**parameters))
            elif entry["type"] == "impressed":
                selectors.append(
                    ImpressedItemsetSelector(
                        numeric_attributes=numeric, categorical_attributes=categorical, **parameters
                    )
                )
            else:
                raise ValueError(f"Unknown strategy type: {entry['type']}")
        return selectors

    def _select_on_training_split(
        self, log: EventLog, name: str, simulated: bool
    ) -> list[SelectionResult]:
        """Select itemsets on the training cases only, to avoid leaking test labels."""
        train, _ = log.stratified_split(
            self.config.get("test_size", 0.2), self.config.get("seed", 42)
        )
        dataset_directory = self.output_directory / name
        dataset_directory.mkdir(parents=True, exist_ok=True)
        results = []
        for selector in self._build_selectors(simulated):
            result = selector.select(train)
            result.details.to_csv(dataset_directory / f"{result.strategy}_details.csv", index=False)
            with open(dataset_directory / f"{result.strategy}_itemsets.json", "w", encoding="utf-8") as handle:
                json.dump([sorted(itemset) for itemset in result.itemsets], handle, indent=1)
            results.append(result)
        return results

    def _summarise(self, identity: dict, log: EventLog, results: list[SelectionResult], gold) -> dict:
        row = dict(identity)
        row.update(
            {
                "cases": len(log),
                "events": log.number_of_events,
                "activities": len(log.activity_names),
                "positive_rate": round(float(log.labels.mean()), 3),
            }
        )
        reference = results[0]
        for result in results:
            row[f"{result.strategy}__itemsets"] = len(result.itemsets)
            for size, count in result.sizes().items():
                row[f"{result.strategy}__size{size}"] = count
            if result is not reference:
                for key, value in compare_strategies(reference, result).items():
                    row[f"{result.strategy}__{key}_vs_{reference.strategy}"] = round(value, 4)
            if gold is not None:
                evaluation = evaluate_against_gold(result, gold)
                row[f"{result.strategy}__location_recall"] = evaluation["location_recall"]
                row[f"{result.strategy}__existence_recall"] = evaluation["existence_recall"]
                row[f"{result.strategy}__noise_only_share"] = round(evaluation["noise_only_share"], 3)
        return row
