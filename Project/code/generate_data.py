"""Generate the synthetic event log and its shuffled negative controls.

The generated files are already included in data/synthetic/; run this only to regenerate them.

Usage:
    python generate_data.py
    python generate_data.py --events 200 --seed 7 --output-directory data
"""

from __future__ import annotations

import argparse
from pathlib import Path

from extensions.shuffle_controls import ShuffleControls
from extensions.synthetic_log import GeneratorConfig, SyntheticLogGenerator


def parse_arguments() -> argparse.Namespace:
    """Command-line options for the size, seed, label noise and output folder."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--events", type=int, default=100,
                        help="exact number of events (default: 100); ignored when --cases is given")
    parser.add_argument("--cases", type=int, help="number of cases instead of a fixed number of events")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument("--label-noise", type=float, default=0.1,
                        help="probability that a label is flipped (default: 0.1)")
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path(__file__).resolve().parent / "data" / "synthetic",
        help="folder for the CSV files (default: data/synthetic)",
    )
    return parser.parse_args()


def main() -> None:
    """Generate the synthetic log and its controls and write each to a CSV file."""
    arguments = parse_arguments()
    # A fixed number of events, unless a number of cases is given.
    config = GeneratorConfig(target_events=None if arguments.cases else arguments.events,
                             number_of_cases=arguments.cases or 15, seed=arguments.seed,
                             label_noise=arguments.label_noise)
    original = SyntheticLogGenerator(config).generate()
    logs = {"synthetic": original, **ShuffleControls(seed=arguments.seed).all_controls(original)}

    arguments.output_directory.mkdir(parents=True, exist_ok=True)
    for name, log in logs.items():
        path = arguments.output_directory / f"{name}.csv"
        log.to_csv(path, index=False)
        print(f"{path.name}: {len(log)} events, {log['case:concept:name'].nunique()} cases")


if __name__ == "__main__":
    main()
