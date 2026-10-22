# Location permutation importance with Apriori and IMPresseD itemsets

Group 11, Process Mining (JM0211): *Explainable Predictive Process Monitoring*.

This folder contains the complete implementation of our assignment. It has two parts:

1. **A reproduction** of the location permutation importance of Vazifehdoostirani et al. (2024),
   *Uncovering the Hidden Significance of Activities Location in Predictive Process Monitoring*.
   The method measures how much the *location* of a set of activities (an *itemset*) in a trace
   matters for an outcome prediction model. It moves the itemset to other positions in the
   traces and measures how much the F1 score of the model drops.
2. **A variant** in which the itemsets are chosen with the pattern discovery of IMPresseD
   (Vazifehdoostirani et al., 2023) instead of with Apriori, and **a comparison** of the two.

| Strategy | Source | How the itemsets are chosen |
|---|---|---|
| `apriori` (baseline) | Vazifehdoostirani et al. (2024) | Every single activity plus the `top_k` most frequent itemsets found with Apriori. |
| `impressed` (variant) | Vazifehdoostirani et al. (2023) | Start with single activities, keep the Pareto front of three interest functions (case coverage, outcome information gain, case distance), extend those patterns by one activity, repeat up to length 3. The ordering relations are dropped afterwards. |

Everything runs from one entry point, `main.py`, with all parameters in one file, `config.yaml`.

## Installation

Python 3.11 or newer is required. Run these commands from the folder that contains this README:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

`requirements.txt` pins the exact versions with which the results were produced.

## Reproducing all results

```bash
python main.py
```

This runs the five steps below in order, with the settings of `config.yaml`, and takes 20-25
minutes (most of it is step 4 on the BPIC11 logs). All output goes to `results/`.
Afterwards, check the code with:

```bash
python -m pytest
```

## How the pipeline works

```
 config.yaml
     │
     ▼
 1 generate ──► data/synthetic/*.csv         synthetic log with a planted effect + 3 shuffled controls
 2 check    ──► results/synthetic_checks.csv is the planted effect really in the data?
 3 select   ──► results/selection/            itemsets of every strategy on every dataset
 4 importance ► results/importance/           F1 drop per itemset (reads the itemsets of step 3)
 5 compare  ──► results/comparison/           Apriori against IMPresseD
```

| Step | What it does | Code |
|---|---|---|
| 1 `generate` | Writes a synthetic log of 100 events in which only the *location* of activity `B` decides the outcome (early: label 0, late: label 1, 10% label noise), plus three copies in which this is destroyed by shuffling. This is our ground truth. | `synthetic_data/` |
| 2 `check` | Checks, without any model, that the position of `B` correlates with the label in the synthetic log and not in the controls (Spearman). | `evaluation/control_checks.py` |
| 3 `select` | Cleans every event log, runs every strategy of `config.yaml` on it and saves the itemsets. Also generates simulated logs with a known gold standard and checks which planted itemsets each strategy selects. | `itemset_selection/`, `evaluation/selection_metrics.py` |
| 4 `importance` | For every dataset, strategy and evaluation setting: 5-fold cross-validation; per fold an XGBoost model on the index-based encoding; then every itemset is permuted 10 times and the drop in weighted F1 is recorded. | `location_permutation/` |
| 5 `compare` | Puts the importance of the Apriori and the IMPresseD itemsets side by side: overlap, rank correlation, best itemsets, figure. | `evaluation/strategy_comparison.py` |

The orchestration of these steps is in `pipeline/runner.py` (class `Pipeline`). Each step reads
its input from disk and writes its output to disk, so a step can be rerun on its own. Step 4
always reads the itemsets that step 3 saved, so both steps use exactly the same itemsets.

### Running parts of the pipeline

```bash
python main.py --steps select importance compare               # only these steps
python main.py --datasets synthetic --evaluate-on test --output-directory results_quick   # seconds
python main.py --datasets BPIC11_f1 --repetitions 2 --max-itemsets 30 --output-directory results_quick
python main.py --steps compare                                 # redo the comparison only
python main.py --config other_config.yaml                      # other parameters
python main.py --help                                          # all options
```

The command line options only override `config.yaml` for one run. Steps 4 and 5 need the
output of the earlier steps in the same output directory.

**Only a plain `python main.py` reproduces the reported results.** A run with fewer datasets or
other settings overwrites files in `results/` (for example `selection/summary.csv` then only holds
the chosen datasets), so give such runs their own `--output-directory`, as above.
Folders named `results_*` are ignored by git.

## Data

| Data | Where | Used for |
|---|---|---|
| BPIC11 f1, f2, f3 | `data/BPIC11_*.csv`, from <https://github.com/MozhganVD/PermutationLocationImportance/tree/master/datasets> | The evaluation of the assignment. Hospital logs of gynaecology patients with a binary outcome label, truncated at 36, 40 and 31 events. |
| Synthetic log and controls | `data/synthetic/` (made by step 1) | Ground truth for the importance: the location of `B` must be found in `synthetic` and nothing in the controls. |
| Simulated logs | `results/selection/simulated/` (made by step 3) | Gold standard for the selection: 1000 cases with a planted location effect of `X` and of the pair `P,Q`, and an existence effect of `Y`. |
| IMPresseD test log | `data/impressed_2023_test_log/` | A small separate log to test an IMPresseD implementation by hand (see its own README). Not used by the pipeline. |

All logs are CSV files with one row per event and the columns `case:concept:name`,
`concept:name` (activity), `event_nr` (position in the case) and `label` (binary outcome, the same
on every event of a case), plus case attributes for the case-distance function of IMPresseD
(BPIC11: `Age`, `Diagnosis`, `Diagnosis code`, `Specialism code`, `Treatment code`).

**Cleaning** (as in the reference code, `location_permutation/log.py`): activity names are written
in lower case without spaces, `-` and `_`, and cases with an activity that occurs only once in the
whole log are removed. Every step uses the cleaned log.

## Configuration (`config.yaml`)

| Section | Parameter | Value | Meaning |
|---|---|---|---|
| general | `seed`, `test_size` | 42, 0.2 | Random seed of everything; share of cases kept out of the selection when `select_on: training_split`. |
| `synthetic_data` | `events`, `label_noise` | 100, 0.1 | Size of the synthetic log; chance that a label is flipped. |
| `datasets` | `name`, `path`, `case_attributes` | | The event logs that steps 3-5 use. |
| `strategies` | `type` | `apriori` or `impressed` | Which selection strategy. |
| | `select_on` | Apriori `all`, IMPresseD `training_split` | Apriori does not use the labels, so it selects on all cases. IMPresseD uses the labels, so it only sees 80% of the cases. |
| | `importance` | `true` / `false` | Whether step 4 measures the importance of these itemsets. `apriori_support0.2_top25` is a sensitivity check of the selection only. |
| | `min_support`, `top_k`, `max_length` | 0.5, 10, 3 | Apriori: the defaults of the authors' code. |
| | `max_pattern_length` | 3 | IMPresseD: pattern lengths 1, 2 and 3, as the assignment asks. |
| `simulation` | `repetitions`, `scenarios` | 10, 2 | Simulated logs per scenario, with the pair `P,Q` in 30% or 60% of the cases. |
| `importance` | `folds`, `repetitions` | 5, 10 | As in the paper. |
| | `constrain` | `true` | Feasibility: an activity only moves to positions where it was observed in the log. |
| | `evaluate_on` | `[train, test]` | Measure the F1 drop on the training cases of each fold (as the reference code) and/or on the held-out test cases. See below. |
| | `exact_tracking`, `accumulate_permutations` | `false`, `false` | Two deviations of the reference code, see below. |
| `comparison` | `reference`, `variant`, `datasets`, `top_k` | apriori, impressed, BPIC11 f1-f3, 10 | What step 5 compares. |

## Output (`results/`)

| File | Content |
|---|---|
| `synthetic_checks.csv` | Step 2: Spearman correlation between the position of `B` and the label per synthetic log. |
| `selection/summary.csv` | Step 3: per dataset (and simulated log) the size of the log, the number of itemsets per strategy and size, the overlap with the first strategy, and for simulated logs the recall of the planted itemsets. |
| `selection/summary_simulated_mean.csv` | The simulated rows averaged per scenario. |
| `selection/<dataset>/<strategy>_itemsets.json` | The selected itemsets (input of step 4). |
| `selection/<dataset>/<strategy>_details.csv` | Why they were selected, e.g. the IMPresseD patterns with their three interest values. |
| `selection/simulated/<log>.csv`, `<log>_gold.json` | The simulated logs and their gold standard. |
| `importance/<dataset>/evaluated_on_<train\|test>/<strategy>_summary.csv` | Step 4: mean and standard deviation of the F1 drop per itemset, most important first. |
| `.../<strategy>_location_importance.csv` | Every F1 drop: one column per itemset, one row per fold and repetition (5 x 10 = 50). |
| `.../<strategy>_fold_scores.csv` | Train and test F1 of the model in each fold, without permutation. |
| `.../<strategy>_boxplot.png` | Box plot of the 20 most important itemsets. |
| `comparison/comparison_metrics.csv` | Step 5: per dataset and setting the number of itemsets per strategy, shared itemsets, overlap of the top 10, rank correlation (Spearman) on the shared itemsets, best itemset per strategy, and how many IMPresseD-only itemsets beat the best Apriori itemset. |
| `comparison/<dataset>_<train\|test>_top10.csv` | The 10 most important itemsets of both strategies side by side. |
| `comparison/comparison_<train\|test>.png` | The most important itemsets of both strategies per dataset, with their mean F1 drop and standard deviation. |

## Expected results

With the settings of `config.yaml` (seed 42), `python main.py` gives these numbers.

Step 2 (`synthetic_checks.csv`): the planted effect is present only in the synthetic log.

| log | cases with B | Spearman | p-value |
|---|---|---|---|
| synthetic | 15/15 | 0.745 | 0.0014 |
| shuffled_within_case | 15/15 | 0.016 | 0.96 |
| shuffled_outcomes | 15/15 | 0.308 | 0.26 |
| shuffled_across_traces | 10/15 | -0.305 | 0.39 |

Steps 4 and 5 on BPIC11 (`comparison/comparison_metrics.csv`), mean F1 drop of the best itemset:

| | f1 train | f2 train | f3 train | f1 test | f2 test | f3 test |
|---|---|---|---|---|---|---|
| Itemsets Apriori / IMPresseD | 174 / 32 | 217 / 62 | 161 / 29 | 174 / 32 | 217 / 62 | 161 / 29 |
| Shared itemsets | 9 | 8 | 7 | 9 | 8 | 7 |
| Spearman on shared itemsets | 1.00 | 1.00 | 1.00 | 0.73 | 1.00 | 0.96 |
| Best Apriori | ac370000 (0.057) | 376400 (0.137) | ac370000 (0.133) | ac370000, ac370442, ac370443 (0.007) | 376400 (0.118) | ac370000 (0.084) |
| Best IMPresseD | ac370000 (0.058) | 376400, 378619a, ac370000 (0.168) | ac370000 (0.134) | 376400, ac370000 (0.005) | 376400, 378619a (0.131) | ac370000 (0.088) |

Mean test F1 of the model without permutation: 0.907 (f1), 0.895 (f2), 0.934 (f3).

## Design decisions

### Reproduction of the reference code

`location_permutation/` reimplements the reference code of the authors
(<https://github.com/MozhganVD/PermutationLocationImportance>): the same log cleaning, the
index-based one-hot encoding, XGBoost with default parameters, 5-fold stratified cross-validation,
10 repetitions, weighted F1, and the authors' `shuffle_sequence` / `find_itemset_indexes`
permutation (feasibility and order constraint). Single activities use the reference
`trace_permutation_importance`, itemsets with 2 or 3 activities the itemset permutation. The
permutation was checked against the reference function: with the same random draws, 300 of 300
shuffles were identical.

Three settings in `config.yaml` concern places where the reference code differs from the paper:

| Setting | Default | Why |
|---|---|---|
| `evaluate_on` | `[train, test]` | The reference code measures the F1 drop on the **training** cases of each fold. The model fits these cases almost perfectly (train F1 ≈ 0.99), so almost any permutation lowers F1: on the label-shuffled control the top "importance" is still about 0.04. On the **test** cases the controls stay near 0, which fits the paper text better. We report both. |
| `exact_tracking` | `false` | The reference `shuffle_sequence` has an off-by-one in its position bookkeeping, so the order inside an itemset is not always kept. `true` fixes this with the same random draws. |
| `accumulate_permutations` | `false` | The reference code never resets the permuted traces, so shuffles pile up over itemsets and repetitions. `false` starts every repetition from the original traces, as the paper describes. |

Other differences: `random_state` is fixed everywhere (folds, model), the random draws come
from `numpy.random.default_rng`, and unchanged cases are not predicted again (same result, faster).

### IMPresseD variant

* **Totally ordered traces.** IMPresseD works on partially ordered traces. The BPIC11 logs contain
  no information about concurrency, so every trace is totally ordered. The rule for concurrent
  events therefore never applies, and the four remaining rules are directly/eventually
  following/preceding. An *eventual* relation needs at least one event in between.
* **Patterns with an eventual relation are not extended further** (`extend_eventual_patterns: false`),
  as in the reference implementation of the IMPresseD authors.
* **Outcome interest** is the information gain between the label and the per-case instance count of
  a pattern (scikit-learn `mutual_info_score`). **Case distance** is the mean distance between cases
  with and without the pattern; the distance of two cases is `(normalised Euclidean + m * Jaccard) / (m + 1)`
  over their case attributes, as defined in the IMPresseD paper (`m` = number of categorical attributes).
* **No time gap constraint** between the events of a pattern.
* **Activity sets.** The assignment asks for activity sets, so the ordering relations are dropped:
  A→B, B→A and A⇒B all give {A, B}, and duplicates are removed.

### Apriori baseline

The paper analyses all single activities and the top 10 frequent itemsets with more than one
activity. The default `min_support` of 0.5 is the default of the authors' code. Because the
IMPresseD variant is limited to length 3, `max_length` is 3 for Apriori as well.

## Project structure

```
main.py                         the only entry point: runs the pipeline (see "How the pipeline works")
config.yaml                     all parameters
requirements.txt                exact package versions
pipeline/
    runner.py                   Pipeline: the five steps and the file handling between them
synthetic_data/
    generator.py                synthetic log with a planted location effect of B
    controls.py                 the three shuffled controls
itemset_selection/
    event_log.py                EventLog: traces, labels, case attributes for the strategies
    selector.py                 ItemsetSelector interface, SelectionResult
    apriori_selector.py         baseline strategy
    impressed_selector.py       IMPresseD strategy (pattern discovery loop)
    pattern.py                  patterns, instances and the four extension rules
    interest.py                 the three interest functions
    case_distance.py            distance between cases for the case-distance interest
    pareto.py                   Pareto front
    simulation.py               simulated logs with a planted gold standard
location_permutation/
    log.py                      PreparedLog: cleaning and index-based encoding (as the reference code)
    permuter.py                 the location permutation of the reference code
    importance.py               cross-validated location permutation importance (XGBoost, F1 drop)
    reporting.py                summary tables and box plots
evaluation/
    control_checks.py           step 2
    selection_metrics.py        overlap of selections, recall against the gold standard
    strategy_comparison.py      step 5
data/                           BPIC11 logs, synthetic/ (step 1), impressed_2023_test_log/
tests/                          unit tests and an end-to-end test of the pipeline (python -m pytest)
results/                        all output (not in git; made by python main.py)
```

## References

* Vazifehdoostirani, M., Abbaspour Onari, M., Grau, I., Genga, L., Dijkman, R. (2024).
  Uncovering the Hidden Significance of Activities Location in Predictive Process
  Monitoring. ICPM 2023 Workshops, LNBIP 503. <https://doi.org/10.1007/978-3-031-56107-8_15>
* Vazifehdoostirani, M., Genga, L., Lu, X., Verhoeven, R., van Laarhoven, H., Dijkman, R.
  (2023). Interactive Multi-interest Process Pattern Discovery. BPM 2023, LNCS 14159.
  <https://doi.org/10.1007/978-3-031-41620-0_18>
