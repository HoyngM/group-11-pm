# Location permutation importance with Apriori and IMPresseD itemsets

Group 11, Process Mining (JM0211): *Explainable Predictive Process Monitoring*.

This folder reproduces the location permutation importance of Vazifehdoostirani et al. (2024),
*Uncovering the Hidden Significance of Activities Location in Predictive Process Monitoring*,
and adds a variant for choosing which activity sets (*itemsets*) are tested:

| Strategy | Source | Idea |
|---|---|---|
| `apriori` (baseline) | Vazifehdoostirani et al. (2024) | Every single activity plus the `top_k` most frequent itemsets (Apriori, `mlxtend`, as in the authors' code). |
| `impressed` (variant) | Vazifehdoostirani et al. (2023), IMPresseD | Port of the automatic mode of the authors' IMPresseD tool: score patterns on outcome interest, frequency and case distance, keep the Pareto front, extend those patterns by one event, repeat up to length 3. Ordering relations are dropped afterwards. |

For every itemset, the location importance is the drop in weighted F1 of an XGBoost model when the
location of the itemset is permuted (5-fold cross-validation, 10 repetitions). The code follows
the authors' repositories, <https://github.com/MozhganVD/PermutationLocationImportance> and
<https://github.com/MozhganVD/InteractivePatternDetection>: the same file, class and function names
(in PEP 8 style), with our additions for the assignment in `extensions/`. Event logs are loaded,
cleaned and queried with [pm4py](https://pm4py.fit.fraunhofer.de/). Every difference with the
authors' code is listed in [`DIFFERENCES.md`](./DIFFERENCES.md).

## Installation

Python 3.10 or newer. From the folder that contains this README (`Project/code`):

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Tested with Python 3.14 and pm4py 2.7.23, mlxtend 0.25.0, paretoset 1.2.5, networkx 3.7,
xgboost 3.4.1, scikit-learn 1.9.1, pandas 3.1.0, numpy 2.5.4, scipy 1.18.1, matplotlib 3.11.2.

## Running

```bash
python main.py
```

This is the only entry point. It reads `config.yaml` and, for each of the six datasets
(three BPIC11 logs, the synthetic log and two shuffled controls) and each of the three
strategies, selects the itemsets, measures their location importance and writes the results
to `results/`. It then runs the selection on the simulated logs and compares it with the
planted itemsets. The full run is long: on a laptop one BPIC11 log takes 7 to 11 minutes with
`apriori` and 25 minutes or more with `impressed`; with the sensitivity setting
`apriori_support0.2_top25` and laptop sleep our run took about 67 hours of wall-clock time. Use
`--datasets` and `--strategies` to run parts. Options:

```bash
python main.py --datasets synthetic                                  # selected datasets or scenarios
python main.py --strategies apriori impressed                        # selected strategies
python main.py --datasets BPIC11_f1 --n-repeats 2 --max-itemsets 20  # quick look
python main.py --k-fold 3 --output-directory results_test            # other settings / folder
python -m pytest                                                     # unit tests (about 30 s)
```

All parameters are in `config.yaml`; nothing has to be changed in the code. The data are in
`data/`: the BPIC11 logs come from the reference repository, and `data/synthetic/` is created by
`python generate_data.py` (already included; only needed to regenerate it, see below).

### Output

| File | Content |
|---|---|
| `results/summary.csv` | One row per dataset and strategy: log size, number and sizes of the itemsets, overlap with the `apriori` selection, mean train/test F1 of the model, the most important itemset. |
| `results/<dataset>/<strategy>_itemsets.json` | The selected itemsets. |
| `results/<dataset>/<strategy>_details.csv` | Why they were selected: support (Apriori) or the patterns with their three interest values (IMPresseD). |
| `results/<dataset>/<strategy>_location_importance.csv` | All importance values: one column per itemset, 50 rows (5 folds x 10 repetitions), as in the reference output. |
| `results/<dataset>/<strategy>_summary.csv` | Mean and standard deviation of the importance per itemset, most important first. |
| `results/<dataset>/<strategy>_boxplot.png` | Box plot of the 20 most important itemsets. |
| `results/<dataset>/<strategy>_fold_scores.csv` | Train and test F1 of the model per fold. |
| `results/<dataset>/<strategy>_excluded_itemsets.json` | Selected itemsets that were not permuted because they contain a fixed-location activity (see below). |
| `results/simulated/summary_simulated_mean.csv` | Recall of the planted itemsets per scenario and strategy, averaged over 10 simulated logs. |

## Code and how it maps to the authors' repositories

| Our file | Authors' file | Content |
|---|---|---|
| `main.py` | — | The only entry point; class `Experiment` runs every dataset and strategy of `config.yaml`. |
| `tools.py` | `tools.py` (permutation repo) | Class `DataManager`: `_load_df`, `frequent_activity_sets`, `split_test_train`, `cross_split_test_train`, `index_encoding`, `trace_permutation_importance`, `itemset_permutation_importance`, `shuffle_sequence`, `find_itemset_indexes`. |
| `cross_validation_process_permutation.py` | `CrossValidation_ProcessPermutation.py` | The k-fold loop: XGBoost per fold, F1 drop per itemset. |
| `plotting_results.py` | `Plotting_results.py` | Summary tables and box plots. |
| `imipd.py` | `IMIPD.py`, `tools.py` (IMPresseD repo) | `create_patient_data`, `calculate_pairwise_case_distance`, `create_pattern_attributes` and the three `*_measuring_patterns`, `trace_graph_generator`, `create_embedded_pattern_in_trace`, `update_pattern_dict`, `pattern_extension`, `single_pattern_extender`. |
| `auto_impid.py` | `Auto_IMPID.py` | `auto_stepwise_ppd`: the automatic pattern discovery. |
| `extensions/itemset_selection.py` | — | `AprioriSelector` and `ImpressedSelector` behind one interface. |
| `extensions/synthetic_log.py`, `shuffle_controls.py` | — | Synthetic log with a planted location effect and its shuffled controls. |
| `extensions/simulation.py`, `evaluation.py` | — | Simulated logs with a gold standard; overlap and recall of the selections. |
| `generate_data.py` | — | (Re)creates `data/synthetic/`. |

The runtime flow of `main.py` is shown in [`main_flowchart.mmd`](./main_flowchart.mmd).

## Method and implementation choices

The main points; [`DIFFERENCES.md`](./DIFFERENCES.md) has the complete list with reasons.

### Event log (`tools.py`, `DataManager._load_df`)

The CSV is loaded into a pm4py dataframe (`pm4py.format_dataframe`). Cleaning follows the reference:
activity names are lower-cased and stripped of spaces, `-` and `_`, and cases with an activity that
occurs fewer than twice in the log are removed (`pm4py.get_event_attribute_values`,
`pm4py.filter_event_attribute_values`). Traces come from `pm4py.project_on_event_attribute` and the
observed locations of every activity (`allowed_locations`, the feasible targets of a permutation)
from `pm4py.get_activity_position_summary`.

The BPIC11 timestamps only have a date: 89% of consecutive events in a case share their timestamp.
The traces are therefore treated as **totally ordered by `event_nr`** (the event number is used as
the time), so there is no concurrency.

### Apriori (`tools.py`, `DataManager.frequent_activity_sets`)

The multi-activity itemsets are chosen exactly as in the reference (`mlxtend` Apriori, `min_support`
0.5, `top_k` 10). As in the paper, every single activity is analysed as well. `max_length` is 3, the
same as the longest IMPresseD pattern. The setting with support 0.2 and the top 25 itemsets is run as a
sensitivity check. Apriori ignores the labels and selects on all cases.

### IMPresseD (`auto_impid.py`, `imipd.py`)

A port of the automatic mode of the authors' tool with the same interest functions (information gain
via `mutual_info_classif`, case support, mean case distance between cases with and without the
pattern), the same `paretoset` selection and the same extension rules. Patterns with an eventual
relation are not extended further, as in the tool. The selection uses the training part of a
stratified 80/20 split (seed 42), because the outcome interest uses the labels. Differences with the tool:

* **Total order.** The concurrency rule never applies (see above) and is left out.
* **No "direct context" rule.** Every step adds exactly one event: `max_extension_step: 2` gives
  pattern lengths 1, 2 and 3.
* **Case distance on the case attributes only** (`Age`, `Diagnosis`, `Diagnosis code`,
  `Specialism code`, `Treatment code`), as defined in the paper. The tool also counts every
  per-activity frequency column as a categorical attribute.
* **Pattern identity** is checked with an exact key for chain patterns instead of a graph isomorphism
  test against every known pattern. The result is the same; it is much faster.
* `max_gap_between_events` (the tool asks the user for it) is 1000, i.e. no limit.

### Location permutation importance (`cross_validation_process_permutation.py`, `tools.py`)

Index-based one-hot encoding, XGBoost with default parameters, 5-fold stratified cross-validation,
10 repetitions, weighted F1, permutation seed 2023, feasibility constraint (only observed locations).
Single activities use `trace_permutation_importance` (importance 0 when fewer than 3 cases change),
larger itemsets `itemset_permutation_importance` with `shuffle_sequence`, which keeps the order of
the activities of an itemset. Following the method as described in the papers, we correct three
implementation issues of the authors' code (DIFFERENCES.md, P4 to P6):

* The F1 drop is measured on the **held-out test cases** of each fold, not on the training cases,
  which the model has largely memorised.
* Every itemset and repetition starts from the **original traces**; in the authors' code the
  permutations pile up over itemsets and repetitions.
* The **order** inside an itemset is always kept; a bookkeeping error in the authors' shuffle
  sometimes moved the wrong event and broke the order.

The fold split is seeded as well (`seed` 42; the authors' code sets none), so that the results are
reproducible.

**Fixed-location activities.** An activity whose every occurrence is at the same position, counted
from the start (e.g. `register`, always first) or from the end (e.g. `close`, always last, whatever the
trace length), cannot have an important location, and moving it would create unrealistic traces. Such
activities are detected from the data (`DataManager.fixed_location_activities`), and every itemset that
contains one is left out of the permutation (`exclude_fixed_locations` in `config.yaml`; the excluded
itemsets are listed in `<strategy>_excluded_itemsets.json`). On BPIC11 this only concerns a few rare
activities with one or two events, which the feasibility constraint could not move anyway.

On a synthetic log of 200 cases (unit tests) the planted activity `b` gets a mean F1
drop of about 0.43 and a noise activity about 0.09, while in the two shuffled controls no itemset
gets more than 0.1 in absolute value.

### Check against the authors' code

Before the three corrections, the port was compared with the authors' own functions, run on the same
data: the observed locations, the case order and the Apriori itemsets on BPIC11 f1 are identical; with
the same random draws 355 of 355 itemset shuffles and 218 of 218 activity shuffles are identical; and
the IMPresseD Pareto patterns are identical on the synthetic log (18 patterns) and on a simulated log
of 300 cases (109 patterns). With the corrected shuffle, 330 of the 355 itemset shuffles are still
identical; in the other 25 the authors' shuffle moved a wrong event, and in 9 of those it broke the
order of the itemset.

## Synthetic data and negative controls

`data/synthetic/synthetic.csv` (100 events, 15 cases) contains a planted location effect: activity `B`
occurs in *every* case and only its location differs, early (label 0) or late (label 1), with 10% label
noise. Apriori cannot see this (the frequency of `B` is 100%); a location-aware method can. Two shuffled
copies serve as negative controls: `shuffled_within_case.csv` (activities permuted inside each case, so
location is gone) and `shuffled_outcomes.csv` (labels permuted over the cases). The unit tests check on a
larger log of 200 cases that the location of `B` correlates with the label only in the original log.

```bash
python generate_data.py                       # regenerate data/synthetic/ (seed 42, 100 events)
python generate_data.py --cases 400 --seed 7  # a larger log, written to the same folder
```

The simulated logs (`extensions/simulation.py`, section `simulation` in `config.yaml`) plant two
location effects (`x`, and the pair `p q`) and one existence effect (`y`) in 1000 cases. They test
whether a strategy selects the planted itemsets at all.

## Results

**The location importance must be rerun** with the corrected permutation (previous section); the
importance values of the previous run (F1 drop on training cases, permutations piling up) are no longer
reported here. The selection results and the model scores below do not depend on the permutation and
stay valid. All numbers are in `results/summary.csv` and `results/<dataset>/`.

**Itemsets and model.**

| Dataset | Cases | Model F1 train / test | Itemsets apriori (sizes 1/2/3) | Itemsets impressed (sizes 1/2/3) | Overlap of 2+ itemsets (Jaccard) |
|---|---|---|---|---|---|
| BPIC11 f1 | 1130 | 0.994 / 0.907 | 174 (164/7/3) | 154 (18/88/48) | 0.02 |
| BPIC11 f2 | 1130 | 0.994 / 0.895 | 217 (207/7/3) | 176 (35/83/58) | 0.02 |
| BPIC11 f3 | 1111 | 0.995 / 0.934 | 161 (156/4/1) | 142 (22/82/38) | 0.00 |

Apriori and IMPresseD select almost entirely different multi-activity itemsets.

**Synthetic log and controls (15 cases).** The model's test F1 is 0.65 on the synthetic log and 0.39 /
0.42 on the two controls: the planted location effect is learnable, the controls carry no signal.

**Simulated logs (10 logs per scenario).** All strategies find the existence effect `y` in every log.
The location effect `x` is selected in every log, except by IMPresseD in the rare-pair scenario (8 of
10 logs). No strategy selects the planted pair `{p, q}` in any log: selection on frequency (Apriori)
or on the presence of patterns (IMPresseD) cannot see a pair whose location, not its presence, matters.
IMPresseD selects fewer pure-noise itemsets (share 0.50 against 0.73 for Apriori).

## Project structure

```
main.py                                   the only entry point (class Experiment)
config.yaml                               all parameters
tools.py                                  DataManager (log, Apriori, encoding, permutation)
cross_validation_process_permutation.py   k-fold location permutation importance
plotting_results.py                       tables and box plots
imipd.py                                  IMPresseD functions (interests, pattern extension)
auto_impid.py                             automatic IMPresseD discovery
generate_data.py                          (re)creates data/synthetic/
requirements.txt
DIFFERENCES.md                            differences with the authors' repositories
data/                                     BPIC11 logs, synthetic/ (synthetic log and two controls)
results/                                  all outputs (ignored by git)
extensions/                               our additions for the assignment
    itemset_selection.py                  AprioriSelector, ImpressedSelector
    synthetic_log.py, shuffle_controls.py synthetic log and negative controls
    simulation.py, evaluation.py          simulated logs, gold standard, overlap and recall
tests/                                    unit tests (python -m pytest)
```

## References

* Vazifehdoostirani, M., Abbaspour Onari, M., Grau, I., Genga, L., Dijkman, R. (2024).
  Uncovering the Hidden Significance of Activities Location in Predictive Process
  Monitoring. ICPM 2023 Workshops, LNBIP 503. <https://doi.org/10.1007/978-3-031-56107-8_15>
* Vazifehdoostirani, M., Genga, L., Lu, X., Verhoeven, R., van Laarhoven, H., Dijkman, R.
  (2023). Interactive Multi-interest Process Pattern Discovery. BPM 2023, LNCS 14159.
  <https://doi.org/10.1007/978-3-031-41620-0_18>
