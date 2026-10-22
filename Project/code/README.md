# Location permutation importance with Apriori and IMPresseD itemsets

Group 11, Process Mining (JM0211): *Explainable Predictive Process Monitoring*.

This folder contains the code of the reproduction of
Vazifehdoostirani et al. (2024), *Uncovering the Hidden Significance of Activities
Location in Predictive Process Monitoring*. The method measures how much the
location of a set of activities (an *itemset*) matters for an outcome prediction
model. Which itemsets to test is decided by a selection strategy. Two strategies
are implemented here, and both return plain activity sets:

| Strategy | Source | Idea |
|---|---|---|
| `apriori` (baseline) | Vazifehdoostirani et al. (2024) | Every single activity plus the `top_k` most frequent itemsets found with Apriori. |
| `impressed` (variant) | Vazifehdoostirani et al. (2023), IMPresseD | Start with single activities, keep the Pareto front of three interest functions (case coverage, outcome information gain, case distance), extend those patterns by one activity, repeat up to length 3. Ordering relations are dropped afterwards. |

The itemset selection is run with `main.py`. The location permutation and the prediction
model (XGBoost) are in `location_permutation/` and run with `run_location_importance.py`
(see "Synthetic data and location permutation importance" below). That script is a
reimplementation of the reference code of the authors; check it against the pipeline of
the other group members before submitting, so that only one version is left.

## Installation

Python 3.10 or newer is required. From the folder that contains this README:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows (PowerShell): .venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Running the experiments

```bash
python main.py
```

This reads `config.yaml`, runs both strategies (and a second Apriori setting) on

* the three BPIC11 logs in `data/` (taken from
  <https://github.com/MozhganVD/PermutationLocationImportance/tree/master/datasets>), and
* two simulated scenarios (10 random logs each) in which the location of
  activities determines the outcome,

and writes everything to `results/` (about 1 minute). Useful options:

```bash
python main.py --datasets BPIC11_f1                 # only selected datasets or scenarios
python main.py --config my_config.yaml              # other parameters
python main.py --output-directory results_test      # other output folder
```

All parameters (support, `top_k`, pattern length, case attributes, number of
simulated logs, seeds, ...) are in `config.yaml`; nothing needs to be edited in
the code. Itemsets are always selected on the training part of a stratified
train/test split (`test_size`, `seed`), so that the labels of the test cases do not
influence which itemsets are chosen.

### Output

| File | Content |
|---|---|
| `results/summary.csv` | One row per dataset (and per simulated log): sizes of the selections, overlap between strategies, recall against the gold standard for simulated logs. |
| `results/summary_simulated_mean.csv` | The simulated rows averaged per scenario. |
| `results/<dataset>/<strategy>_itemsets.json` | The selected itemsets: input for the permutation importance. |
| `results/<dataset>/<strategy>_details.csv` | Why they were selected, e.g. the patterns with their three interest values. |
| `results/simulated/<log>.csv`, `<log>_gold.json` | The simulated event logs and which itemsets we planted. |

## Using a strategy from other code

```python
from itemset_selection import EventLog, ImpressedItemsetSelector

log = EventLog.from_csv(
    "data/BPIC11_f1_trunc36.csv",
    numeric_attributes=["Age"],
    categorical_attributes=["Diagnosis", "Diagnosis code", "Specialism code", "Treatment code"],
)
train, test = log.stratified_split(test_size=0.2, seed=42)

result = ImpressedItemsetSelector(max_pattern_length=3, numeric_attributes=["Age"],
                                  categorical_attributes=["Diagnosis"]).select(train)
for itemset in result.itemsets:       # list of frozensets of activity names
    print(sorted(itemset))
```

`AprioriItemsetSelector` has the same interface (`select(log) -> SelectionResult`).

## Synthetic data and location permutation importance

This part belongs to the evaluation: a small synthetic log with a planted location effect, shuffled negative
controls, and the 5-fold location permutation importance that compares the Apriori and IMPresseD itemsets.
The commands below are run from this folder, after the installation above.

### Idea

A shuffled copy of existing data is **not** proof that the algorithm works: shuffling destroys the
structure the algorithm looks for, so a correct implementation should find (almost) nothing there.
We therefore use two things:

1. **A synthetic log with a planted pattern** (ground truth). Activity `B` occurs in *every* case, only its
   location differs: early in the trace gives label 0, late gives label 1 (10% label noise). Apriori cannot see
   this (frequency of `B` is 100%); a location-aware method can. The other activities are noise.
2. **Three shuffled copies as negative controls**, in exactly the same format:

| File | Shuffle | What is destroyed |
|---|---|---|
| `data/synthetic/synthetic.csv` | none | (original, planted effect present) |
| `data/synthetic/shuffled_within_case.csv` | activities permuted inside each case | order and location |
| `data/synthetic/shuffled_outcomes.csv` | labels permuted over the cases | link between trace and outcome |
| `data/synthetic/shuffled_across_traces.csv` | activity column permuted over all events | patterns and activity sets |

The column layout (`case:concept:name`, `concept:name`, `time:timestamp`, `event_nr`, `Age`, `Category`,
`label`) matches the BPIC11 files used by `data/BPIC11_*.csv`, so the same code runs on these files without changes.

### Running the synthetic data and the checks

```bash
python generate_data.py            # writes the four CSV files to data/synthetic/
python check_controls.py           # planted effect present in synthetic, absent in the controls
python -m pytest                   # all unit tests
```

Options: `python generate_data.py --events 200 --seed 7 --label-noise 0.2 --output-directory other_folder`.

#### Expected output of `check_controls.py` (seed 42, 15 cases = 100 events)

Spearman correlation between the relative position of `B` in a case and the label:

| log | cases with B | Spearman | p-value |
|---|---|---|---|
| synthetic | 15/15 | 0.745 | 0.0014 |
| shuffled_within_case | 15/15 | 0.016 | 0.96 |
| shuffled_outcomes | 15/15 | 0.308 | 0.26 |
| shuffled_across_traces | 10/15 | -0.305 | 0.39 |

### Location permutation importance (BPIC11 and synthetic logs)

`location_permutation/` reimplements the pipeline of Vazifehdoostirani et al. (2024) after the reference code
(<https://github.com/MozhganVD/PermutationLocationImportance>): same log cleaning, index-based one-hot encoding,
XGBoost with default parameters, 5-fold stratified cross-validation, 10 repetitions, weighted F1, and the
authors' `shuffle_sequence` / `find_itemset_indexes` permutation (feasibility + order constraint). Single
activities use the reference `trace_permutation_importance`, itemsets with 2 or 3 activities use the itemset
permutation. The itemsets come from the group's Apriori and IMPresseD selectors
(`itemset_selection/`).

```bash
python run_location_importance.py                                   # all datasets in importance_config.yaml
python run_location_importance.py --datasets synthetic              # selected datasets only
python run_location_importance.py --datasets BPIC11_f1 --repetitions 2 --max-itemsets 30   # quick look
python run_location_importance.py --evaluate-on test                # F1 drop on the held-out cases
```

Output per dataset in `results/importance/<dataset>/evaluated_on_<train|test>/`: `<strategy>_location_importance.csv`
(one column per itemset, 50 rows = 5 folds x 10 repetitions), `<strategy>_summary.csv`, `<strategy>_boxplot.png`,
`<strategy>_fold_scores.csv`. The unit tests compare nothing with the reference output, but the permutation was
checked once against the reference function: with the same random draws, 300 of 300 shuffles were identical.

#### Choices that are not obvious (`importance_config.yaml`)

| Setting | Default | Why |
|---|---|---|
| `evaluate_on` | `train` | The reference code measures the F1 drop on the **training** cases of each fold. The model has memorised them (train F1 = 1.0), so any permutation lowers F1: in the label-shuffled control the top "importance" is 0.38. `test` gives the clean picture (about 0 on the controls) and fits the paper text better. Report which one you use. |
| `exact_tracking` | `false` | The reference `shuffle_sequence` has an off-by-one in its position bookkeeping, so the order inside an itemset is not always kept. `true` fixes this, with the same random draws. |
| `accumulate_permutations` | `false` | The reference code never resets the permuted traces, so shuffles pile up over itemsets and repetitions. `false` starts every repetition from the original traces, as the paper describes. |
| `select_on` | Apriori `all`, IMPresseD `training_split` | Apriori ignores the labels. IMPresseD uses them, so it selects on 80% of the cases (open question 5 in `PLAN.md`). |

Other differences with the reference: `random_state` is fixed everywhere (folds, model), the random draws come
from `numpy.random.default_rng`, and unchanged cases are not predicted again (same result, faster).

#### Results (seed 42, 5 folds x 10 repetitions)

Top itemset by mean F1 drop:

| dataset | evaluated on | Apriori | IMPresseD |
|---|---|---|---|
| BPIC11 f1 | train | ac370000 (0.057) | ac370000 (0.058) |
| BPIC11 f1 | test | ac370000, ac370442, ac370443 (0.007) | 376400, ac370000 (0.005) |
| synthetic (100 events) | train | b, close, register (0.240) | b (0.228) |
| synthetic (100 events) | test | b, register (0.228) | b, register (0.231) |
| shuffled_within_case | train | b, f, register (0.064) | b, f (0.059) |
| shuffled_within_case | test | b, close, register (0.003) | b, close, e (0.013) |
| shuffled_outcomes | train | b, close, register (0.039) | b, e (0.042) |
| shuffled_outcomes | test | c (0.047) | b, d (0.020) |

With only 15 cases the result is noisy: the standard deviation over folds and repetitions (0.2-0.3) is as large as the
mean, because every test fold holds 3 cases. The separation is still visible (synthetic 0.23 against at most 0.06 on
the controls). With 400 cases (`python generate_data.py --cases 400`) `b` drops F1 by 0.41 on the test cases and
the controls stay near 0 (top drop at most 0.02); that run takes 2-5 s as well.

On BPIC11 f1 with the reference setting, `ac370000` and itemsets containing it are most important, as in the paper;
`{ac370000, ac379999}` is selected by IMPresseD. On the synthetic log the location of `b` is found; on the
controls nothing is found when evaluating on the test cases.

### Comparing the two strategies

After `run_location_importance.py` has run for the BPIC11 logs (both `--evaluate-on train` and `--evaluate-on test`):

```bash
python compare_strategies.py                                 # BPIC11 f1-f3, train and test
python compare_strategies.py --evaluate-on test --top-k 5    # other settings
```

Output in `results/importance/comparison/`:

| File | Content |
|---|---|
| `comparison_metrics.csv` | Per dataset and setting: number of itemsets per strategy, shared itemsets (count and Jaccard), overlap of the top-k itemsets, Jaccard of the activities in the top-k, Spearman correlation of the mean F1 drop of the shared itemsets, best itemset per strategy, and how many IMPresseD-only itemsets beat the best Apriori itemset. |
| `<dataset>_<train\|test>_top10.csv` | The top-k itemsets of both strategies side by side, with a flag whether the other strategy selected them too. |
| `comparison_<train\|test>.png` | Mean F1 drop (and standard deviation) of the top 8 itemsets of each strategy, one panel per dataset. A missing marker means that strategy did not select the itemset. |

The results of the full runs on BPIC11 f1-f3 (22 Oct), with an explanation of every number and the
conclusions, are in `../contributions/job_oosterkamp/README.md`.

## Tests

```bash
python -m pytest
```

## Project structure

```
main.py                         entry point of the itemset selection experiments
config.yaml                     parameters of main.py
run_location_importance.py      entry point of the location permutation importance
importance_config.yaml          parameters of run_location_importance.py
compare_strategies.py           compares the location importance of the Apriori and IMPresseD itemsets
generate_data.py                synthetic log and shuffled controls -> data/synthetic/
check_controls.py               checks the planted effect in the synthetic log and the controls
data/                           BPIC11 logs, synthetic/ (generated), impressed_2023_test_log/ (22 patients)
results/                        all outputs (ignored by git)
location_permutation/           log preparation, reference permutation, cross-validated importance,
                                comparison of two strategies (comparison.py)
synthetic_data/                 generator with planted effect, shuffle controls
itemset_selection/
    event_log.py                EventLog: traces, labels, case attributes
    selector.py                 ItemsetSelector interface, SelectionResult
    apriori_selector.py         baseline strategy
    impressed_selector.py       IMPresseD strategy (pattern discovery loop)
    pattern.py                  patterns, instances and the four extension rules
    interest.py                 the three interest functions
    case_distance.py            distance between cases for the case-distance interest
    pareto.py                   Pareto front
    simulation.py               simulated logs with a planted location effect
    evaluation.py               overlap between strategies, recall against the gold standard
    experiment.py               runs the configured experiments
tests/                          unit tests
```

## Design decisions and simplifications

* **Totally ordered traces.** IMPresseD works on partially ordered traces. The BPIC11
  logs contain no information about concurrency, so every trace is totally
  ordered. The rule for concurrent events therefore never applies, and the four
  remaining rules are directly/eventually following/preceding. An *eventual*
  relation needs at least one event in between.
* **Patterns with an eventual relation are not extended further** (`extend_eventual_patterns:
  false`), as in the reference implementation of the IMPresseD authors.
* **Outcome interest** is the information gain between the label and the
  per-case instance count of a pattern (scikit-learn `mutual_info_score`).
  **Case distance** is the mean distance between cases with and without the pattern.
  The distance of two cases is `(normalised Euclidean + m * Jaccard) / (m + 1)` over
  their case attributes, as defined in the IMPresseD paper (`m` = number of
  categorical attributes).
* **No time gap constraint** between the events of a pattern.
* **Apriori baseline.** The paper analyses all single activities and the top 10 frequent
  itemsets with more than one activity. The default `min_support` of 0.5 is the
  default of the authors' code. Because the IMPresseD variant is limited to length 3,
  `max_length` is 3 for Apriori as well.

## References

* Vazifehdoostirani, M., Abbaspour Onari, M., Grau, I., Genga, L., Dijkman, R. (2024).
  Uncovering the Hidden Significance of Activities Location in Predictive Process
  Monitoring. ICPM 2023 Workshops, LNBIP 503. <https://doi.org/10.1007/978-3-031-56107-8_15>
* Vazifehdoostirani, M., Genga, L., Lu, X., Verhoeven, R., van Laarhoven, H., Dijkman, R.
  (2023). Interactive Multi-interest Process Pattern Discovery. BPM 2023, LNCS 14159.
  <https://doi.org/10.1007/978-3-031-41620-0_18>
