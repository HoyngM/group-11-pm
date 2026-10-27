# Differences between our code and the authors' repositories

Group 11. This note lists every difference between our implementation (`Project/code`) and the
two repositories of the authors, and why we made it. Use it to check the code and to write the
implementation section of the paper.

| Short name | Repository | Paper | What we use from it |
|---|---|---|---|
| **Permutation repo** | <https://github.com/MozhganVD/PermutationLocationImportance> | Vazifehdoostirani et al. (2024), location permutation importance | Data cleaning, Apriori itemsets, encoding, XGBoost, cross-validation, the permutation |
| **IMPresseD repo** | <https://github.com/MozhganVD/InteractivePatternDetection> | Vazifehdoostirani et al. (2023), IMPresseD | The automatic pattern discovery, used as our itemset selection variant |

**Status labels.** *Same* means the behaviour is identical. *Verified* means we ran the authors' code
and ours on the same data and compared the output. *Deviation* means we deliberately do something
different, with a reason.

## Summary

| # | Topic | Status | Effect on results |
|---|---|---|---|
| S1 | File, class and function names follow the repos (PEP 8 style) | Structure | None |
| S2 | Our additions are in `extensions/`; one entry point `main.py` | Structure | None |
| P1 | Data loading and cleaning (now with pm4py) | Same, verified | None |
| P2 | Apriori itemsets (`mlxtend`) | Same, verified | None |
| P3 | Permutation of activities and itemsets | Same, verified (before P4 to P6) | None |
| P4 | F1 drop measured on the held-out test cases | Deviation (correction) | No inflated importance; controls near 0 |
| P5 | Every repetition starts from the original traces | Deviation (correction) | Itemsets no longer influence each other |
| P6 | Order of an itemset always kept | Deviation (correction) | 25 of 355 BPIC11 shuffles differ |
| P7 | Fold split is seeded | Deviation | Results become reproducible |
| P8 | Single activities and itemsets in one run | Deviation (organisation only) | None |
| P9 | Order in which single activities are permuted | Deviation (small) | Individual values differ slightly |
| P10 | Column order of the encoding | Deviation (small) | Possibly tiny differences in the model |
| P11 | No crash on an impossible permutation | Deviation | None where the reference works |
| P12 | Apriori: max. itemset size 3, extra setting 0.2/25 | Deviation | Fewer large itemsets; one extra run |
| P13 | Datasets | Deviation | BPIC11 f4, BPIC12 and Sepsis are not run |
| P14 | Not ported: other experiments of the repo | Scope | None |
| P15 | Method signatures and return values | Interface | None |
| P16 | Itemsets with a fixed-location activity are not permuted | Addition | Start/end activities no longer tested |
| I1 | Total order instead of timestamps | Deviation | No concurrency patterns |
| I2 | "Direct context" rule off | Deviation | Pattern lengths are exactly 1, 2, 3 |
| I3 | Case distance on case attributes only | Deviation | Different case distance interest |
| I4 | Pattern identity by key instead of isomorphism | Same output, verified | Faster |
| I5 | Fixed settings instead of GUI input | Deviation (choices) | See I5 |
| I6 | Output: activity sets instead of pattern features | Deviation (purpose) | Order of patterns is dropped |
| I7 | Quirks of the tool that we kept | Same (kept on purpose) | See I7 |
| I8 | Function signatures and return values | Interface | None |
| G1 | Things that exist in neither repo | Addition | None on the method |

## Part 0: Structure and names

### S1. File, class and function names follow the repos

Someone who knows the authors' code should recognise it in ours. We therefore use the same file,
class, function and variable names, written in PEP 8 style as the course guidelines require
(`Allowed_locations` -> `allowed_locations`, `Pattern_extension` -> `pattern_extension`,
`CrossValidation_ProcessPermutation.py` -> `cross_validation_process_permutation.py`).

| Our file | Authors' file | Main names |
|---|---|---|
| `tools.py` | `tools.py` (permutation repo) | `DataManager`, `_load_df`, `frequent_activity_sets`, `split_test_train`, `cross_split_test_train`, `index_encoding`, `trace_permutation_importance`, `itemset_permutation_importance`, `shuffle_sequence`, `find_itemset_indexes`, `allowed_locations`, `case_id`, `activity`, `outcome`, `frq_threshold` |
| `cross_validation_process_permutation.py` | `CrossValidation_ProcessPermutation.py` | `k_fold` (`K_fold`), `train_list`, `test_list`, `permutation_importance_train_all` |
| `plotting_results.py` | `Plotting_results.py` | box plot of the importance |
| `imipd.py` | `IMIPD.py`, `tools.py` (IMPresseD repo) | `create_patient_data`, `calculate_pairwise_case_distance`, `create_pattern_frame`, `frequency_measuring_patterns`, `predictive_measuring_patterns`, `similarity_measuring_patterns`, `create_pattern_attributes`, `trace_graph_generator`, `create_embedded_pattern_in_trace`, `update_pattern_dict`, `pattern_extension`, `single_pattern_extender` |
| `auto_impid.py` | `Auto_IMPID.py` | `auto_stepwise_ppd` (`AutoStepWise_PPD`), `max_extension_step`, `max_gap_between_events`, `paretoset_patterns` |

The reference keeps the inner loop that moves one activity inside `trace_permutation_importance`; we
put it in a small helper, `DataManager._shuffle_activity`, so that it can be tested on its own. The
same holds for `_max_possible_locations` (the `max_possibles` part of `shuffle_sequence`) and `_f1_drop`
(the re-encoding and prediction after a permutation).

### S2. Our additions are separate; one entry point

The authors run one script per experiment (`CrossValidation_ProcessPermutation.py`, the IMPresseD GUI).
The course guidelines ask for one main method, so `main.py` is the only entry point: its class
`Experiment` runs all datasets and strategies of `config.yaml`. Everything that exists in neither
repository is in `extensions/` (see G1).

## Part 1: Permutation repo

### P1. Data loading and cleaning: same, verified

| Reference (`DataManager._load_df`) | Ours (`DataManager._load_df` in `tools.py`) |
|---|---|
| `pd.read_csv`, manual sorting by case and `event_nr` | `pm4py.format_dataframe`, with `event_nr` as the time column |
| Lower-case the activity names, remove spaces, `-`, `_` | The same |
| `deviant`/`regular` to 1/0 | The same |
| Count events per activity, drop cases with an activity seen fewer than 2 times | The same, with `pm4py.get_event_attribute_values` and `pm4py.filter_event_attribute_values` |
| `Allowed_locations`: the `event_nr` values per activity | `allowed_locations` from `pm4py.get_activity_position_summary` + 1 (pm4py counts from 0) |
| Traces from repeated `DataFrame` filters | `traces` from `pm4py.project_on_event_attribute` |

*Verified on BPIC11 f1:* the allowed locations and the order of the cases are identical.

Two parts of `_load_df` were not ported because the cross-validation script never uses them:
`L_max` and `indexing_info` (mean and std of each activity's position). The handling of a
`lifecycle:transition` column (keep only `COMPLETE` events) was not ported either; the BPIC11 files
have no such column. A log that does have one would need that step added. Our `DataManager` also
keeps the case attributes (`case_table`), which IMPresseD needs for the case distance.

### P2. Apriori itemsets: same, verified

`DataManager.frequent_activity_sets` does what the reference's does: `mlxtend` `TransactionEncoder` +
`apriori`, sorted by support, keep the `top_k` itemsets with two or more activities, and return them as
lists plus the Apriori table. *Verified on BPIC11 f1* (support 0.5, top 10): the same 10 itemsets.

### P3. The permutation: same, verified

`DataManager.shuffle_sequence` and `find_itemset_indexes` are literal ports, and so is the activity
shuffle of `trace_permutation_importance` (`_shuffle_activity`, see S1). The reference calls
`np.random.seed(2023)` at the start of each importance function and then `np.random.choice`; we create
`np.random.RandomState(2023)` at the same point, which gives the same random draws. The rules are the
same as well: feasibility constraint, importance 0 when fewer than 3 cases change (single activities),
and skipping traces that consist of only the itemset.

*Verified on BPIC11 f1 with the same seed, before the corrections P4 to P6:* 355 of 355 itemset
shuffles and 218 of 218 activity shuffles give exactly the same trace. After the correction of the
order (P6), 330 of the 355 itemset shuffles are still identical (see P6).

P4 to P6 are three places where the reference code does not do what the papers (theirs and ours)
describe. We first ported them literally and verified the port (above), then corrected them, so that
the code matches the method as published and as written in Section 2 of our paper.

### P4. F1 drop measured on the held-out test cases: deviation (correction)

The reference trains the model on the training cases of a fold and measures the F1 drop **on
those same training cases**; the test F1 is only printed. The papers describe a cross-validated
evaluation, so we measure the F1 drop on the **held-out test cases** of each fold, with the test F1
as the baseline.

*Effect.* The model has largely memorised its training cases, so on those cases almost any
permutation lowers F1. With the literal version, on our synthetic log of 200 cases a pure noise
activity got an importance of about 0.32 and the label-shuffled control 0.26 to 0.36, against 0.46
for the planted activity `b`. Measured on the test cases, `b` gets about 0.43, the noise activity
about 0.09, and in both shuffled controls no itemset exceeds 0.1 in absolute value. As a side effect
the importance runs faster: only about a fifth of the cases is re-encoded and predicted.

### P5. Every repetition starts from the original traces: deviation (correction)

In the reference, `sub_data` is changed in place and never restored. A permutation of itemset 2
therefore starts from the traces as itemset 1 and all its repetitions left them, so the importance of
an itemset depends on which itemsets came before it. We create a fresh copy of the original traces
(`sub_data`) for every itemset and repetition, so that every itemset is permuted independently, as the
method describes. As in the reference, within a repetition only the shuffled cases get a new encoding;
the others keep their original prediction (`Shuffled_X` starts as a copy of the original `X`).

### P6. Order of an itemset always kept: deviation (correction)

The reference's `shuffle_sequence` tracks where each event is after earlier moves with a dictionary
that is off by one when an event moves backwards. The next event of the itemset can then be the wrong
one, and the order inside the itemset is not always kept, although the paper describes order
preservation. We track every event by its original position instead, with the same random draws.
On BPIC11 f1 (355 itemset shuffles) the reference moved a wrong event in 25 shuffles and broke the
order in 9 of them; our version keeps the order in all of them. A unit test
(`test_itemset_shuffle_keeps_the_order_of_the_itemset`) checks this.

### P7. The fold split is seeded: deviation

The reference's `cross_split_test_train` uses `StratifiedKFold(shuffle=True)` without `random_state`,
so every run gives different folds and different numbers. Ours takes a `seed` (42, from `config.yaml`)
so the results in the paper can be reproduced, which the course guidelines require. The model is
`XGBClassifier()` with default parameters in both; its default is already deterministic.

### P8. Single activities and itemsets in one run: deviation in organisation only

The reference has a flag `--Multi_activity`: `True` runs the itemsets with 2+ activities
(`itemset_permutation_importance`), `False` runs every single activity
(`trace_permutation_importance`). These are two separate program runs. Our
`CrossValidationProcessPermutation.run` calls both in every fold: first the single activities, then
the larger itemsets, each starting from the original traces with a fresh random state with seed 2023.
That is equivalent to the two reference runs.

A side note on the reference's command line: `type=bool` in `argparse` turns any text into `True`,
so `--constrain False` does not switch the constraint off there. We use `config.yaml` instead.

### P9. Order of the single activities: small deviation

The reference permutes the single activities in order of first appearance in the whole log
(`data[activity].unique()`) and includes every activity. We permute the single activities that
the strategy selected, in alphabetical order. For Apriori that is every activity, as in the
reference, but in a different order. Because the random draws carry over from one activity to the
next, individual importance values are not identical to a reference run; the method is.

### P10. Column order of the encoding: small deviation

Both use the index-based one-hot encoding: one 0/1 column per (position, activity), with `0` for
the padding of short traces. The reference's `index_encoding` builds it with `pivot_table` +
`get_dummies` (columns `e1_act`, ...); ours builds the same 0/1 values directly with numpy, in a
different column order. XGBoost can break ties between equally good splits differently when the
columns are in another order, so the model can differ slightly. We did not compare F1 scores with a
reference run, because the reference folds are not seeded (P7).

### P11. No crash on an impossible permutation: deviation

When no observed location can keep the order of an itemset, the reference crashes (`max()` of an
empty list or `np.random.choice` of an empty list). We skip that occurrence. When an itemset is in
none of the training cases, the reference would encode an empty table; we give importance 0. Where
the reference works, our behaviour is identical.

### P12. Apriori settings: deviation

* `max_length: 3` (the reference has no limit), so that Apriori and IMPresseD consider the same
  itemset sizes.
* A second Apriori setting (support 0.2, top 25) as a sensitivity check.
* Every single activity is selected as well, as in the paper. In the reference this happens in the
  separate single-activity run (P8); we do it in `AprioriSelector` (`extensions/itemset_selection.py`)
  with `DataManager.activity_support`.

### P13. Datasets: deviation

The repository has ten logs (BPIC11 f1 to f4, three BPIC12 and three Sepsis logs). The assignment
asks for BPIC11, and we run BPIC11 f1, f2 and f3. **To check:** whether f4 should be included.
We added a synthetic log, two shuffled controls and simulated logs (G1). The repository keeps its logs
in `datasets/`, we keep ours in `data/`.

### P14. Not ported: scope

The repository also contains `Process_Permutation.py` (without cross-validation),
`Classical_Permutation.py`, `PDP_ProcessLocation.py`, `Pixel_Flipping_Process.py`,
`dist_location_calculator.py` and, in `tools.py`, prefix generation, frequency and binary encodings,
`optimized_activity_sets`, `get_act_dict`, `PDP_process_location` and regression support. The paper's
main result uses only the cross-validated location permutation, so only that was ported.

### P15. Method signatures and return values: interface only

| Method | Reference | Ours |
|---|---|---|
| `DataManager(...)` | `address, min_length, max_length, frq_threshold, L_max_perc` | `address` (path or DataFrame), `columns`, case attributes, `frq_threshold` |
| `frequent_activity_sets` | `min_support, top_k`; all cases | also `max_length` and `case_list` |
| `split_test_train` | `train_ratio`, not stratified, `random_state=142`; not used by the CV script | `test_size, seed`, stratified, returns the training case ids (as IMPresseD's tool splits; used for the IMPresseD selection) |
| `cross_split_test_train` | `K_fold`; returns case ids per fold | `k_fold, seed`; returns row numbers per fold |
| `index_encoding` | a DataFrame of events; returns a DataFrame with case id and label | a list of traces; returns a numpy array |
| `trace_permutation_importance` | loops over all activities of the log; returns a wide DataFrame | gets the `activities` to test; returns one row per activity and repetition |
| `itemset_permutation_importance` | gets a dict id -> itemset and a `constrain` flag that `shuffle_sequence` never uses | gets a list of itemsets; no `constrain` flag |

### P16. Itemsets with a fixed-location activity are not permuted: addition

The reference permutes every selected itemset. An activity that is always at the same position
cannot have an important *location*, and moving it creates unrealistic traces: the feasibility
constraint keeps a start activity at position 1, but an end activity such as `close` (always last,
at position 6 or 7 depending on the trace length) could be moved to position 6 of a 7-event trace.
`DataManager.fixed_location_activities` finds these activities from the data, without naming them:
every occurrence at the same position counted from the start, or counted from the end.
`CrossValidationProcessPermutation.run` leaves out every itemset that contains one
(`exclude_fixed_locations: true`) and returns them as `excluded_itemsets`.

*Effect.* On the synthetic and simulated logs `register` and `close` are excluded, with every itemset
that contains them (10 of 18 Apriori and 12 of 18 IMPresseD itemsets on the synthetic log). On BPIC11
the rule finds 3 to 5 rare activities (1 or 2 events in the log) that happen to be at one position;
they are single-activity itemsets whose importance was always 0, because they have no alternative
observed location.

## Part 2: IMPresseD repo

The tool is a Tkinter GUI (`GUI_IMPresseD_tool.py`) with an interactive and an automatic mode. We
ported only the **automatic mode**: `AutoStepWise_PPD` (`Auto_IMPID.py`) as `auto_stepwise_ppd`
(`auto_impid.py`), and the functions of `IMIPD.py` and `tools.py` it calls into `imipd.py`. The GUI,
the interactive mode, the plots, `VariantSelection` and the numerical outcome (Spearman) were not
ported; our outcome is binary.

*Verified:* with the settings I1 to I3 applied to the authors' own code, their automatic mode and
our port give exactly the same Pareto-optimal patterns on the synthetic log (15 cases, 18 patterns)
and on a simulated log of 300 cases (109 patterns).

### I1. Total order instead of timestamps: deviation

The tool's `Trace_graph_generator` builds a partially ordered trace graph: two consecutive events
whose timestamps are within `delta_time` count as concurrent. In BPIC11 only the date is recorded,
and 89% of consecutive events in a case share their timestamp. With the tool's approach most of each
trace would become one block of "concurrent" events. We order the events by `event_nr` instead (the
event number is used as the time), so our `trace_graph_generator` builds a chain. The concurrency rule
then never applies and is left out of the port. This is question 1 for Laura in `PLAN.md`.

### I2. "Direct context" rule switched off: deviation

In the tool's `Pattern_extension`, the first extension of an activity can directly create
*predecessor + activity + successor* (three events in one step). Our `pattern_extension` leaves this
rule out, so every step adds exactly one event and `max_extension_step: 2` gives patterns of length
1, 2 and 3, as the assignment asks. The other rules (directly preceding/following, eventually
preceding/following) are unchanged.

### I3. Case distance on the case attributes only: deviation

The paper defines the case distance on case attributes (for BPIC11: `Age`, `Diagnosis`,
`Diagnosis code`, `Specialism code`, `Treatment code`). In the tool, however, the per-activity count
columns are already in `patient_data` when the distance is computed, so they are counted as extra
categorical attributes. With about 160 activities in BPIC11 the case attributes then hardly matter.
We follow the paper and pass only the case attributes to `calculate_pairwise_case_distance`, with the
tool's formula and scipy calls: `(m * jaccard + minmax(euclidean)) / (m + 1)` for `m` categorical
attributes. The tool caches the distances in a pickle file next to the data; we compute them on every run.

### I4. Pattern identity by key: same output, verified

The tool's `update_pattern_dict` checks whether a new pattern instance matches a known pattern with
`networkx.is_isomorphic`, against every known pattern. Our patterns are chains (I1, I2), and two
chains are isomorphic exactly when their activities and their relation types (direct/eventual) match
in order. Our `update_pattern_dict` uses that as a dictionary key (kept in `PatternsDictionary`), which
gives the same answer much faster. Pattern ids are numbered as in the tool (`<core>_<number>`), and
an entry has the tool's structure: `{"pattern": graph, "Instances": {"case": [...], "emb_trace": [...]}}`.

### I5. Fixed settings instead of GUI input: choices

| Setting in the tool | Our value | Why |
|---|---|---|
| Interest functions and directions (user ticks them) | All three: outcome (max), frequency (max), case distance (min) | As in the paper |
| `Max_extension_step` | `max_extension_step: 2` | Pattern lengths 1, 2, 3 (assignment) |
| `Max_gap_between_events` | `max_gap_between_events: 1000` (no limit) | No time gap constraint in our design |
| Test size of the internal split | 0.2, stratified, seed 42 (same call as the tool) | Selection on training cases only |
| `delta_time` | not used | I1 |

### I6. Output: activity sets instead of pattern features (purpose)

The tool's automatic mode returns train and test tables with one column per discovered pattern,
meant as features for a prediction model, plus a JSON file per pattern. We need activity sets for
the permutation, so `auto_stepwise_ppd` returns the patterns on the Pareto fronts of **all** steps
(lengths 1, 2 and 3), and `ImpressedSelector` (`extensions/itemset_selection.py`) drops the ordering
relations (`a -> b`, `b -> a` and `a => b` all become `{a, b}`) and removes duplicates. Using the union
of the fronts is our choice (question 4 for Laura in `PLAN.md`).

### I7. Quirks of the tool that we kept

* Patterns with an eventual relation are not extended further.
* A pattern found from two different starting activities (e.g. `a -> b` from `a` and from `b`)
  gets the instances of the second search added to the first pattern's list, which affects later
  extensions. Counts are taken right after each starting activity, as in the tool.
* When a pattern occurs in all or none of the training cases, its case distance is `NaN` (the mean
  of an empty list); `paretoset` can still put such a pattern on the front.
* The categorical Jaccard distance is computed on label-encoded values, so the attribute value with
  code 0 is treated like "absent" by scipy's `jaccard`.
* Activity counts are computed on all cases, but the interests (and so the Pareto fronts) only on
  the training cases.

### I8. Function signatures and return values: interface only

| Function | Tool | Ours |
|---|---|---|
| `AutoStepWise_PPD` / `auto_stepwise_ppd` | gets the data, `patient_data`, distances, column names, interest settings, a save path; splits train/test itself; returns `train_X`, `test_X` | gets `max_extension_step`, `max_gap_between_events`, the `DataManager` and the training case ids; returns the selected patterns with their interests |
| `calculate_pairwise_case_distance` | returns the condensed list of pair distances, used with `pair_cases` and `start_search_points` | returns the square matrix (same values) |
| `similarity_measuring_patterns` | looks pairs up in the condensed list | reads the block of the square matrix (same values) |
| `frequency_measuring_patterns` | also stores `Pattern_Frequency` and `Case_Support` | only `Frequency_Interest`, the one used for the Pareto front |
| `Trace_graph_generator` / `trace_graph_generator` | data, `delta_time`, case, colours, column names | the trace (I1); no colours, as there are no plots |
| `Pattern_extension`, `Single_Pattern_Extender` | rule flags as arguments | the four rules always on, direct context off (I2) |

## Part 3: Things that exist in neither repository (G1)

* **One entry point** (`main.py`, class `Experiment`, `config.yaml`) runs selection and importance for
  all datasets; in the authors' work these are separate tools.
* **pm4py** for loading and querying the logs (P1).
* **The strategy interface** (`extensions/itemset_selection.py`): `AprioriSelector` and
  `ImpressedSelector` both return plain activity sets.
* **Synthetic log with a planted location effect** and two shuffled negative controls
  (`extensions/synthetic_log.py`, `extensions/shuffle_controls.py`, `generate_data.py`).
* **Simulated logs with a gold standard** (`extensions/simulation.py`, `extensions/evaluation.py`) to
  test whether a strategy selects the planted itemsets.
* **Overlap between strategies** (Jaccard of itemsets and activities) in `results/summary.csv`.
* **Unit tests** (`tests/`, 30 tests).

## Open points for the group

1. The location importance has to be rerun after the corrections P4 to P6; the importance values of
   the previous full run are no longer valid (the selection results are).
2. P13: include BPIC11 f4?
3. Runtime: the previous full run (6 datasets x 3 strategies) took about 67 hours of wall-clock time on
   a laptop that also slept. Measuring on the test cases (P4) makes the importance step faster, but
   it is still long; we may want to drop the sensitivity setting for the importance step or run the
   datasets on separate machines.
4. Simulated logs: no strategy selects the planted pair `{p, q}` (README, Results). This supports the
   limitation that selecting itemsets on frequency or presence cannot find pure location effects.
