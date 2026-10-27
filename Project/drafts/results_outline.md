# Outline: Section 4, Evaluation Design and Results

Group 11. Outline for Section 4 of `main.tex`, aligned with the short paper template and rubric item
S4 (10 of the 35 short paper points). The template merges evaluation design and results into one
section, so this outline covers the whole section, using the three paragraphs that are already in
`main.tex`: *Data*, *Evaluation design*, *Results*.

## What S4 asks for, and where it is covered

| Rubric S4 requirement | Covered in |
|---|---|
| Data described in sufficient detail | 4.1 |
| Evaluation design correct, clearly described and **motivated** | 4.2 (evaluation questions E1 to E4) |
| How the gold standard was created, and why it is reasonable | 4.2, gold standard |
| Parameter settings explained, tested, and their choice motivated | 4.2, parameters; 4.3 E4 |
| Results explained and **visualised** (tables, figures) | 4.3, Table A, Table B, Figure C |
| Corner cases / failed cases, with **why** they failed | 4.3 E2 (`{p, q}` never selected), E1 (controls) |
| Comparison with related methods | 4.3 E3 (Apriori, the original method, vs IMPresseD, our variant) |

**Page budget.** About 1.3 to 1.5 pages, with two small tables and one figure. Recheck this after
Jerrik's rewrite: Sections 1 to 3 must leave enough room within 5 pages excluding references.

---

## 4.1 Data (paragraph *Data*, about 6 lines)

Three kinds of logs, each with its own purpose:

1. **BPIC11 f1, f2, f3** (from the authors' repository, `data/BPIC11_f*_trunc*.csv`): hospital log,
   with the same labelling as in the original paper (Teinemaa et al.). One line per log: cases (1130 /
   1130 / 1111), activities, maximum trace length (36 / 40 / 31 after truncation), share of
   positive labels. Mention the case attributes used by the IMPresseD case distance (already
   listed in Section 2, so refer back to it).
   *Source:* `results/summary.csv` (columns `cases`, `events`, `activities`, `positive_rate`).
2. **Synthetic log and two negative controls** (`data/synthetic/`): 15 cases, 100 events. Activity `B`
   occurs in every case and only its *location* decides the label (early → 0, late → 1, 10% label noise).
   Controls: `shuffled_within_case` (location removed) and `shuffled_outcomes` (labels permuted).
   Purpose: check that the *importance* step finds a location effect and finds nothing when there
   is none.
3. **Simulated logs** (`extensions/simulation.py`): 1000 cases, 10 logs for each of two scenarios
   (`SIM_rare_pair` with the pair in 30% of cases, `SIM_frequent_pair` with 60%). Planted: a single
   activity with a location effect (`x`), a pair with a location effect (`{p, q}`) and an activity
   with an existence effect (`y`). Purpose: check that the *selection* step picks the itemsets that matter.

## 4.2 Evaluation design (paragraph *Evaluation design*, about 12 lines)

Open with the four evaluation questions. They motivate the design, which the rubric asks for.

- **E1, correctness of the importance step.** Does a location effect give a clear F1 drop, and do
  the controls give a drop of about 0?
- **E2, correctness of the selection step.** Does each strategy select the planted itemsets?
- **E3, main comparison (the assignment question).** On BPIC11, does choosing itemsets with
  IMPresseD instead of Apriori change *which* activities and itemsets have an important location,
  and how many itemsets each strategy needs?
- **E4, parameter sensitivity.** Does the Apriori baseline change with `min_support` 0.2 and
  `top_k` 25 instead of 0.5 and 10?

**Metrics.**

- Model quality: weighted F1 on the held-out fold, train and test (to show overfitting).
- Location importance: F1 drop per itemset, 50 values (5 folds × 10 repetitions), reported as
  mean ± standard deviation and box plots, as in the original paper.
- Selection against the gold standard: *location recall* (share of the planted location itemsets
  that were selected), *existence recall*, and *noise share* (share of the selected itemsets that
  contain only noise activities).
- Strategy comparison: number and sizes of the itemsets, Jaccard overlap of the selected itemsets
  and activities, overlap of the top 10 most important itemsets, Spearman rank correlation of the
  importance of the itemsets that both strategies select.

**Gold standard (say why it is reasonable).** In the synthetic and simulated logs we generate the
labels ourselves, so the itemsets that truly matter are known. Real BPIC11 has no ground truth, so
for BPIC11 the evaluation is a *comparison* between strategies, not an accuracy test. The negative
controls are a gold standard of "nothing matters". State this distinction explicitly; it answers the
"how did you create the gold standard" question of the rubric.

**Parameters.** Refer to Table 1 in Section 2 instead of repeating it. Motivate the main choices
in one sentence each: Apriori 0.5/10 are the authors' defaults; IMPresseD length 3 matches Apriori
`max_length`; 5 folds and 10 repetitions as in the paper; selection on an 80% split for IMPresseD
because it uses the labels.

## 4.3 Results (paragraph *Results*, about 0.8 page plus visuals)

Follow the order E1 → E2 → E3 → E4, so that the trust in the method is established before
the main comparison.

**E1, synthetic log and controls (2 to 3 lines, numbers in Table B).**
- Model test F1: synthetic 0.65 against 0.39 / 0.42 for the controls → the location effect can be learned.
- Importance of `B` on the synthetic log against the maximum |F1 drop| in the controls.
  *Needs the rerun.* The README states ~0.43 for `B` and < 0.1 in the controls, but that comes from
  the 200-case unit test log, not the 15-case log in `data/`.

**E2, simulated logs: selection against the gold standard (3 to 4 lines, Table B).**
- `y` (existence) is selected by all strategies in every log; `x` (location) by all, except
  IMPresseD in the rare-pair scenario (8 of 10 logs).
- **Failure case (required by the rubric):** no strategy selects `{p, q}` in any log. Why: Apriori
  selects on frequency and IMPresseD on the *presence* of a pattern, while for `{p, q}` only the
  *location* matters, so neither selection criterion can see it. Connect this to Section 5.
- IMPresseD selects fewer noise itemsets (noise share 0.50 against 0.73 for Apriori, per the README on `main`).
  *Source:* `results/simulated/summary_simulated_mean.csv`.

**E3, BPIC11: Apriori against IMPresseD (main part, Table A and Figure C).**
1. *Selection:* IMPresseD selects mostly itemsets of size 2 and 3, Apriori mostly single activities;
   the multi-activity itemsets hardly overlap (Jaccard ≈ 0.00 to 0.02). Table A, rows 1 to 3.
2. *Model:* train F1 ≈ 0.99 against test ≈ 0.90 to 0.93. Note that this overfitting is the reason
   the importance is measured on the test folds (refer to the correction in Section 3).
3. *Importance:* the most important itemset per strategy and log; whether the same activities come
   out on top (top-10 overlap, Spearman on shared itemsets); whether IMPresseD finds itemsets that
   are more important than the best Apriori itemset. Figure C: box plots of the top itemsets of both
   strategies side by side, for one log (f2 is the most interesting in the old run).
4. *Efficiency:* number of itemsets tested and runtime per strategy (Apriori 7 to 11 min, IMPresseD 25+ min
   per log).
5. *Interpretation:* discuss one top itemset the way the original paper does (where in the
   trace it occurs, and why its location could matter).

**E4, sensitivity of the baseline (1 to 2 lines).** How the number of itemsets and the top itemsets
change with support 0.2 / top 25. If the full run is not feasible, report only the selection
(number of itemsets, overlap) and say so.

## Planned visuals

| ID | Type | Content | Source |
|---|---|---|---|
| Table A | Table | Per BPIC11 log × strategy: number of itemsets (sizes 1/2/3), Jaccard with Apriori, train/test F1, top itemset and its mean F1 drop, top-10 overlap, Spearman | `results/summary.csv` + comparison step (see dependencies) |
| Figure C | Figure | Box plots of the F1 drop of the top 10 itemsets, Apriori and IMPresseD next to each other, one BPIC11 log | `results/<log>/<strategy>_location_importance.csv` |
| Table B | Small table | Synthetic vs controls (test F1, importance of `B`, max \|drop\| controls) and simulated recall (location, existence, noise share per strategy) | `results/summary.csv`, `results/simulated/summary_simulated_mean.csv` |

---

## Status: what can be written now and what waits

**Can be written now** (independent of the permutation fix, numbers from the README on `main`):
4.1 Data, 4.2 Evaluation design, E2 (simulated logs), the selection part of E3 (itemset counts and
overlap), model F1 on BPIC11 and on synthetic vs controls.

**Waits for a fresh run of `python main.py`** (the README states the importance must be rerun with
the corrected permutation): every F1 drop number, Figure C, the importance part of E3, the
importance of `B` in E1.

## Dependencies and open checks

1. **Comparison metrics are not on `main`.** On `main`, `results/summary.csv` only compares the
   *selection* (Jaccard). Top-10 overlap, Spearman on shared itemsets and "IMPresseD itemsets above
   the best Apriori itemset" are in `evaluation/strategy_comparison.py` on the branch
   `job/location-importance-comparison`. That code has to be ported to `main` (as a small step after
   the importance run), or E3 point 3 must be limited to the top itemset per strategy.
2. **Do not use the numbers in the local `results/` folder.** It has the old folder layout
   (`importance/`, `selection/`, `comparison/`) and old settings; e.g. it lists 32 / 62 / 29 IMPresseD
   itemsets where the README on `main` says 154 / 176 / 142. Every number in the paper must come
   from a fresh run on `main` (the implementation is graded on reproducibility).
3. **Runtime.** `apriori` + `impressed` on the three BPIC11 logs plus synthetic take about 2 hours;
   the sensitivity setting took about 67 hours. Decide as a group: drop E4, run it on f1 only, or
   report only its selection.
4. **The synthetic log is small** (15 cases, so 3 test cases per fold). Consider running E1 on a
   larger log (`python generate_data.py --cases 400`), or present E1 as a sanity check only.
5. **BPIC11 labels.** The labels follow Teinemaa et al.: a case is positive when a specific activity
   occurs. Check whether the top itemsets (e.g. `ac370000`, `376400`) are related to those label
   activities. If so, add one sentence that high importance partly reflects how the label is defined.
6. **Draft deadline 28 Oct 23:59.** For the peer-review draft: write 4.1, 4.2, E2 and the selection
   part of E3, and leave `\todo{}` markers for the importance numbers.
