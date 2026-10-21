# Synthetic test log for IMPresseD

Small synthetic event log (102 events, 22 patients, 13 trace variants) for testing an implementation of
*Interactive Multi-interest Process Pattern Discovery* (Vazifehdoostirani et al., BPM 2023). It mimics the
paper's outcome-oriented use case (palliative oncology treatments, outcome = survival time) and contains
**planted patterns with known properties**, so the algorithm's output can be checked against a ground truth.
All data is synthetic; the clinical content is illustrative and not medically meaningful.

## Files

| File | Purpose |
|---|---|
| `synthetic-event-log.csv` | The event log (one row per event) |
| `generate_synthetic_log.py` | Regenerates the log (fixed seed, deterministic) |
| `check_planted_patterns.py` | Computes simplified versions of the three interest functions for the planted patterns |

## Columns

| Column | Level | Meaning |
|---|---|---|
| `case:concept:name` | case | Patient / case ID |
| `concept:name` | event | Activity: consultation, capecitabine, oxaliplatin, paclitaxel, radiotherapy, supportive care |
| `time:timestamp` | event | Single timestamp per event (as in the authors' tool) |
| `org:resource` | event | Pseudonymous clinician code |
| `case:age`, `case:gender`, `case:morphology` | case | Initial case attributes, the potential confounders for **case distance** |
| `survival_days` | case | **Numerical outcome** (use Spearman correlation) |
| `label` | case | **Categorical outcome**: `long` if `survival_days >= 365`, else `short` (use information gain) |

Case attributes and outcomes are repeated on every row of a case, as in the authors' example log.

## Tool settings (authors' tool, `GUI_IMPresseD_tool.py`)

- Case ID column: `case:concept:name`, activity column: `concept:name`, timestamp column: `time:timestamp`
- Outcome: `survival_days` (numerical) or `label` (categorical)
- Delta time: **259200 s (3 days)**. Events closer together than this count as concurrent, which matches the
  paper's conversion oracle (treatments starting within three days of each other are parallel).
- Case attributes for the distance function: `case:age`, `case:gender`, `case:morphology`

## Planted ground truth

Values from `check_planted_patterns.py` (frequency = share of cases with the pattern; OI = Spearman between
per-case instance count and `survival_days`; case distance = mean distance between cases with and without the pattern).

| Pattern | Freq. | OI | Case dist. | What it tests |
|---|---|---|---|---|
| oxaliplatin ‖ capecitabine, then capecitabine continues | 0.36 | +0.60 | 0.46 | **Should be found**: frequent enough, strong positive effect, low confounding. Mirrors the paper's Fig. 4 example. |
| capecitabine (single activity) | 0.86 | +0.25 | 0.41 | Frequent but weak: should be on the frequency side of the Pareto front, not a top outcome pattern. |
| paclitaxel (single activity) | 0.27 | -0.69 | **0.65** | **Confounded**: all paclitaxel patients are elderly men with diffuse morphology, so the strong negative effect is unreliable. Case distance should flag it. |
| radiotherapy ‖ capecitabine | 0.09 | +0.47 | 0.41 | Rare but strong: only 2 patients, tests how a minimum-frequency filter treats it. |
| supportive care | 0.50 | -0.07 | 0.45 | Noise: should be dominated in the Pareto front. |
| consultation | 1.00 | n/a | n/a | Occurs in every case: trivial pattern, no outcome variance. |

Trace-variant ground truth (case groups): A = 8 patients with the oxaliplatin/capecitabine combination
(survival 390-640 d), B = 6 elderly patients with paclitaxel (70-210 d), C = 4 capecitabine/radiotherapy
patients (150-410 d), D = 2 patients with radiotherapy ‖ capecitabine (610-700 d), E = 2 short-survival
patients without paclitaxel (140-250 d).

## Limitations

22 cases are far too few for statistical significance; the log is a functional test with a known answer, not
an evaluation dataset. For quantitative evaluation, also use a larger log (for example the authors' `Production_trunc23.csv`
or a log provided on Canvas).
