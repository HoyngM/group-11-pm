# Plan: Group 11, Explainable Predictive Process Monitoring

Task (from the assignment description): reproduce the location permutation importance
pipeline of Vazifehdoostirani et al. (2024) with Apriori itemsets, add one variant that uses
the IMPresseD patterns (2023) with pattern lengths 1, 2 and 3 and the ordering relations
ignored, and compare how the importance of activities and their locations changes.
Data: the three BPIC11 logs. Short paper: 5 pages excluding references.

Max's parts (with Jerrik for the conceptual approach): simulation data, itemset selection
and its code, conceptual approach. Floris and Job: permutation importance pipeline.

## Deadlines

| Date | What |
|---|---|
| 25 Oct, 15:00 | Questions for the office hour of 26 Oct (Laura Genga) |
| 28 Oct, 23:59 | Short paper draft for peer review |
| 3 Nov, 23:59 | Peer feedback on the partner group's draft |
| 9 Nov, 23:59 | Final deliverables: code, short paper, poster, technology statement |
| 12 Nov, 13:45 | Poster session |

## Done

- Itemset selection: Apriori baseline and IMPresseD variant (`code/itemset_selection/`),
  YAML config, README, `requirements.txt`, 13 unit tests.
- Simulated logs with a planted location effect and a gold standard.
- Synthetic log of exactly 100 events with a planted location effect, three shuffled negative controls
  (`generate_data.py`, `check_controls.py`) and a port of the reference location permutation importance
  (`location_permutation/`, `run_location_importance.py`). To check against the pipeline of Floris and Job.
- Selection results on BPIC11 and on the simulated logs (`python main.py`).
- Mutual information selector removed: the assignment prescribes IMPresseD as the variant.
- `main.tex`: conceptual approach (with figure), implementation of the selection, data,
  gold standard, parameter settings, selection results and the limitations of the selection
  are written. The paper compiles to 5 pages including references and the empty parts.
- Simulation decision: keep the current logs as a demonstration that selection on the
  existence of patterns cannot see a location effect, and use it in the limitations.

## To do

### Max and Jerrik
- [ ] Jerrik reviews Section 2 (conceptual approach) of `main.tex`.
- [ ] Send the questions below to Laura (by 25 Oct, 15:00).
- [ ] Fix the author name and the course code in `main.tex`. The name says "Hoijng" and
  your university e-mail says "hoijng", so check which spelling is right. The assignment
  says JM0210 while the template and `main.tex` say JM0211.
- [ ] Technology statement (mandatory, 0 points without it): fill in `main.tex` yourselves
  after checking the work. Each student states what AI tools they used and that they checked it.

### Floris and Job
- [ ] Index-based encoding, XGBoost, baseline F1 with cross-validation.
- [ ] Location permutation (feasibility, order preservation, 10 repetitions) and the F1 drop
  per itemset, as box plots.
- [ ] Read the itemsets from `code/results/<dataset>/<strategy>_itemsets.json`
  (`python main.py` creates them) for the BPIC11 and simulated logs.
- [ ] Put the code in this repository with one `main`, and extend the README.
- [ ] Write the implementation and evaluation design parts for the permutation in `main.tex`
  (the red TODOs in Sections 3 and 4).

### Whole group
- [ ] Decide how the selection and the cross-validation fit together (question 5).
- [ ] Evaluation: compare the two strategies (which activities and locations are important,
  how many itemsets each needs) and check the simulated logs against the gold standard.
- [ ] Short paper: write Section 1 (motivation and background), the permutation results,
  and the remaining limitations and future work. Space is tight: the selection parts take
  about 3.7 pages, so Section 1 and the results must fit in about 1.3 pages.
- [ ] Remove all red TODOs from `main.tex`.
- [ ] Poster from the Canvas template (`Assignments/Opdracht en templates/Poster_GroupX.vsdx`) for 12 Nov.
- [ ] Peer feedback on the partner group's draft by 3 Nov.
- [ ] Final check on 9 Nov: README, `requirements.txt` and no absolute paths are correct, and
  the results in the paper and poster match what the code gives.

## Questions for Laura (office hour 26 Oct)

1. The BPIC11 logs contain no information about concurrency. Is it fine to treat the traces as
   totally ordered, so that the IMPresseD rule for concurrent events is never used?
2. In the reference implementation of IMPresseD, patterns with an eventual relation are not
   extended further. Should we do the same?
3. For Apriori we use minimum support 0.5 and the top 10 itemsets with more than one activity
   (the defaults of the authors' code, plus every single activity), and test 0.2 and the top 25.
   Are these settings reasonable?
4. For the variant we take the union of the Pareto fronts of pattern lengths 1, 2 and 3 and drop
   the ordering relations. Is this what you intend, or should we use only the front of the
   longest length?
5. We select the itemsets on 80% of the cases, so that the labels of the test cases do not
   influence the selection. Should the cross-validation of the pipeline use the same training
   cases, or should we repeat the selection inside each fold?

## Where the old text went

The previous working draft (`Draft_Max_Group11.docx`) was replaced by this plan, and its
text is now in `main.tex`. It is still in the git history:
`git show bf055a2:drafts/Draft_Max_Group11.docx > Draft_Max_Group11.docx`.
