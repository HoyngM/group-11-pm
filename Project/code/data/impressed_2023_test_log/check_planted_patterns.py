"""Sanity check: compute the three IMPresseD interest functions for the planted patterns.

Simplified re-implementation for validation only (not the group's algorithm):
  frequency      = share of cases with >= 1 pattern instance
  outcome (OI)   = Spearman correlation between per-case instance count and survival_days
  case distance  = mean pairwise distance between cases with / without the pattern, using
                   (normalised age distance + Jaccard-style mismatch on gender, morphology) / 3
Concurrency oracle: events of two activities within DELTA_DAYS of each other are concurrent.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

DELTA_DAYS = 3
path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("synthetic-event-log.csv")
df = pd.read_csv(path, parse_dates=["time:timestamp"])
cases = df.groupby("case:concept:name").first()[["case:age", "case:gender", "case:morphology", "survival_days"]]
ids = np.array(cases.index)
age_range = cases["case:age"].max() - cases["case:age"].min()


def count(case, pattern):
    e = df[df["case:concept:name"] == case]
    if pattern[0] == "single":
        return int((e["concept:name"] == pattern[1]).sum())
    a = e[e["concept:name"] == pattern[1]]["time:timestamp"]
    b = e[e["concept:name"] == pattern[2]]["time:timestamp"]
    concurrent = any(abs((x - y).days) <= DELTA_DAYS for x in a for y in b)
    if pattern[0] == "conc":
        return int(concurrent)
    # "then": concurrent start, followed by a later cycle of the second activity
    return int(concurrent and any((z - x).days > DELTA_DAYS for x in a for z in b))


def distance(c1, c2):
    r1, r2 = cases.loc[c1], cases.loc[c2]
    num = abs(r1["case:age"] - r2["case:age"]) / age_range
    cat = (r1["case:gender"] != r2["case:gender"]) + (r1["case:morphology"] != r2["case:morphology"])
    return (num + cat) / 3


def report(name, pattern):
    counts = np.array([count(c, pattern) for c in ids])
    has = counts > 0
    freq = has.mean()
    rho = spearmanr(counts, cases["survival_days"])[0] if counts.std() > 0 else float("nan")
    cd = (np.mean([distance(a, b) for a in ids[has] for b in ids[~has]])
          if 0 < has.sum() < len(ids) else float("nan"))
    print(f"{name:34s} freq={freq:4.2f} ({has.sum():2d}/{len(ids)})  OI(spearman)={rho:+.2f}  "
          f"caseDist={cd:.2f}  median survival in/out="
          f"{cases['survival_days'][has].median():.0f}/{cases['survival_days'][~has].median():.0f}")


print(f"{len(df)} events, {len(ids)} cases")
for act in ["consultation", "capecitabine", "oxaliplatin", "paclitaxel", "radiotherapy", "supportive care"]:
    report(act, ("single", act))
report("oxaliplatin || capecitabine", ("conc", "oxaliplatin", "capecitabine"))
report("(oxa || cap) -> capecitabine", ("then", "oxaliplatin", "capecitabine"))
report("radiotherapy || capecitabine", ("conc", "radiotherapy", "capecitabine"))
