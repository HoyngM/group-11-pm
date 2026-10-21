"""Generate a small synthetic oncology-style event log with planted ground-truth patterns.

The log is meant to test an implementation of IMPresseD (Vazifehdoostirani et al., BPM 2023):
interactive multi-interest process pattern discovery for outcome-oriented patterns.
The three interest dimensions are frequency, outcome correlation and case distance
(confounding). Each planted group below targets one of those dimensions.

Usage:  python generate_synthetic_log.py [output.csv]
The output path defaults to synthetic-event-log.csv next to this script.
"""
import random
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

SEED = 2026
random.seed(SEED)

RESOURCES = ["MD-01", "MD-02", "MD-03", "MD-04"]  # pseudonymous clinician codes
LONG_SURVIVAL_DAYS = 365  # threshold for the categorical outcome label


def cycles(drug, first_day, n):
    """n treatment cycles of `drug`, one every 21 days from first_day."""
    return [(drug, first_day + 21 * k) for k in range(n)]


def build_cases():
    """Return a list of case specs: (group, age, gender, morphology, activities, survival).

    Attributes and counts are assigned deterministically so the planted properties are
    stable; only timestamps/resources use the seeded RNG.
    """
    cases = []

    # A: oxaliplatin + capecitabine start together (within 3 days = concurrent), OXA stops,
    #    capecitabine continues. GOOD pattern: frequent, strongly positive outcome, and the
    #    patients look like the rest of the population (low case distance).
    ages = [48, 52, 55, 58, 60, 62, 64, 66]
    morph = ["intestinal", "intestinal", "intestinal", "other", "intestinal", "intestinal", "diffuse", "intestinal"]
    surv = [390, 430, 460, 500, 520, 560, 600, 640]
    cap_n = [2, 2, 2, 2, 2, 2, 2, 2]
    for i in range(8):
        acts = [("oxaliplatin", 7)] + cycles("capecitabine", 8, cap_n[i])
        if i in (1, 3, 4, 6):
            acts.append(("supportive care", 35))
        if i in (0, 5):
            acts += cycles("capecitabine", 8 + 21 * cap_n[i], 1)  # one longer cycle history
        cases.append(("A", ages[i], "FM"[i % 2], morph[i], acts, surv[i]))

    # B: paclitaxel only in elderly male patients with diffuse morphology and poor prognosis.
    #    CONFOUNDED pattern: strongly negative outcome but high case distance (age/gender/morphology).
    ages = [71, 73, 76, 78, 80, 83]
    surv = [210, 180, 150, 120, 95, 70]
    pac_n = [2, 3, 2, 3, 2, 3]
    cap_first = [0, 2, 2, 2, 2, 2]  # most patients got capecitabine before paclitaxel
    for i in range(6):
        acts = cycles("capecitabine", 7, cap_first[i]) if cap_first[i] else []
        acts += cycles("paclitaxel", 7 + 21 * cap_first[i], pac_n[i])
        if i in (0, 2, 4):
            acts.append(("supportive care", 21))
        cases.append(("B", ages[i], "M", "diffuse", acts, surv[i]))

    # C: capecitabine monotherapy (some with radiotherapy), mixed patients, mid-range outcome.
    #    FREQUENT-BUT-WEAK pattern.
    ages = [52, 59, 63, 67]
    surv = [150, 230, 330, 410]
    morph = ["intestinal", "other", "intestinal", "intestinal"]
    cap_n = [2, 2, 2, 2]
    for i in range(4):
        acts = cycles("capecitabine", 7, cap_n[i]) if i != 3 else [("radiotherapy", 7), ("radiotherapy", 28)]
        if i == 0:
            acts.append(("radiotherapy", 40))
        if i in (1, 2):
            acts.append(("supportive care", 30))
        cases.append(("C", ages[i], "FM"[i % 2], morph[i], acts, surv[i]))

    # D: radiotherapy concurrent with capecitabine in only 2 patients.
    #    RARE-BUT-STRONG pattern: low frequency, strongly positive outcome.
    for i in range(2):
        acts = [("radiotherapy", 7)] + cycles("capecitabine", 8, 2)
        if i == 0:
            acts.append(("supportive care", 35))
        cases.append(("D", [54, 59][i], "FM"[i], "intestinal", acts, [610, 700][i]))

    # E: two patients with short survival who never received paclitaxel
    #    (one gets supportive care only, so capecitabine is not given to everyone).
    acts = [("supportive care", 14), ("supportive care", 35)]
    cases.append(("E", 58, "F", "intestinal", acts, 140))
    acts = cycles("capecitabine", 7, 2)
    cases.append(("E", 63, "M", "intestinal", acts, 250))
    return cases


def to_rows(cases):
    rows = []
    start_pool = [datetime(2022, 3, 1) + timedelta(days=17 * k) for k in range(len(cases))]
    random.shuffle(start_pool)
    ids = list(range(1, len(cases) + 1))
    random.shuffle(ids)
    for (group, age, gender, morph, acts, survival), case_no, start in zip(cases, ids, start_pool):
        case_id = f"Patient_{case_no:03d}"
        label = "long" if survival >= LONG_SURVIVAL_DAYS else "short"
        events = [("consultation", 0)] + acts
        for act, day in sorted(events, key=lambda x: x[1]):
            ts = start + timedelta(days=day, hours=random.randint(8, 16), minutes=random.randint(0, 59))
            rows.append({
                "case:concept:name": case_id,
                "concept:name": act,
                "time:timestamp": ts.strftime("%Y-%m-%d %H:%M:%S"),
                "org:resource": random.choice(RESOURCES),
                "case:age": age,
                "case:gender": gender,
                "case:morphology": morph,
                "survival_days": survival,
                "label": label,
            })
    df = pd.DataFrame(rows)
    df["_t"] = pd.to_datetime(df["time:timestamp"])
    return df.sort_values(["case:concept:name", "_t"]).drop(columns="_t").reset_index(drop=True)


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("synthetic-event-log.csv")
    log = to_rows(build_cases())
    log.to_csv(out, index=False)
    print(f"{len(log)} events, {log['case:concept:name'].nunique()} cases -> {out}")
