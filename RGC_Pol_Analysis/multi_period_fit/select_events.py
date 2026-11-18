"""Pick a small, representative set of Polarize events across calibration periods.

Per online-cc period with enough events: the N_PER_SIGN strongest positive and negative events
(by |area|), each from a different data file where possible. Writes selected_events.yaml.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data_creation/data-d"
EVENT_MAP = HERE.parent / "baseline_refit/cache/event_baseline_map.csv"
OUT = HERE / "selected_events.yaml"

N_PER_SIGN = 2
MIN_PERIOD_EVENTS = 1000
MIN_ABS_POL = 0.10
MAX_ABS_POL = 1.0
PROBLEM_PERIODS = {-14.5: "lineshape check: online P ~2.3x too high"}
SKIP_PERIODS = {-5.246731: "online processing unusable (Jun-Jul 2022)"}


def main():
    want = defaultdict(set)
    base_of = {}
    with open(EVENT_MAP) as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "Polarize":
                want[row["file"]].add(int(row["event_index"]))
                base_of[(row["file"], int(row["event_index"]))] = int(row["baseline_id"])

    by_cc = defaultdict(list)
    for fname, idxs in sorted(want.items()):
        with open(DATA / fname) as fh:
            for i, line in enumerate(fh):
                if i not in idxs:
                    continue
                d = json.loads(line)
                f = d["freq_list"]
                if not (32.29 < f[0] < 32.31 and 33.09 < f[-1] < 33.11) or d["channel"]["name"] != "Deuteron5T":
                    continue
                if not (MIN_ABS_POL <= abs(d["pol"]) <= MAX_ABS_POL) or abs(d["cc"] * d["area"] - d["pol"]) > 1e-9:
                    continue
                by_cc[float(d["cc"])].append({"file": fname, "index": i, "time": d["start_time"][:19],
                                              "cc": float(d["cc"]), "pol": float(d["pol"]), "area": float(d["area"]),
                                              "baseline_id": base_of[(fname, i)]})

    selected = []
    for cc, evs in sorted(by_cc.items()):
        if cc in SKIP_PERIODS or len(evs) < MIN_PERIOD_EVENTS:
            continue
        for sign in (+1, -1):
            pool = sorted([e for e in evs if e["pol"] * sign > 0], key=lambda e: -abs(e["area"]))
            used_files = set()
            for e in pool:
                if e["file"] in used_files:
                    continue
                used_files.add(e["file"])
                selected.append({**e, "period_note": PROBLEM_PERIODS.get(cc, "")})
                if len(used_files) == N_PER_SIGN:
                    break
    OUT.write_text(yaml.safe_dump(selected, sort_keys=False))
    periods = defaultdict(lambda: [0, 0])
    for e in selected:
        periods[e["cc"]][0 if e["pol"] > 0 else 1] += 1
    for cc, (npos, nneg) in sorted(periods.items()):
        print(f"cc {cc:8.3f}: {npos} positive, {nneg} negative  {PROBLEM_PERIODS.get(cc, '')}")
    print(f"{len(selected)} events -> {OUT}")


if __name__ == "__main__":
    main()
