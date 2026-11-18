"""Lineshape cc for every online-cc period in data-d (parallel).

Events are grouped by their online cc value (one group ~ one calibration period). In each group the
strongest-signal Polarize events (top fraction by |area|, spread in time) are fitted with the Dulya
doublet at free amplitude; the horn asymmetry gives P_shape and cc_shape = P_shape / area.

Reliability: the asymmetry is large only at high polarization. Groups whose median |P_shape| is below
MIN_RELIABLE_P are flagged "low P".

Requires ../baseline_refit/cache/event_baseline_map.csv (collect_baselines.py).
Usage: python shape_cc_survey.py
"""
import csv
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data_creation/data-d"
EVENT_MAP = HERE.parent / "baseline_refit/cache/event_baseline_map.csv"
OUT = HERE / "results" / "shape_cc_survey"
sys.path.insert(0, str(HERE))
from shape_cc_check import shape_p  # noqa: E402

N_PER_GROUP = 20
TOP_FRACTION = 0.3
MIN_GROUP_EVENTS = 20
MIN_RELIABLE_P = 0.25
N_WORKERS = 14
SWEEP = (32.29, 33.11)


def scan():
    """One pass over the unique files: per Polarize event, its time, area, cc, channel."""
    files = defaultdict(set)
    with open(EVENT_MAP) as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "Polarize":
                files[row["file"]].add(int(row["event_index"]))
    events = []
    for fname, idxs in sorted(files.items()):
        with open(DATA / fname) as fh:
            for i, line in enumerate(fh):
                if i not in idxs:
                    continue
                d = json.loads(line)
                f = d["freq_list"]
                if not (SWEEP[0] < f[0] < 32.31 and 33.09 < f[-1] < SWEEP[1]):
                    continue
                events.append({"file": fname, "index": i, "time": d["start_time"][:19], "area": float(d["area"]),
                               "cc": float(d["cc"]), "pol": float(d["pol"]), "channel": d["channel"]["name"]})
    return events


def pick(group):
    group = sorted(group, key=lambda e: abs(e["area"]), reverse=True)
    strong = sorted(group[: max(N_PER_GROUP, int(len(group) * TOP_FRACTION))], key=lambda e: e["time"])
    idx = np.unique(np.linspace(0, len(strong) - 1, min(N_PER_GROUP, len(strong))).astype(int))
    return [strong[i] for i in idx]


def fit_event(e):
    with open(DATA / e["file"]) as fh:
        for i, line in enumerate(fh):
            if i == e["index"]:
                d = json.loads(line)
                break
    p = shape_p(np.asarray(d["freq_list"]), np.asarray(d["fitsub"]))
    return {**e, "P_shape": p, "cc_shape": p / e["area"]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    events = scan()
    groups = defaultdict(list)
    for e in events:
        groups[e["cc"]].append(e)
    todo = [e for cc, g in groups.items() if len(g) >= MIN_GROUP_EVENTS for e in pick(g)]
    print(f"scanned {len(events)} Polarize events in {time.time() - t0:.0f} s; "
          f"{len(todo)} events to fit from {sum(len(g) >= MIN_GROUP_EVENTS for g in groups.values())} cc groups", flush=True)

    t1 = time.time()
    with Pool(N_WORKERS) as pool:
        fitted = pool.map(fit_event, todo, chunksize=2)
    print(f"fitted in {time.time() - t1:.0f} s", flush=True)

    by_cc = defaultdict(list)
    for r in fitted:
        by_cc[r["cc"]].append(r)
    table = []
    for cc, rows in sorted(by_cc.items(), key=lambda kv: min(r["time"] for r in groups[kv[0]])):
        ccs = np.array([r["cc_shape"] for r in rows])
        ps = np.array([abs(r["P_shape"]) for r in rows])
        sign_ok = np.mean([np.sign(r["P_shape"]) == np.sign(r["pol"]) for r in rows])
        med = float(np.median(ccs))
        table.append({
            "cc_online": cc, "n_events_in_period": len(groups[cc]), "n_fitted": len(rows),
            "first": min(e["time"] for e in groups[cc]), "last": max(e["time"] for e in groups[cc]),
            "channels": sorted({e["channel"] for e in groups[cc]}),
            "median_abs_P_shape": float(np.median(ps)), "median_abs_P_online": float(np.median([abs(r["pol"]) for r in rows])),
            "cc_shape_median": med,
            "cc_shape_err": float(1.2533 * np.std(ccs) / np.sqrt(len(ccs))),
            "cc_shape_spread": float(np.std(ccs)),
            "ratio_shape_to_online": med / cc,
            "sign_agreement": float(sign_ok),
            "reliable": bool(np.median(ps) >= MIN_RELIABLE_P),
            "events": rows,
        })
    (OUT / "shape_cc_survey.yaml").write_text(yaml.safe_dump(table, sort_keys=False))
    plot(table)

    print(f"\n{'period':23} {'online cc':>10} {'lineshape cc':>16} {'ratio':>6} {'|P| shape':>9} {'n ev':>6}  note")
    for r in table:
        note = "" if r["reliable"] else "low P (less reliable)"
        if r["sign_agreement"] < 0.8:
            note += " sign mismatch"
        print(f"{r['first'][:10]}..{r['last'][5:10]}   {r['cc_online']:10.3f} {r['cc_shape_median']:+9.2f} ± {r['cc_shape_err']:.2f} "
              f"{r['ratio_shape_to_online']:6.2f} {r['median_abs_P_shape']:9.2f} {r['n_events_in_period']:6d}  {note}")
    print(f"\ntotal {time.time() - t0:.0f} s; wrote {OUT}")


def plot(table):
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    ax = axes[0]
    for r in table:
        c = "tab:blue" if r["reliable"] else "0.6"
        ax.errorbar(-r["cc_online"], -r["cc_shape_median"], yerr=r["cc_shape_err"], fmt="o", color=c,
                    ms=4 + 2 * np.log10(r["n_events_in_period"]), capsize=3)
        ax.annotate(r["first"][:7], (-r["cc_online"], -r["cc_shape_median"]), fontsize=7, xytext=(4, 2),
                    textcoords="offset points", color=c)
    lim = [0.5, 30]
    ax.plot(lim, lim, color="0.3", lw=0.8, ls="--", label="lineshape = online")
    ax.fill_between(lim, [0.9 * x for x in lim], [1.1 * x for x in lim], color="0.85", label="±10 %")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("|online cc|")
    ax.set_ylabel("|lineshape cc|  (P_shape / area)")
    ax.set_title("Lineshape cc vs online cc per period (blue: |P| ≥ 0.25; grey: low P, less reliable)\nmarker size ~ log(# events)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    from datetime import datetime
    for r in table:
        c = "tab:blue" if r["reliable"] else "0.6"
        t = [datetime.fromisoformat(e["time"]) for e in r["events"]]
        ax.plot(t, [e["cc_shape"] / r["cc_online"] for e in r["events"]], "o", ms=3, color=c, alpha=0.7)
    ax.axhline(1, color="0.3", lw=0.8, ls="--")
    ax.axhspan(0.9, 1.1, color="0.85")
    ax.set_yscale("log")
    ax.set_ylabel("lineshape cc / online cc (per event)")
    ax.set_title("Per-event ratio over the run")
    ax.grid(alpha=0.3, which="both")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / "lineshape_vs_online_cc.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
