"""Polarization from the deuteron lineshape alone (no online cc), for high-P events in every period.

The raw sweep is fitted with the bounded Q-meter + Dulya model (chi'' and its Kramers-Kronig chi'
inside the coil, so Q-meter mixing cannot fake horn asymmetry). P follows from the spin-1 transition
ratio, P = (r^2 - 1) / (r^2 + r + 1); no cc, no area. eta and wq are fixed to their period-stable values.
Per period: cc_lineshape = median(P_lineshape / online area).

The same fits also store P_area (online cc x sum chi'') and the online pol, for later comparison.

Usage: python run_lineshape.py [cc ...]    (no args: all periods; with cc values: rerun only those periods)
"""
import os
import csv
import json
import sys
import time
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "multi_period_fit"))
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import fit_selected as FS  # noqa: E402
import model as M  # noqa: E402

DATA = HERE.parents[1] / "data_creation/data-d"
EVENT_MAP = HERE.parent / "baseline_refit/cache/event_baseline_map.csv"
OUT = HERE / "results"

N_PER_PERIOD = 6
OUTLIER_FACTOR = 1.5   # drop glitch events: |area| > 1.5 x the period's 95th percentile
MIN_SPACING_H = 0.5    # picked events at least 30 min apart
MIN_PERIOD_EVENTS = 100
SKIP_CC = {-5.246731: "online processing unusable (Jun - early Jul 2022)"}
CHANNELS = ("Deuteron5T", "Deuteron5T-300mV", "Deuteron5T-400mV", "Deuteron5T-150mV")
ETA_FIXED, WQ_FIXED = 0.108, 0.0825
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 4) - 2)))  # all cores but 2


def select():
    want, base_of = defaultdict(set), {}
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
                if d["channel"]["name"] not in CHANNELS or not (32.29 < f[0] < 32.31 and 33.09 < f[-1] < 33.11):
                    continue
                if abs(d["cc"] * d["area"] - d["pol"]) > 1e-9 or not np.any(d["fitsub"]):
                    continue
                by_cc[float(d["cc"])].append({
                    "file": fname, "index": i, "time": d["start_time"][:19], "cc": float(d["cc"]),
                    "pol": float(d["pol"]), "area": float(d["area"]), "baseline_id": base_of[(fname, i)],
                    "channel": d["channel"]["name"], "period_note": ""})
    picks, periods = [], {}
    for cc, evs in sorted(by_cc.items()):
        periods[cc] = {"n_polarize": len(evs), "first": min(e["time"] for e in evs), "last": max(e["time"] for e in evs)}
        if cc in SKIP_CC or len(evs) < MIN_PERIOD_EVENTS:
            periods[cc]["skipped"] = SKIP_CC.get(cc, f"< {MIN_PERIOD_EVENTS} events")
            continue
        uniq = {}
        for e in evs:  # live current_* files can repeat events under another file name
            uniq.setdefault(e["time"], e)
        evs = list(uniq.values())
        a95 = np.percentile([abs(e["area"]) for e in evs], 95)
        evs = [e for e in evs if abs(e["area"]) <= OUTLIER_FACTOR * a95]
        periods[cc]["n_after_dedup_and_outlier_cut"] = len(evs)

        def spaced(e, chosen):
            t = np.datetime64(e["time"])
            return all(abs((t - np.datetime64(c["time"])) / np.timedelta64(1, "h")) >= MIN_SPACING_H for c in chosen)

        chosen = []
        for sign in (+1, -1):
            for e in sorted([e for e in evs if np.sign(e["area"]) == sign], key=lambda e: -abs(e["area"])):
                if sum(np.sign(x["area"]) == sign for x in chosen) >= N_PER_PERIOD // 2:
                    break
                if spaced(e, chosen):
                    chosen.append(e)
        for e in sorted(evs, key=lambda e: -abs(e["area"])):  # one sign only: fill up
            if len(chosen) >= N_PER_PERIOD:
                break
            if e not in chosen and spaced(e, chosen):
                chosen.append(e)
        picks += chosen
    return picks, periods


def init_worker():
    lo, hi = list(M.SIGNAL_LO), list(M.SIGNAL_HI)
    i_eta, i_wq = M.SIGNAL_NAMES.index("eta_q"), M.SIGNAL_NAMES.index("wq")
    lo[i_eta], hi[i_eta] = ETA_FIXED - 1e-6, ETA_FIXED + 1e-6
    lo[i_wq], hi[i_wq] = WQ_FIXED - 1e-6, WQ_FIXED + 1e-6
    M.SIGNAL_LO, M.SIGNAL_HI = tuple(lo), tuple(hi)
    FS.SHAPE0.update(eta_q=ETA_FIXED, wq=WQ_FIXED)
    FS.OUT = OUT


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    picks, periods = select()
    only = {float(a) for a in sys.argv[1:]}
    previous = []
    if only:  # rerun selected periods only; keep the other periods' fits
        old = yaml.safe_load((OUT / "lineshape_fits.yaml").read_text())
        previous = [r for r in old["events"] if not any(abs(r["cc"] - c) < 1e-4 for c in only)]
        picks = [e for e in picks if any(abs(e["cc"] - c) < 1e-4 for c in only)]
    print(f"selected {len(picks)} events from {sum('skipped' not in p for p in periods.values())} periods "
          f"in {time.time() - t0:.0f} s", flush=True)
    base = yaml.safe_load(FS.BASE_FITS.read_text())["baselines"]
    jobs = []
    for e in picks:
        b = base.get(e["baseline_id"])
        jobs.append((e, [b["params"][n] for n in M.CIRCUIT_NAMES] if b else None))
    results = []
    with Pool(N_WORKERS, initializer=init_worker) as pool:
        for r in pool.imap_unordered(FS.fit_one, jobs):
            results.append(r)
            print(f"  {r['time']} cc {r['cc']:9.4f}  P_lineshape {r['P_lineshape']:+.3f} ± {r['P_lineshape_err']:.3f}  "
                  f"(online {r['P_online']:+.3f})  rms {r['fit_rms_V'] * 1e6:.0f} µV  [{r['fit_seconds']:.0f} s]", flush=True)
    results = sorted(previous + results, key=lambda r: r["time"])
    (OUT / "lineshape_fits.yaml").write_text(yaml.safe_dump({"periods": periods, "events": results}, sort_keys=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT / 'lineshape_fits.yaml'}")


if __name__ == "__main__":
    main()
