"""Stage 0: event inventory and tune segments (no fitting, no cc used for any decision).

Reads every event in data-d once and writes one row per unique event with:
  identity (file, index, time), label, channel, stored baseline id, online area / pol / cc (kept only as
  metadata), Q-meter tune readings, microwave, beam, temperature and dose, plus quality flags.

Tune segment = consecutive events with the same channel and the same stored baseline sweep, split where the
time gap exceeds GAP_H. Inside a segment the circuit (gain, phase) is expected to be unchanged, so the online
area is proportional to vector P there without needing a cc.

The online-cc period and its verdict (lineshape_polarization) are carried as columns for reference only.

Outputs: results/events.csv, results/segments.csv, plots/stage0_segments.png
"""
import os
import csv
import json
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "baseline_refit"))
import collect_baselines as CB  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT.parent / "data_creation/data-d"
EVENT_MAP = ROOT / "baseline_refit/cache/event_baseline_map.csv"
VERDICTS = ROOT / "lineshape_polarization/results/comparison_per_period.yaml"
RES, PLOTS = HERE / "results", HERE / "plots"

GAP_H = 6.0            # split a segment where no event was taken for this long
OUTLIER_FACTOR = 1.5   # glitch: |area| > 1.5 x the segment's 95th percentile
STANDARD_CHANNEL = "Deuteron5T"
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 4) - 2)))  # all cores but 2

COLS = ["time", "file", "index", "label", "channel", "baseline_id", "segment", "cc", "period_verdict",
        "area_online", "pol_online", "sign", "diode_vout", "phase_vout", "uwave_freq", "uwave_power",
        "beam_current", "nmr_T", "dose", "sweeps", "freq_ok", "fitsub_ok", "pol_eq_cc_area", "glitch",
        "duplicate_of", "candidate"]


def read_file(job):
    fname, base_of = job
    rows = []
    with open(DATA / fname) as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            f = d.get("freq_list") or [0]
            ep = d.get("epics") or {}
            rows.append({
                "time": d["start_time"][:19], "file": fname, "index": i, "label": d.get("label", ""),
                "channel": d["channel"]["name"], "baseline_id": base_of.get(i, -1),
                "cc": float(d.get("cc", np.nan)), "area_online": float(d.get("area", np.nan)),
                "pol_online": float(d.get("pol", np.nan)),
                "diode_vout": d.get("diode_vout"), "phase_vout": d.get("phase_vout"),
                "uwave_freq": d.get("uwave_freq"), "uwave_power": d.get("uwave_power"),
                "beam_current": ep.get("scaler_calc1"), "nmr_T": ep.get("TGT:PT12:NMR_T"),
                "dose": ep.get("TGT:PT12:Target_Dose"), "sweeps": d.get("sweeps"),
                "freq_ok": len(f) == 512 and 32.29 < f[0] < 32.31 and 33.09 < f[-1] < 33.11,
                "fitsub_ok": bool(np.any(d.get("fitsub") or [0])),
                "pol_eq_cc_area": abs(d.get("cc", 0) * d.get("area", 0) - d.get("pol", 0)) < 1e-9,
            })
    return rows


def verdict_of(cc, channel):
    if not VERDICTS.exists():
        return ""
    for r in yaml.safe_load(VERDICTS.read_text()):
        if abs(r["cc"] - cc) < 1e-3:
            if abs(cc + 14.5) < 1e-6:  # the -14.5 period: wrong on 5T, unresolved on the 400 mV channel
                return "online wrong" if channel == STANDARD_CHANNEL else "unresolved"
            return r["status"]
    return "no result"


def main():
    RES.mkdir(parents=True, exist_ok=True)
    PLOTS.mkdir(parents=True, exist_ok=True)
    base_of = defaultdict(dict)
    with open(EVENT_MAP) as fh:
        for r in csv.DictReader(fh):
            base_of[r["file"]][int(r["event_index"])] = int(r["baseline_id"])
    files = CB.unique_data_files()  # same file list (and event indices) as the baseline map
    with Pool(N_WORKERS) as pool:
        rows = [r for rs in pool.map(read_file, [(f, base_of[f]) for f in files]) for r in rs]
    print(f"read {len(rows)} event records from {len(files)} non-empty files")

    # duplicates: live current_* files repeat events of the run files (same start time); keep the run-file copy
    rows.sort(key=lambda r: (r["time"], r["file"].startswith("current_"), r["file"], r["index"]))
    first = {}
    for r in rows:
        k = (r["time"], r["channel"])
        r["duplicate_of"] = f"{first[k]['file']}:{first[k]['index']}" if k in first else ""
        first.setdefault(k, r)
    uniq = [r for r in rows if not r["duplicate_of"]]

    # tune segments: same channel + same stored baseline, contiguous in time
    seg, prev, n_seg = {}, None, -1
    for r in uniq:
        key = (r["channel"], r["baseline_id"])
        t = np.datetime64(r["time"])
        if prev is None or key != prev[0] or (t - prev[1]) / np.timedelta64(1, "h") > GAP_H:
            n_seg += 1
        r["segment"] = n_seg
        prev = (key, t)

    vcache = {}
    by_seg = defaultdict(list)
    for r in uniq:
        k = (r["cc"], r["channel"])
        if k not in vcache:
            vcache[k] = verdict_of(*k)
        r["period_verdict"] = vcache[k]
        r["sign"] = int(np.sign(r["area_online"])) if np.isfinite(r["area_online"]) else 0
        if r["label"] == "Polarize":
            by_seg[r["segment"]].append(r)
    for rs in by_seg.values():
        a95 = np.percentile([abs(r["area_online"]) for r in rs], 95)
        for r in rs:
            r["glitch"] = abs(r["area_online"]) > OUTLIER_FACTOR * a95
    for r in uniq:
        r.setdefault("glitch", False)
        r["candidate"] = (r["label"] == "Polarize" and r["channel"] == STANDARD_CHANNEL and r["freq_ok"]
                          and r["fitsub_ok"] and r["pol_eq_cc_area"] and not r["glitch"] and r["baseline_id"] >= 0)

    with open(RES / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLS)
        w.writeheader()
        for r in rows:
            r.setdefault("segment", -1)
            r.setdefault("period_verdict", "")
            r.setdefault("sign", 0)
            r.setdefault("glitch", False)
            r.setdefault("candidate", False)
            w.writerow({c: r[c] for c in COLS})

    segs = []
    for s in sorted({r["segment"] for r in uniq}):
        rs = [r for r in uniq if r["segment"] == s]
        cand = [r for r in rs if r["candidate"]]
        if not cand:
            continue
        a = np.array([r["area_online"] for r in cand])
        segs.append({
            "segment": s, "channel": rs[0]["channel"], "baseline_id": rs[0]["baseline_id"],
            "start": rs[0]["time"], "end": rs[-1]["time"], "n_events": len(rs), "n_candidates": len(cand),
            "n_pos": int((a > 0).sum()), "n_neg": int((a < 0).sum()),
            "area_min": float(a.min()), "area_max": float(a.max()),
            "online_cc": ";".join(f"{c:g}" for c in sorted({r["cc"] for r in cand})),
            "period_verdict": ";".join(sorted({r["period_verdict"] for r in cand})),
        })
    with open(RES / "segments.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(segs[0]))
        w.writeheader()
        w.writerows(segs)

    # ---- summary ----
    cand = [r for r in uniq if r["candidate"]]
    print(f"unique events {len(uniq)}  (duplicates removed {len(rows) - len(uniq)})")
    print("labels:", dict(Counter(r["label"] for r in uniq).most_common()))
    pol = [r for r in uniq if r["label"] == "Polarize"]
    print(f"Polarize {len(pol)}: channel {dict(Counter(r['channel'] for r in pol))}")
    print(f"  rejected: bad freq {sum(not r['freq_ok'] for r in pol)}, empty fitsub {sum(not r['fitsub_ok'] for r in pol)}, "
          f"pol != cc*area {sum(not r['pol_eq_cc_area'] for r in pol)}, glitch {sum(r['glitch'] for r in pol)}")
    print(f"candidates (Polarize, 5T, clean) {len(cand)} in {len(segs)} tune segments")
    print("  by period verdict:", dict(Counter(r["period_verdict"] for r in cand)))
    print(f"  sign: + {sum(r['sign'] > 0 for r in cand)}, - {sum(r['sign'] < 0 for r in cand)}")
    sizes = np.array([s["n_candidates"] for s in segs])
    print(f"  segment size: median {np.median(sizes):.0f}, segments with >= 100 candidates: {(sizes >= 100).sum()} "
          f"({sizes[sizes >= 100].sum()} events)")

    # ---- plot: |online area| vs time, coloured by segment ----
    fig, ax = plt.subplots(figsize=(18, 6))
    cmap = plt.cm.tab20(np.linspace(0, 1, 20))
    for i, s in enumerate(segs):
        rs = [r for r in cand if r["segment"] == s["segment"]]
        t = np.array([np.datetime64(r["time"]) for r in rs])
        a = np.array([r["area_online"] for r in rs])
        ax.plot(t, a, ".", ms=1.5, color=cmap[i % 20])
    shade = {"online wrong": "#fbe3c8", "unresolved": "#e6e6e6", "no result": "#f4f4f4"}
    for s in segs:
        v = s["period_verdict"].split(";")[0]
        if v in shade:
            ax.axvspan(np.datetime64(s["start"]), np.datetime64(s["end"]), color=shade[v], zorder=0, lw=0)
    ax.axhline(0, color="0.5", lw=0.6)
    ax.set_ylabel("online area (arb.; sign = sign of P)")
    ax.set_title(f"Stage 0: {len(cand)} candidate events (Polarize, 5T channel, clean) in {len(segs)} tune segments "
                 f"(colour = segment; orange band = online cc wrong, grey = unresolved)", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS / "stage0_segments.png", dpi=150)
    print(f"wrote {RES / 'events.csv'}, {RES / 'segments.csv'}, {PLOTS / 'stage0_segments.png'}")


if __name__ == "__main__":
    sys.exit(main())
