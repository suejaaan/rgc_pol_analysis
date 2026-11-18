"""Survey every TE run in data-d and calibrate cc only where the NMR area reaches a plateau.

TE events (label "TE") are grouped into runs: a new run starts after a gap > GAP_H hours.
For each run, after dropping glitch events, we look at the last LAST_H hours:

  plateau      : |relative change per hour| over the last window < MAX_REL_SLOPE
  cc_TE        : P_TE(T_vp, end) / mean area over the last window
  ratio_online : mean area / area expected at equilibrium with the online cc

Runs that never flatten are reported but not calibrated (see te_run_analysis.py for why
extrapolating an unfinished relaxation does not constrain cc).

Requires ../baseline_refit/cache/event_baseline_map.csv (run collect_baselines.py first).
"""
import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data_creation/data-d"
EVENT_MAP = HERE.parent / "baseline_refit/cache/event_baseline_map.csv"
OUT = HERE / "results" / "survey"

H, KB, NU = 6.62607015e-34, 1.380649e-23, 32.7e6
T_KEYS = ("TGT:PT12:VaporPressure_T", "TGT:PT12:Bath_Top_T", "TGT:PT12:Bath_Bottom_T")
GAP_H = 1.5
LAST_H = 0.75
MIN_EVENTS = 15
MAX_REL_SLOPE = 0.05
GLITCH_FACTOR = 5.0


def p_te(T):
    x = H * NU / (KB * T)
    return 2 * np.sinh(x) / (1 + 2 * np.cosh(x))


def load_te_events():
    wanted = defaultdict(set)
    with open(EVENT_MAP) as fh:
        for row in csv.DictReader(fh):
            if row["label"] == "TE":
                wanted[row["file"]].add(int(row["event_index"]))
    events = []
    for fname, idxs in wanted.items():
        with open(DATA / fname) as fh:
            for i, line in enumerate(fh):
                if i not in idxs:
                    continue
                d = json.loads(line)
                slow = d.get("epics") or d.get("status") or {}
                events.append({
                    "t": datetime.fromisoformat(d["start_time"][:19]),
                    "area": float(d["area"]), "cc": float(d["cc"]), "pol": float(d["pol"]),
                    "T": [float(slow[k]) if isinstance(slow.get(k), (int, float)) else np.nan for k in T_KEYS],
                    "uwave": d.get("uwave_power"), "file": fname, "index": i, "channel": d["channel"]["name"],
                })
    return sorted(events, key=lambda e: e["t"])


def split_runs(events):
    runs, cur = [], [events[0]]
    for e in events[1:]:
        if (e["t"] - cur[-1]["t"]).total_seconds() / 3600 > GAP_H:
            runs.append(cur)
            cur = []
        cur.append(e)
    runs.append(cur)
    return runs


def drop_glitches(run):
    a = np.array([e["area"] for e in run])
    med = np.median(a)
    mad = np.median(np.abs(a - med)) + 1e-12
    return [e for e in run if abs(e["area"] - med) < GLITCH_FACTOR * 10 * mad] if len(run) > 5 else run


def analyse(run):
    clean = drop_glitches(run)
    t = np.array([(e["t"] - clean[0]["t"]).total_seconds() / 3600 for e in clean])
    a = np.array([e["area"] for e in clean])
    T_vp = np.array([e["T"][0] for e in clean])
    T_bath = np.nanmean([[e["T"][1], e["T"][2]] for e in clean], axis=1)
    ccs = sorted({e["cc"] for e in run})
    info = {
        "start": str(run[0]["t"]), "end": str(run[-1]["t"]), "hours": float(t[-1]) if len(t) else 0.0,
        "n_events": len(run), "n_clean": len(clean), "channel": run[0]["channel"], "cc_online": ccs,
        "files": sorted({e["file"] for e in run}),
    }
    last = t >= t[-1] - LAST_H if len(t) else np.array([], bool)
    if len(clean) < MIN_EVENTS or last.sum() < 5 or t[-1] < LAST_H:
        info["status"] = "too short"
        return info, clean
    slope, icpt = np.polyfit(t[last], a[last], 1)
    a_end = float(np.mean(a[last]))
    a_err = float(np.std(a[last]) / np.sqrt(last.sum()))
    T_end = float(np.nanmean(T_vp[last]))
    T_end_bath = float(np.nanmean(T_bath[last]))
    pte = p_te(T_end)
    cc_on = ccs[-1]
    info.update({
        "area_start": float(np.mean(a[: min(5, len(a))])), "area_end": a_end, "area_end_err": a_err,
        "rel_slope_per_h": float(slope / abs(a_end)) if a_end else None,
        "T_vp_end": T_end, "T_bath_end": T_end_bath, "P_TE_percent": float(pte * 100),
        "area_expected_online_cc": float(pte / cc_on), "ratio_to_expected": float(a_end / (pte / cc_on)),
    })
    flat = abs(info["rel_slope_per_h"]) < MAX_REL_SLOPE
    if not flat:
        info["status"] = "still relaxing" if abs(info["area_start"]) > abs(a_end) else "not flat"
        return info, clean
    if np.sign(a_end) == np.sign(cc_on):
        info["status"] = "plateau, wrong sign"
        return info, clean
    cc = pte / a_end
    cc_bath = p_te(T_end_bath) / a_end
    info.update({
        "status": "plateau",
        "cc_TE": float(cc),
        "cc_TE_err_stat": float(abs(cc) * a_err / abs(a_end)),
        "cc_TE_err_thermometer": float(abs(cc - cc_bath)),
        "cc_TE_err_slope": float(abs(cc) * abs(info["rel_slope_per_h"]) * LAST_H),
    })
    return info, clean


def plot_runs(results):
    n = len(results)
    ncol = 5
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.1 * nrow), squeeze=False)
    colors = {"plateau": "tab:green", "still relaxing": "tab:orange", "not flat": "tab:red",
              "too short": "0.5", "plateau, wrong sign": "tab:purple"}
    for ax, (info, clean) in zip(axes.flat, results):
        if not clean:
            ax.set_axis_off()
            continue
        t = [(e["t"] - clean[0]["t"]).total_seconds() / 3600 for e in clean]
        ax.plot(t, [e["area"] * 1e3 for e in clean], ".", ms=2.5, color=colors[info["status"]])
        if "area_expected_online_cc" in info:
            ax.axhline(info["area_expected_online_cc"] * 1e3, color="tab:blue", ls=":", lw=1, label="TE expected, online cc")
            ax.axhline(0, color="0.6", lw=0.6)
        title = f"{info['start'][:16]} · {info['hours']:.1f} h · {info['status']}"
        if info["status"] == "plateau":
            title += f"\ncc_TE = {info['cc_TE']:.2f} (online {info['cc_online'][-1]})"
        elif "ratio_to_expected" in info:
            title += f"\nend/expected = {info['ratio_to_expected']:.1f}, T = {info['T_vp_end']:.2f} K"
        ax.set_title(title, fontsize=8, color=colors[info["status"]])
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    for ax in list(axes.flat)[n:]:
        ax.set_axis_off()
    fig.suptitle("All TE runs in data-d: online area ×10³ vs hours (blue dotted = equilibrium area expected with online cc)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "all_te_runs.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = load_te_events()
    runs = split_runs(events)
    results = [analyse(r) for r in runs]
    plot_runs(results)
    table = [info for info, _ in results]
    (OUT / "te_runs.yaml").write_text(yaml.safe_dump(table, sort_keys=False))

    print(f"{len(events)} TE events in {len(runs)} runs\n")
    print(f"{'start':16} {'h':>5} {'n':>4} {'status':18} {'T_vp':>5} {'end/exp':>8} {'cc online':>9} {'cc_TE':>14}")
    for i in table:
        cc = (f"{i['cc_TE']:.3f}±{np.hypot(i['cc_TE_err_stat'], i['cc_TE_err_thermometer']):.3f}"
              if i["status"] == "plateau" else "")
        print(f"{i['start'][:16]} {i['hours']:5.1f} {i['n_events']:4d} {i['status']:18} "
              f"{i.get('T_vp_end', float('nan')):5.2f} {i.get('ratio_to_expected', float('nan')):8.2f} "
              f"{i['cc_online'][-1]:9.3f} {cc:>14}")
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
