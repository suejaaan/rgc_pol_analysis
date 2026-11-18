"""Tables and plots for run_lineshape.py: lineshape polarization per event and lineshape cc per period."""
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
PLOTS = HERE / "plots"
BAD_RMS = 80e-6
LOW_P = 0.25
WEAK_FRACTION = 0.3   # events with |area| < 30 % of the period's strongest fitted event


def flags(r):
    f = []
    if r["fit_rms_V"] > BAD_RMS:
        f.append("bad fit")
    if abs(r["P_lineshape"]) < LOW_P:
        f.append("low P")
    if abs(r["P_lineshape"]) > 0.94:
        f.append("at bound")
    return f


def main():
    PLOTS.mkdir(exist_ok=True)
    data = yaml.safe_load((RES / "lineshape_fits.yaml").read_text())
    seen, events = set(), []
    for r in data["events"]:  # live current_* files can overlap a run file under a different name
        key = (r["time"], round(r["area_online"], 12))
        if key not in seen:
            seen.add(key)
            events.append(r)
    for r in events:
        r["cc_lineshape"] = r["P_lineshape"] / r["area_online"]
        r["flags"] = flags(r)
    by = defaultdict(list)
    for r in events:
        by[r["cc"]].append(r)
    for rs in by.values():
        amax = max(abs(r["area_online"]) for r in rs)
        for r in rs:
            if abs(r["area_online"]) < WEAK_FRACTION * amax:
                r["flags"].append("weak signal")

    rows = []
    for cc, rs in sorted(by.items(), key=lambda kv: min(r["time"] for r in kv[1])):
        good = [r for r in rs if not r["flags"]]
        use = good if good else rs
        ccl = np.array([r["cc_lineshape"] for r in use])
        rows.append({
            "cc_online": cc, "first": min(r["time"] for r in rs)[:10], "last": max(r["time"] for r in rs)[:10],
            "n_fitted": len(rs), "n_used": len(use), "only_flagged": not good,
            "median_abs_P_lineshape": float(np.median([abs(r["P_lineshape"]) for r in use])),
            "cc_lineshape": float(np.median(ccl)),
            "cc_lineshape_err": float(1.2533 * np.std(ccl) / np.sqrt(len(ccl))) if len(ccl) > 1 else float("nan"),
            "cc_lineshape_spread": float(np.std(ccl)),
            "median_fit_rms_uV": float(np.median([r["fit_rms_V"] for r in rs]) * 1e6),
        })
    (RES / "lineshape_cc_per_period.yaml").write_text(yaml.safe_dump(rows, sort_keys=False))

    lines = ["| period (online cc) | dates | events used / fitted | median \\|P\\| lineshape | **cc from lineshape** | spread | fit rms | note |",
             "|---|---|---|---|---|---|---|---|"]
    for p in rows:
        note = "only flagged fits (low P / bad fit)" if p["only_flagged"] else ""
        lines.append(f"| {p['cc_online']:g} | {p['first']}..{p['last'][5:]} | {p['n_used']}/{p['n_fitted']} | "
                     f"{p['median_abs_P_lineshape']:.2f} | **{p['cc_lineshape']:+.2f} ± {p['cc_lineshape_err']:.2f}** | "
                     f"{p['cc_lineshape_spread']:.2f} | {p['median_fit_rms_uV']:.0f} µV | {note} |")
    ev = ["| time | online cc | P lineshape | ± | fit rms | flags |", "|---|---|---|---|---|---|"]
    for r in events:
        ev.append(f"| {r['time'][:16]} | {r['cc']:g} | {r['P_lineshape']:+.3f} | {r['P_lineshape_err']:.3f} | "
                  f"{r['fit_rms_V'] * 1e6:.0f} µV | {', '.join(r['flags'])} |")
    text = "## Lineshape cc per period\n\n" + "\n".join(lines) + "\n\n## Lineshape polarization per event\n\n" + "\n".join(ev) + "\n"
    (PLOTS / "lineshape_tables.md").write_text(text)
    print("\n".join(lines))

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
    ccs = [p["cc_online"] for p in rows]
    cmap = plt.cm.tab20.colors
    col = {cc: cmap[i % 20] for i, cc in enumerate(ccs)}
    for r in events:
        t = datetime.fromisoformat(r["time"])
        m = "o" if not r["flags"] else "x"
        axes[0].errorbar(t, r["P_lineshape"], yerr=r["P_lineshape_err"], fmt=m, color=col[r["cc"]], ms=5, capsize=2)
    axes[0].axhline(0, color="0.5", lw=0.7)
    axes[0].axhspan(-LOW_P, LOW_P, color="0.92", zorder=0)
    axes[0].set_ylabel("P from lineshape (no cc)")
    axes[0].set_title("Lineshape polarization of high-signal events (x = flagged: low P, bad fit or at bound; grey band = |P| < 0.25)")
    axes[0].grid(alpha=0.3)
    for p in rows:
        t0, t1 = datetime.fromisoformat(p["first"]), datetime.fromisoformat(p["last"])
        tm = t0 + (t1 - t0) / 2
        axes[1].errorbar(tm, -p["cc_lineshape"], yerr=p["cc_lineshape_err"], fmt="o" if not p["only_flagged"] else "x",
                         color=col[p["cc_online"]], ms=7, capsize=3)
        axes[1].annotate(f"{p['cc_online']:g}", (tm, -p["cc_lineshape"]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    axes[1].set_yscale("log")
    axes[1].set_ylabel("|cc| from lineshape")
    axes[1].set_title("Calibration constant implied by the lineshape, per period (label = online cc of that period)")
    axes[1].grid(alpha=0.3, which="both")
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(PLOTS / "lineshape_polarization.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"\nwrote {PLOTS}")


if __name__ == "__main__":
    main()
