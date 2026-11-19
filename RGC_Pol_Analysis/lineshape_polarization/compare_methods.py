"""Presentation table + plot: online P vs area method (online cc) vs lineshape P, per period and per event.

Uses the fits in results/lineshape_fits.yaml (no refitting). Per period, all three methods are compared on
the same events: fits with rms > 80 µV or a weak signal (< 30 % of the period's strongest) are excluded.
"""
from collections import defaultdict
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
WEAK_FRACTION = 0.3

NOTES = {
    -3.701902: ("no result", "no strong signal"),
    -15.0: ("unresolved", "400 mV: RF saturation?"),
    -20.0: ("unresolved", "400 mV: RF saturation?"),
    -14.5: ("online wrong", "online P too high"),
    -2.36: ("online wrong", "short entry"),
    -1.122: ("unresolved", "low P; cc from unfinished TE"),
    -1.0: ("unresolved", "area drifts, P_ls constant"),
    -4.0: ("unresolved", "area drifts, low P"),
    -3.55: ("unresolved", "area drifts, low P"),
}
STATUS_COLOR = {"consistent": "#d9f2d9", "online wrong": "#fbe3c8", "unresolved": "#f2f2f2", "no result": "#e6e6e6"}


def load():
    events = []
    seen = set()
    for r in yaml.safe_load((RES / "lineshape_fits.yaml").read_text())["events"]:
        key = (r["time"], round(r["area_online"], 12))
        if key not in seen:
            seen.add(key)
            events.append(r)
    by = defaultdict(list)
    for r in events:  # -1.121803 and -1.1218 are the same period
        r["cc"] = round(r["cc"], 3) if abs(r["cc"] + 1.1218) < 1e-3 else r["cc"]
        by[r["cc"]].append(r)
    for rs in by.values():
        amax = max(abs(r["area_online"]) for r in rs)
        for r in rs:
            r["use"] = r["fit_rms_V"] <= BAD_RMS and abs(r["area_online"]) >= WEAK_FRACTION * amax
    return events, by


def period_rows(by):
    rows = []
    for cc, rs in sorted(by.items(), key=lambda kv: min(r["time"] for r in kv[1])):
        use = [r for r in rs if r["use"]] or rs
        p_on = np.array([r["P_online"] for r in use])
        p_ar = np.array([r["P_area"] for r in use])
        p_ls = np.array([r["P_lineshape"] for r in use])
        r_ar = float(np.median(p_ar / p_on))
        r_ls = float(np.median(p_ls / p_on))
        cc_ls = float(np.median([r["P_lineshape"] / r["area_online"] for r in use]))
        d_ls = float(np.median(np.abs(p_ls) - np.abs(p_on)))
        d_ar = float(np.median(np.abs(p_ar) - np.abs(p_on)))
        rel_ls = float(np.median(np.abs(p_ls) / np.abs(p_on)) - 1) * 100
        rel_ar = float(np.median(np.abs(p_ar) / np.abs(p_on)) - 1) * 100
        z = lambda v, nd: round(v, nd) + 0.0  # avoid "-0.00"
        dp = (f"lineshape {z(d_ls, 2):+.2f} ({z(rel_ls, 0):+.0f}%) · "
              f"area {z(d_ar, 2):+.2f} ({z(rel_ar, 0):+.0f}%)")
        status, reason = NOTES.get(cc, (None, ""))
        if status is None:
            status = "consistent" if abs(r_ls - 1) <= 0.12 and abs(r_ar - 1) <= 0.06 else "check"
        note = dp + (f"  — {reason}" if reason else "")
        rows.append({
            "cc": cc, "dates": f"{min(r['time'] for r in rs)[:10]} – {max(r['time'] for r in rs)[5:10]}",
            "n": len(use), "n_pol_events": None,
            "P_on": float(np.median(np.abs(p_on))), "P_ar": float(np.median(np.abs(p_ar))), "P_ls": float(np.median(np.abs(p_ls))),
            "r_ar": r_ar, "r_ls": r_ls, "cc_ls": cc_ls, "status": status, "note": note,
            "channel": ", ".join(sorted({r["channel"].replace("Deuteron5T", "5T") or "5T" for r in use})),
        })
    return rows


def table_png(rows):
    header = ["Period (dates)", "Channel", "Online\ncc", "Lineshape\ncc", "n",
              "|P| online", "|P| area\n(online cc)", "|P| lineshape\n(no cc)",
              "area /\nonline", "lineshape /\nonline", "Verdict", "ΔP vs online  (Δ|P| and relative)"]
    cells, colors = [], []
    for r in rows:
        cells.append([r["dates"], r["channel"], f"{r['cc']:g}", f"{r['cc_ls']:+.2f}", str(r["n"]),
                      f"{r['P_on']:.2f}", f"{r['P_ar']:.2f}", f"{r['P_ls']:.2f}",
                      f"{r['r_ar']:.2f}", f"{r['r_ls']:.2f}", r["status"], r["note"]])
        colors.append([STATUS_COLOR.get(r["status"], "white")] * len(header))
    widths = np.array([0.105, 0.07, 0.045, 0.06, 0.022, 0.05, 0.06, 0.07, 0.045, 0.055, 0.07, 0.33])
    h = 0.36 * (len(rows) + 1.6)
    fig = plt.figure(figsize=(21, h + 1.3))
    ax = fig.add_axes([0.005, 0.9 / (h + 1.3), 0.99, h / (h + 1.3)])
    ax.set_axis_off()
    t = ax.table(cellText=cells, colLabels=header, cellColours=colors, cellLoc="center",
                 colWidths=list(widths / widths.sum()), bbox=[0, 0, 1, 1])
    t.auto_set_font_size(False)
    t.set_fontsize(10.5)
    for (i, j), c in t.get_celld().items():
        if i == 0:
            c.set_facecolor("#3b4a6b")
            c.set_text_props(color="white", weight="bold")
        if j == 11 and i > 0:
            c.set_text_props(ha="left")
            c._loc = "left"
        if j in (3, 9) and i > 0:
            c.set_text_props(weight="bold")
    fig.suptitle("RGC deuteron polarization: online vs area method vs lineshape (6 high-signal events per period)",
                 fontsize=15, weight="bold", y=1 - 0.15 / (h + 1.3))
    fig.text(0.005, 0.15 / (h + 1.3),
             "Area method: raw-sweep Q-meter + Dulya fit, P = online cc × Σχ″.   Lineshape: P from the spin-1 transition "
             "ratio of the same fit (no cc, no area).   Lineshape cc = median(P_lineshape / online area).\n"
             "Medians over the same events per period (fits with rms > 80 µV or weak signal excluded).   "
             "ΔP = |P_method| − |P_online| (absolute polarization), relative = |P_method| / |P_online| − 1.   "
             "Green: methods agree · orange: online cc inconsistent · grey: cannot be resolved with the lineshape.",
             fontsize=10, color="0.25")
    fig.savefig(PLOTS / "comparison_table.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def scatter_png(events, rows):
    status = {r["cc"]: r["status"] for r in rows}
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.6))
    lim = 1.05
    for ax, key, title in [(axes[0], "P_area", "Area method (online cc, physical background)"),
                           (axes[1], "P_lineshape", "Lineshape (no cc)")]:
        ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=0.8, ls="--")
        for st, c, lab in [("consistent", "tab:green", "consistent periods"), ("online wrong", "tab:orange", "online cc inconsistent"),
                           ("unresolved", "0.6", "unresolved"), ("no result", "0.8", None)]:
            sel = [r for r in events if r["use"] and status.get(r["cc"]) == st]
            ax.plot([r["P_online"] for r in sel], [r[key] for r in sel], "o", color=c, ms=6, alpha=0.85, label=lab if sel else None)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("online polarization")
        ax.set_ylabel("polarization from fit")
        ax.set_title(title, fontsize=12)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("Per event: fitted polarization vs online polarization", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "comparison_scatter.png", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main():
    PLOTS.mkdir(exist_ok=True)
    events, by = load()
    rows = period_rows(by)
    table_png(rows)
    scatter_png(events, rows)
    (RES / "comparison_per_period.yaml").write_text(yaml.safe_dump(rows, sort_keys=False))
    for r in rows:
        print(f"{r['dates']:24s} {r['channel']:12s} cc {r['cc']:9.4f} cc_ls {r['cc_ls']:+7.2f} n {r['n']}  "
              f"|P| on {r['P_on']:.2f} ar {r['P_ar']:.2f} ls {r['P_ls']:.2f}  ratios {r['r_ar']:.2f} {r['r_ls']:.2f}  {r['status']}")


if __name__ == "__main__":
    main()
