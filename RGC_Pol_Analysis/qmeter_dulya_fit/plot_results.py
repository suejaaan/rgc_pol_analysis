"""Comparison plots for fit_events.py output. Writes PNGs to plots/ and a markdown table to results/.

Usage: python plot_results.py [summary.yaml] [plot_dir]
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
RES_DIR = HERE / "results"
PLOT_DIR = HERE / "plots"
SIGNAL_REGION_MHZ = (32.48, 32.90)

C_ONLINE, C_AREA, C_FIT = "0.35", "tab:blue", "tab:red"


def load(summary_path):
    summary = yaml.safe_load(Path(summary_path).read_text())
    arrays = {i: dict(np.load(RES_DIR / f"event_{i}_arrays.npz")) for i in summary}
    return summary, arrays


def shade_signal(ax):
    ax.axvspan(*SIGNAL_REGION_MHZ, color="0.9", zorder=0)


def plot_fit_overview(summary, arrays):
    ev = list(summary)
    fig, axes = plt.subplots(len(ev), 2, figsize=(14, 3.8 * len(ev)), squeeze=False,
                             gridspec_kw={"width_ratios": [1.3, 1]})
    for row, i in zip(axes, ev):
        a, s = arrays[i], summary[i]
        f = a["f"]
        online_bg = a["online_baseline"] + a["online_fitcurve"]
        ax = row[0]
        shade_signal(ax)
        ax.plot(f, a["raw"], color="k", lw=1.4, label="raw sweep (data)")
        ax.plot(f, a["model"], color=C_FIT, lw=1.0, ls="--", label="Q-meter + Dulya fit")
        ax.plot(f, a["background"], color=C_AREA, lw=1.2, label="fitted Q-meter background (χ = 0)")
        ax.plot(f, online_bg, color=C_ONLINE, lw=1.0, ls=":", label="online background (baseline + wing poly)")
        ax.set_title(f"evt {i}: raw sweep fit   (rms {s['fit_rms_V'] * 1e6:.0f} µV)")
        ax.set_ylabel("Q-meter output [V]")

        ax = row[1]
        shade_signal(ax)
        resid = (a["raw"] - a["model"]) * 1e6
        ax.plot(f, resid, color="k", lw=0.8)
        ax.axhline(0, color="0.5", lw=0.7)
        ax.axhspan(-a["sigma"] * 1e6, a["sigma"] * 1e6, color=C_FIT, alpha=0.15, label=f"±σ_fit = {a['sigma'] * 1e6:.0f} µV")
        peak = np.max(np.abs(a["signal"])) * 1e6
        ax.set_title(f"residual raw − fit   (signal peak ≈ {peak:.0f} µV)")
        ax.set_ylabel("µV")
        for x in row:
            x.set_xlabel("frequency [MHz]")
            x.grid(alpha=0.3)
            x.legend(fontsize=8)
    fig.suptitle("Q-meter + Dulya fit to the raw sweep (grey band = signal region, white = wings)", fontsize=12)
    fig.tight_layout()
    return save(fig, "1_raw_sweep_fit.png")


def plot_signal_comparison(summary, arrays):
    ev = list(summary)
    fig, axes = plt.subplots(len(ev), 3, figsize=(18, 4 * len(ev)), squeeze=False)
    for row, i in zip(axes, ev):
        a, s = arrays[i], summary[i]
        f = a["f"]
        data_sig = (a["raw"] - a["background"]) * 1e3

        ax = row[0]
        ax.plot(f, data_sig, color="k", lw=1.0, label="raw − fitted Q-meter bg")
        ax.plot(f, a["signal"] * 1e3, color=C_FIT, lw=1.3, label="fitted signal")
        ax.plot(f, a["online_fitsub"] * 1e3, color=C_ONLINE, lw=1.0, ls="--", label="online fitsub")
        ax.set_title(f"evt {i}: extracted signal\nΣ χ″ part = {s['area_absorb']:.4f}   Σ total = {s['area_total_output']:.4f}   Σ online = {s['area_online']:.4f}")
        ax.set_ylabel("mV")

        ax = row[1]
        ax.plot(f, a["absorb"] * 1e3, color=C_AREA, lw=1.2, label="from χ″ (absorption)")
        ax.plot(f, a["disperse"] * 1e3, color="tab:green", lw=1.2, label="from χ′ (dispersion, Q-meter mixing)")
        ax.plot(f, a["signal"] * 1e3, color=C_FIT, lw=0.8, ls="--", label="total")
        ax.axhline(0, color="0.5", lw=0.7)
        title = "signal decomposition"
        if sig := s.get("signal"):
            title += f"\nη = {sig['eta_q']:.3f}, g = {sig['g']:.4f}, wq = {sig['wq'] * 1e3:.1f} kHz, f0 = {sig['f0']:.4f} MHz"
        ax.set_title(title)
        ax.set_ylabel("mV")

        ax = row[2]
        online_bg = a["online_baseline"] + a["online_fitcurve"]
        ax.plot(f, (a["background"] - online_bg) * 1e6, color="tab:purple", lw=1.2,
                label="Q-meter bg − online bg")
        ax.plot(f, (a["signal"] - a["online_fitsub"]) * 1e6, color="k", lw=0.8, alpha=0.7,
                label="fitted signal − online fitsub")
        ax.axhline(0, color="0.5", lw=0.7)
        shade_signal(ax)
        ax.set_title("where the two backgrounds disagree")
        ax.set_ylabel("µV")
        for x in row:
            x.set_xlabel("frequency [MHz]")
            x.grid(alpha=0.3)
            x.legend(fontsize=8)
    fig.tight_layout()
    return save(fig, "2_signal_comparison.png")


def plot_polarization(summary):
    ev = list(summary)
    x = np.arange(len(ev))
    p_on = np.array([summary[i]["P_online"] for i in ev]) * 100
    p_ar = np.array([summary[i]["P_area"] for i in ev]) * 100
    e_ar = np.array([summary[i]["P_area_err_stat"] for i in ev]) * 100
    labels = [f"evt {i}\n{summary[i]['start_time'][11:16]} UTC" for i in ev]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    ax = axes[0]
    ax.plot(x - 0.08, p_on, "s", color=C_ONLINE, ms=8, label="online (cc × Σ fitsub)")
    ax.errorbar(x + 0.08, p_ar, yerr=e_ar, fmt="o", color=C_AREA, ms=8, capsize=4,
                label="area method (cc × Σ fitted χ″ part)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("polarization [%]")
    ax.set_title("Polarization: online vs Q-meter + Dulya area method")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1]
    rel = (np.abs(p_ar) - np.abs(p_on)) / np.abs(p_on) * 100
    ax.errorbar(x, rel, yerr=e_ar / np.abs(p_on) * 100, fmt="o", color=C_AREA, ms=8, capsize=4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(x, labels)
    ax.set_ylabel("(|P_area| − |P_online|) / |P_online| [%]")
    ax.set_title("Relative difference (same cc; isolates the background treatment)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save(fig, "3_polarization_comparison.png")


def write_table(summary):
    lines = [
        "| event | time (UTC) | cc | P online | P area (fit, χ″) | Δ|P| / |P online| | Σ online | Σ χ″ fit | Σ total fit output |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for i, s in summary.items():
        rel = (abs(s["P_area"]) - abs(s["P_online"])) / abs(s["P_online"]) * 100
        lines.append(
            f"| {i} | {s['start_time'][:19]} | {s['cc_online']:.3f} | {s['P_online']:+.4f} | "
            f"{s['P_area']:+.4f} ± {s['P_area_err_stat']:.4f} | {rel:+.1f} % | "
            f"{s['area_online']:.4f} | {s['area_absorb']:.4f} | {s['area_total_output']:.4f} |"
        )
    path = PLOT_DIR / "results_table.md"
    path.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return path


def save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


def main():
    global PLOT_DIR
    summary_path = sys.argv[1] if len(sys.argv) > 1 else RES_DIR / "results.yaml"
    if len(sys.argv) > 2:
        PLOT_DIR = Path(sys.argv[2]).resolve()
    summary, arrays = load(summary_path)
    plot_fit_overview(summary, arrays)
    plot_signal_comparison(summary, arrays)
    plot_polarization(summary)
    write_table(summary)


if __name__ == "__main__":
    main()
