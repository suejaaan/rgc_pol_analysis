"""Meeting plots and tables for fit_selected.py results."""
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


def load():
    fits = yaml.safe_load((RES / "fits.yaml").read_text())
    for r in fits:
        r["arr"] = dict(np.load(RES / f"event_{r['file'][:19]}_{r['index']}.npz"))
    return fits


def period_label(cc):
    return f"cc {cc:g}"


def colors_for(fits):
    ccs = sorted({r["cc"] for r in fits})
    cmap = plt.cm.tab10.colors
    return {cc: cmap[i % 10] for i, cc in enumerate(ccs)}


def save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {PLOTS / name}")


def fig_polarization(fits, col):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    ax = axes[0]
    lim = 0.75
    ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=0.8, ls="--")
    for r in fits:
        prob = bool(r["period_note"])
        ax.errorbar(r["P_online"], r["P_area"], yerr=r["P_area_err"], fmt="o", color=col[r["cc"]], ms=7,
                    mec="k" if prob else col[r["cc"]], mew=1.5 if prob else 0.5, capsize=2)
        ax.plot(r["P_online"], r["P_lineshape"], "x", color=col[r["cc"]], ms=7, mew=1.5)
    for cc, c in col.items():
        ax.plot([], [], "o", color=c, label=period_label(cc) + ("  (problem period)" if cc == -14.5 else ""))
    ax.plot([], [], "o", color="0.5", label="● area method (cc × Σχ″, physical fit)")
    ax.plot([], [], "x", color="0.5", mew=1.5, label="× lineshape P (horn ratio, no cc)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("online polarization (cc × Σ fitsub)")
    ax.set_ylabel("polarization from Q-meter + Dulya fit of raw sweep")
    ax.set_title("Raw-sweep fit vs online, 30 events / 8 periods, both signs")
    ax.legend(fontsize=7.5, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ccs = sorted(col)
    for k, cc in enumerate(ccs):
        rs = [r for r in fits if r["cc"] == cc]
        for j, r in enumerate(rs):
            x = k + (j - (len(rs) - 1) / 2) * 0.12
            ax.errorbar(x, r["mismatch_rel"] * 100, yerr=r["P_area_err"] / abs(r["P_online"]) * 100,
                        fmt="^" if r["P_online"] > 0 else "v", color=col[cc], ms=7, capsize=2)
            ax.plot(x, (r["P_lineshape"] - r["P_online"]) / abs(r["P_online"]) * 100, "x", color=col[cc], ms=6, mew=1.3)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.axhspan(-3, 3, color="0.9", zorder=0)
    ax.set_xticks(range(len(ccs)), [period_label(c) for c in ccs], rotation=30, fontsize=8)
    ax.set_ylabel("(P_fit − P_online) / |P_online|  [%]")
    ax.set_ylim(-80, 40)
    ax.set_title("Mismatch per event (▲ P>0, ▼ P<0 area method; × lineshape P; grey = ±3 %)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "1_polarization_vs_online.png")


def pick_gallery(fits):
    picks = []
    for cc in sorted({r["cc"] for r in fits}, key=lambda c: (c != -14.5, c)):
        for sign in (1, -1):
            rs = [r for r in fits if r["cc"] == cc and np.sign(r["P_online"]) == sign]
            if rs:
                picks.append(max(rs, key=lambda r: abs(r["P_online"])))
    return picks[:8]


def fig_regeneration(fits, col):
    picks = pick_gallery(fits)
    fig, axes = plt.subplots(3, len(picks), figsize=(3.6 * len(picks), 9.5), squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 2.2, 1]})
    for k, r in enumerate(picks):
        a = r["arr"]
        f = a["f"]
        ax = axes[0, k]
        ax.plot(f, a["raw"], color="k", lw=1.6, label="original raw sweep")
        ax.plot(f, a["regenerated"], color=col[r["cc"]], lw=0.9, ls="--", label="regenerated from fit")
        ax.set_title(f"{r['time'][:10]} · {period_label(r['cc'])}\nP_online {r['P_online']:+.2f} · P_area {r['P_area']:+.2f}",
                     fontsize=8.5, color="crimson" if r["period_note"] else "k")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax = axes[1, k]
        ax.plot(f, a["fitsub_online"] * 1e3, color="k", lw=1.2, label="online fitsub (data)")
        ax.plot(f, a["fitsub_regen"] * 1e3, color=col[r["cc"]], lw=1.0, ls="--", label="online chain on regenerated")
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax = axes[2, k]
        ax.plot(f, (a["regenerated"] - a["raw"]) * 1e6, color="0.3", lw=0.6)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xlabel("MHz", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
        ax.set_title(f"regen − raw: rms {r['regen_rms_diff_V'] * 1e6:.0f} µV (noise σ {r['noise_sigma_V'] * 1e6:.0f})", fontsize=8)
    axes[0, 0].set_ylabel("raw Q-meter output [V]")
    axes[1, 0].set_ylabel("signal after online chain [mV]")
    axes[2, 0].set_ylabel("µV")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.suptitle("Signals regenerated from fitted Q-meter + Dulya parameters vs originals (red titles: cc = −14.5 problem period)", fontsize=12)
    fig.tight_layout()
    save(fig, "2_regenerated_vs_original.png")


def fig_signal_split(fits, col):
    picks = pick_gallery(fits)
    fig, axes = plt.subplots(1, len(picks), figsize=(3.6 * len(picks), 3.8), squeeze=False)
    for ax, r in zip(axes[0], picks):
        a = r["arr"]
        f = a["f"]
        ax.plot(f, (a["raw"] - a["background"]) * 1e3, color="k", lw=0.9, label="raw − fitted Q-meter bg")
        ax.plot(f, a["absorb"] * 1e3, color="tab:blue", lw=1.2, label="χ″ part (∝ P)")
        ax.plot(f, a["disperse"] * 1e3, color="tab:green", lw=1.2, label="χ′ part (mixing)")
        ax.plot(f, a["fitsub_online"] * 1e3, color="0.5", lw=0.9, ls="--", label="online fitsub")
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_title(f"{r['time'][:10]} · {period_label(r['cc'])} · P_online {r['P_online']:+.2f}", fontsize=8.5)
        ax.set_xlabel("MHz", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.3)
    axes[0, 0].set_ylabel("mV")
    axes[0, 0].legend(fontsize=7)
    fig.tight_layout()
    save(fig, "3_signal_decomposition_gallery.png")


def fig_closure_and_leg(fits, col):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    ax = axes[0]
    lim = 0.75
    ax.plot([-lim, lim], [-lim, lim], color="0.4", lw=0.8, ls="--")
    for r in fits:
        ax.plot(r["P_online"], r["P_regen_online_chain"], "o", color=col[r["cc"]], ms=6,
                mec="k" if r["period_note"] else col[r["cc"]])
    d = np.array([(r["P_regen_online_chain"] - r["P_online"]) / abs(r["P_online"]) for r in fits if not r["period_note"]]) * 100
    ax.set_xlabel("online polarization (data)")
    ax.set_ylabel("online chain applied to regenerated sweep")
    ax.set_title(f"Closure test: online chain on regenerated sweeps vs online P\n"
                 f"good periods: median |Δ| = {np.median(np.abs(d)):.1f} %, max {np.max(np.abs(d)):.1f} %  (black-edged: cc −14.5)")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.grid(alpha=0.3)
    ax = axes[1]
    for r in fits:
        ax.plot(r["P_area"], r["leg_data_mV"], "o", color=col[r["cc"]], ms=6,
                mec="k" if r["period_note"] else col[r["cc"]])
    ax.axhline(0, color="0.5", lw=0.7)
    ax.axvline(0, color="0.5", lw=0.7)
    ax.set_xlabel("polarization (area method, signed)")
    ax.set_ylabel("right-wing 'leg' in raw − fitted bg [mV]")
    ax.set_title("χ′ test: the right-wing 'leg' flips sign with P → dispersion, not background")
    ax.grid(alpha=0.3)
    for cc, c in col.items():
        ax.plot([], [], "o", color=c, label=period_label(cc))
    ax.legend(fontsize=7)
    fig.tight_layout()
    save(fig, "4_closure_and_dispersion_test.png")


def fig_params(fits, col):
    names = [("eta_q", "η"), ("g", "g"), ("wq", "wq [MHz]"), ("f0", "f0 [MHz]")]
    fig, axes = plt.subplots(1, 4, figsize=(17, 3.8))
    ccs = sorted(col)
    for ax, (key, lab) in zip(axes, names):
        for k, cc in enumerate(ccs):
            v = [r["signal"][key] for r in fits if r["cc"] == cc]
            ax.plot([k] * len(v), v, "o", color=col[cc], ms=6)
        ax.set_xticks(range(len(ccs)), [period_label(c) for c in ccs], rotation=40, fontsize=7)
        ax.set_title(lab)
        ax.grid(alpha=0.3)
    fig.suptitle("Fitted lineshape parameters per period (inputs for the Monte Carlo)", fontsize=11)
    fig.tight_layout()
    save(fig, "5_lineshape_parameters.png")


def tables(fits):
    lines = ["| period cc | n (+/−) | median mismatch area method | max \\|mismatch\\| | median lineshape/online | fit rms in signal region |",
             "|---|---|---|---|---|---|"]
    by = defaultdict(list)
    for r in fits:
        by[r["cc"]].append(r)
    for cc, rs in sorted(by.items()):
        mm = np.array([r["mismatch_rel"] for r in rs]) * 100
        ls = np.array([r["P_lineshape"] / r["P_online"] for r in rs])
        npos = sum(r["P_online"] > 0 for r in rs)
        lines.append(f"| {cc:g}{' ⚠' if rs[0]['period_note'] else ''} | {len(rs)} ({npos}/{len(rs) - npos}) | {np.median(mm):+.1f} % | "
                     f"{np.max(np.abs(mm)):.1f} % | {np.median(ls):.2f} | {np.median([r['fit_rms_signal_region_V'] for r in rs]) * 1e6:.0f} µV |")
    ev = ["| time | cc | P online | P area (fit) | mismatch | P lineshape | closure P (regen) |", "|---|---|---|---|---|---|---|"]
    for r in fits:
        ev.append(f"| {r['time'][:16]} | {r['cc']:g} | {r['P_online']:+.3f} | {r['P_area']:+.3f} ± {r['P_area_err']:.3f} | "
                  f"{r['mismatch_rel'] * 100:+.1f} % | {r['P_lineshape']:+.3f} | {r['P_regen_online_chain']:+.3f} |")
    text = "## Per period\n\n" + "\n".join(lines) + "\n\n## Per event\n\n" + "\n".join(ev) + "\n"
    (PLOTS / "tables.md").write_text(text)
    print(text)


def main():
    fits = load()
    col = colors_for(fits)
    fig_polarization(fits, col)
    fig_regeneration(fits, col)
    fig_signal_split(fits, col)
    fig_closure_and_leg(fits, col)
    fig_params(fits, col)
    tables(fits)


if __name__ == "__main__":
    main()
