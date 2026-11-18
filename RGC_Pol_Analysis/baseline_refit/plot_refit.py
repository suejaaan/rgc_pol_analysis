"""Plots for refit_baselines.py: example fits, old-vs-new comparison, parameter time series, quality."""
import csv
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
sys.path.insert(0, str(REPO / "data_creation"))
import model as M  # noqa: E402
from physics.baseline import Baseline  # noqa: E402

CACHE = HERE / "cache"
FITS = HERE / "results" / "baseline_fits_physical.yaml"
OLD_FITS = REPO / "data_creation/fitting/baseline_fits_data_d.yaml"
PLOT_DIR = HERE / "plots"
MUST_BE_POSITIVE = ["U", "Cknob", "L0", "Rcoil", "R", "R1", "r", "alpha", "beta1", "Z_cable", "D", "M", "Cstray"]
OLD_PARAM_ORDER = ["U", "Cknob", "eta", "trim", "Cstray", "phi_const", "DC_offset", "L0", "Rcoil", "R", "R1", "r",
                   "alpha", "beta1", "Z_cable", "D", "M", "delta_C", "delta_phi", "delta_phase", "delta_l"]
TEST_BASE_FILE = "current_2022-09-20_19-55-15.txt"


def to_dt(s):
    return datetime.fromisoformat(s[:19])


def load():
    fits = yaml.safe_load(FITS.read_text())["baselines"]
    z = np.load(CACHE / "baselines.npz")
    curves = {}
    for bid in fits:
        b, f = z["baseline"][bid], z["freq"][bid]
        ok = ~np.isnan(b)
        curves[bid] = (f[ok], b[ok])
    return fits, curves


def save(fig, name):
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    path = PLOT_DIR / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def pick_examples(fits):
    by_events = max(fits, key=lambda i: fits[i]["n_events"])
    test = next(i for i in fits if (fits[i]["base_file"] or "").endswith(TEST_BASE_FILE))
    worst = max(fits, key=lambda i: fits[i]["nrmse"] or 0)
    wide = next((i for i in fits if fits[i]["channel"].endswith("800kHz")), None)
    rf = [max((i for i in fits if fits[i]["channel"] == ch), key=lambda i: fits[i]["n_events"])
          for ch in ("Deuteron5T-400mV", "Deuteron5T-150mV") if any(fits[i]["channel"] == ch for i in fits)]
    early = min(fits, key=lambda i: fits[i]["first_event"])
    picks = []
    for i, why in [(by_events, "most-used baseline"), (test, "data-test run's baseline"), (worst, "worst nrmse"),
                   (wide, "wide sweep"), *[(i, "RF-level channel") for i in rf], (early, "earliest")]:
        if i is not None and i not in [p for p, _ in picks]:
            picks.append((i, why))
    return picks[:8]


def plot_examples(fits, curves):
    picks = pick_examples(fits)
    ncol = 4
    nrow = int(np.ceil(len(picks) / ncol))
    fig = plt.figure(figsize=(5 * ncol, 5.2 * nrow))
    gs = fig.add_gridspec(nrow * 2, ncol, height_ratios=[3, 1] * nrow, hspace=0.08)
    for k, (bid, why) in enumerate(picks):
        r, c = divmod(k, ncol)
        v = fits[bid]
        f, y = curves[bid]
        model = M.qmeter(f, *[v["params"][n] for n in M.CIRCUIT_NAMES])
        ax = fig.add_subplot(gs[2 * r, c])
        ax.plot(f, y, color="k", lw=1.6, label="baseline sweep")
        ax.plot(f, model, color="tab:red", lw=1.0, ls="--", label="physical Q-meter fit")
        ax.set_title(f"#{bid} ({why})\n{v['channel']} · {v['base_time'][:16] if v['base_time'] else '?'} · "
                     f"{v['n_events']} events", fontsize=9)
        ax.tick_params(labelbottom=False)
        ax.grid(alpha=0.3)
        if k == 0:
            ax.set_ylabel("V")
            ax.legend(fontsize=8)
        axr = fig.add_subplot(gs[2 * r + 1, c], sharex=ax)
        axr.plot(f, (y - model) * 1e6, color="k", lw=0.7)
        axr.axhline(0, color="0.5", lw=0.7)
        axr.set_title(f"rms {v['rms_V'] * 1e6:.1f} µV  (ptp {v['ptp_V'] * 1e3:.0f} mV)", fontsize=8, y=0.72)
        axr.set_xlabel("frequency [MHz]")
        axr.grid(alpha=0.3)
        if c == 0:
            axr.set_ylabel("resid [µV]")
    fig.suptitle("Physical Q-meter fits to data-d baseline sweeps (examples)", fontsize=13, y=0.995)
    save(fig, "1_example_fits.png")


def old_fits_by_baseline():
    """Map each old (file, event) fit to a baseline id via the event map (current_* -> its run twin)."""
    old = yaml.safe_load(OLD_FITS.read_text())
    emap = {}
    with open(CACHE / "event_baseline_map.csv") as fh:
        for row in csv.DictReader(fh):
            emap[(row["file"], int(row["event_index"]))] = int(row["baseline_id"])
    runs = {k[0].split("__")[0]: k[0] for k in emap if "__" in k[0]}
    out = {}
    for fname, info in old.items():
        src = runs.get(fname[len("current_"):-4], fname) if fname.startswith("current_") else fname
        for idx, ev in info["events"].items():
            bid = emap.get((src, int(idx)))
            if bid is not None and bid not in out:
                out[bid] = ev
    return out


def plot_old_vs_new(fits, curves):
    old = old_fits_by_baseline()
    common = sorted(set(old) & set(fits))
    new_rms = np.array([fits[i]["rms_V"] for i in common]) * 1e6
    old_rms = np.array([old[i]["rmse"] for i in common]) * 1e6
    old_neg = np.array([sum(old[i]["params"].get(p, 1) < 0 for p in MUST_BE_POSITIVE) for i in common])
    new_neg = np.array([sum(fits[i]["params"][p] < 0 for p in M.CIRCUIT_NAMES if p in MUST_BE_POSITIVE) for i in common])

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax = axes[0]
    lim = [min(new_rms.min(), old_rms.min()) * 0.7, max(new_rms.max(), old_rms.max()) * 1.4]
    ax.loglog(old_rms, new_rms, "o", ms=5, alpha=0.7)
    ax.plot(lim, lim, color="0.4", lw=0.8, ls="--", label="equal rms")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("old fit rms [µV]  (21 params, unbounded)")
    ax.set_ylabel("new fit rms [µV]  (9 params, physical bounds)")
    ax.set_title(f"Fit quality, same baseline ({len(common)} baselines)\n"
                 f"median old {np.median(old_rms):.1f} µV · new {np.median(new_rms):.1f} µV")
    ax.legend()
    ax.grid(alpha=0.3, which="both")

    ax = axes[1]
    bins = np.arange(0, max(old_neg.max(), 1) + 2) - 0.5
    ax.hist(old_neg, bins=bins, color="tab:gray", alpha=0.8, label="old fits")
    ax.hist(new_neg, bins=bins, color="tab:blue", alpha=0.8, label="new fits")
    ax.set_xlabel("# of must-be-positive parameters that are negative")
    ax.set_ylabel("baselines")
    ax.set_title("Physicality: L, R, C, cable impedance/loss must be > 0")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[2]
    bid = next(i for i in common if (fits[i]["base_file"] or "").endswith(TEST_BASE_FILE))
    f, y = curves[bid]
    new = M.qmeter(f, *[fits[bid]["params"][n] for n in M.CIRCUIT_NAMES])
    p = old[bid]["params"]
    old_curve = Baseline(f, *[p[n] for n in OLD_PARAM_ORDER[:2]], p["eta"], p["trim"], p["Cstray"], p["phi_const"],
                         p["DC_offset"], "deuteron", **{n: p[n] for n in OLD_PARAM_ORDER[7:]})
    ax.plot(f, (y - old_curve) * 1e6, color="tab:gray", lw=1.0, label=f"old: rms {old[bid]['rmse'] * 1e6:.1f} µV")
    ax.plot(f, (y - new) * 1e6, color="tab:blue", lw=1.0, label=f"new: rms {fits[bid]['rms_V'] * 1e6:.1f} µV")
    ax.axhline(0, color="0.5", lw=0.7)
    neg = [n for n in MUST_BE_POSITIVE if p.get(n, 1) < 0]
    ax.set_title(f"Residuals, data-test baseline #{bid}\nold fit has negative: {', '.join(neg)}", fontsize=10)
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("data − fit [µV]")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "2_old_vs_new.png")
    return len(common), np.median(old_rms), np.median(new_rms), int(np.sum(old_neg > 0)), int(np.sum(new_neg > 0))


def plot_param_timeseries(fits):
    ids = sorted(fits, key=lambda i: fits[i]["base_time"] or "")
    t = [to_dt(fits[i]["base_time"]) for i in ids]
    chans = sorted({fits[i]["channel"] for i in ids})
    colors = dict(zip(chans, plt.cm.tab10.colors))
    units = {"U": "V", "Cknob": "×20 pF", "trim": "rad", "phi_deg": "deg", "DC_offset": "V", "L0": "×10 nH",
             "Rcoil": "Ω", "alpha": "cable loss", "Cstray": "pF"}
    fig, axes = plt.subplots(3, 3, figsize=(18, 11), sharex=True)
    for ax, name in zip(axes.flat, M.CIRCUIT_NAMES):
        for ch in chans:
            sel = [k for k, i in enumerate(ids) if fits[i]["channel"] == ch]
            ax.errorbar([t[k] for k in sel], [fits[ids[k]]["params"][name] for k in sel],
                        yerr=[min(fits[ids[k]]["param_errors"][name], 1e3) for k in sel],
                        fmt="o", ms=4, color=colors[ch], alpha=0.8, label=ch, capsize=0, lw=0.8)
        ax.axhline(M.CIRCUIT_LO[M.CIRCUIT_NAMES.index(name)], color="crimson", lw=0.6, ls=":")
        ax.axhline(M.CIRCUIT_HI[M.CIRCUIT_NAMES.index(name)], color="crimson", lw=0.6, ls=":")
        lo = min(fits[i]["params"][name] for i in ids)
        hi = max(fits[i]["params"][name] for i in ids)
        pad = 0.15 * (hi - lo) if hi > lo else 1
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_title(f"{name} [{units[name]}]")
        ax.grid(alpha=0.3)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes.flat[0].legend(fontsize=7, loc="best")
    for ax in axes[-1]:
        ax.tick_params(axis="x", rotation=30)
    fig.suptitle("Fitted Q-meter parameters per baseline vs time (dotted red = fit bounds)", fontsize=13)
    fig.tight_layout()
    save(fig, "3_parameters_vs_time.png")


def plot_quality(fits):
    ids = sorted(fits, key=lambda i: fits[i]["base_time"] or "")
    t = [to_dt(fits[i]["base_time"]) for i in ids]
    rms = np.array([fits[i]["rms_V"] for i in ids]) * 1e6
    nr = np.array([fits[i]["nrmse"] for i in ids])
    nev = np.array([fits[i]["n_events"] for i in ids])
    fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
    sc = axes[0].scatter(t, rms, c=np.log10(nev), cmap="viridis", s=25)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("fit rms [µV]")
    axes[0].set_title("Fit rms per baseline (colour: log10 # events using it)")
    plt.colorbar(sc, ax=axes[0])
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    axes[0].tick_params(axis="x", rotation=30)
    axes[1].hist(rms, bins=np.logspace(np.log10(rms.min() * 0.8), np.log10(rms.max() * 1.2), 30), color="tab:blue")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("fit rms [µV]")
    axes[1].set_title(f"rms distribution: median {np.median(rms):.1f} µV")
    axes[2].hist(nr * 1e3, bins=30, color="tab:green")
    axes[2].set_xlabel("rms / peak-to-peak  [×10⁻³]")
    axes[2].set_title(f"normalized rms: median {np.median(nr) * 1e3:.2f} ×10⁻³")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "4_fit_quality.png")


def main():
    fits, curves = load()
    plot_examples(fits, curves)
    n, old_med, new_med, n_old_bad, n_new_bad = plot_old_vs_new(fits, curves)
    plot_param_timeseries(fits)
    plot_quality(fits)
    print(f"\nold vs new on {n} common baselines: median rms {old_med:.1f} -> {new_med:.1f} µV; "
          f"fits with a negative must-be-positive param: {n_old_bad} -> {n_new_bad}")


if __name__ == "__main__":
    main()
