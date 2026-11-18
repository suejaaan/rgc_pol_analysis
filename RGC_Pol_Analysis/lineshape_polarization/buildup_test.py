"""Does the lineshape P follow the true P below ~0.3? Test along polarization build-ups in consistent periods.

Inside one build-up the cc, channel and tune are fixed, so the online P (cc x area) is proportional to the true P.
About 10 events spread evenly in online P are fitted per build-up (same fit as run_lineshape.py). If the lineshape
works, P_lineshape follows P_online from ~0 upward; if it has a floor, P_lineshape stays near ~0.3.

Usage: python buildup_test.py     (fits cached in results/buildup/)
"""
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import confirm_shapes as CS
import run_lineshape as RL
from run_lineshape import FS, M

RES, PLOTS = CS.RES, CS.PLOTS
OUT = RES / "buildup"
N_PER_BUILDUP = 10
BUILDUPS = [  # (label, cc, start, end) — online P goes from ~0 to its plateau with constant cc
    ("cc −1.34, Sep 7–8 2022 (P > 0)", -1.34, "2022-09-07T17:00", "2022-09-08T05:00"),
    ("cc −1.41, Sep 11–13 2022 (P < 0)", -1.41, "2022-09-11T19:00", "2022-09-14T00:00"),
    ("cc −3.65, Feb 24–25 2023 (P < 0)", -3.65, "2023-02-24T12:00", "2023-02-25T12:00"),
    ("cc −3.65, Mar 1 2023 (P > 0)", -3.65, "2023-03-01T00:00", "2023-03-02T02:00"),
]
COLORS = ["tab:blue", "tab:orange", "tab:green", "tab:purple"]


def init_worker():
    RL.init_worker()
    FS.OUT = OUT


def horn_ratio(f, f0, y):
    """Peak height of the upper-frequency horn over the lower one (signal sign removed)."""
    y = y * np.sign(np.sum(y))
    x = f - f0
    return float(y[(x > 0.05) & (x < 0.10)].max() / y[(x > -0.10) & (x < -0.05)].max())


def select(pool):
    picks = []
    for k, (label, cc, t0, t1) in enumerate(BUILDUPS):
        evs = [e for e in pool if abs(e["cc"] - cc) < 1e-6 and t0 <= e["time"].replace(" ", "T") <= t1]
        sign = np.sign(np.median([e["pol"] for e in evs]))
        evs = [e for e in evs if np.sign(e["pol"]) == sign]
        targets = np.linspace(min(abs(e["pol"]) for e in evs), max(abs(e["pol"]) for e in evs), N_PER_BUILDUP)
        chosen = []
        for t in targets:
            e = min((e for e in evs if e not in chosen), key=lambda e: abs(abs(e["pol"]) - t))
            chosen.append(e)
        picks += [dict(e, buildup=k) for e in chosen]
        print(f"{label}: {len(evs)} events, picked |P| " + " ".join(f"{abs(e['pol']):.2f}" for e in chosen), flush=True)
    return picks


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    cache_file = OUT / "buildup_fits.yaml"
    cache = {(r["time"], r["buildup"]): r for r in yaml.safe_load(cache_file.read_text())} if cache_file.exists() else {}
    print("reading consistent-period events ...", flush=True)
    pool = CS.consistent_pool({cc for _, cc, _, _ in BUILDUPS})
    picks = select(pool)
    todo = [e for e in picks if (e["time"], e["buildup"]) not in cache]
    if todo:
        base = yaml.safe_load(FS.BASE_FITS.read_text())["baselines"]
        jobs = []
        for e in todo:
            b = base.get(e["baseline_id"])
            jobs.append((e, [b["params"][n] for n in M.CIRCUIT_NAMES] if b else None))
        print(f"fitting {len(jobs)} events on {RL.N_WORKERS} cores ...", flush=True)
        with Pool(RL.N_WORKERS, initializer=init_worker) as p:
            for r in p.imap_unordered(FS.fit_one, jobs):
                cache[(r["time"], r["buildup"])] = r
                print(f"  {r['time']}  P_online {r['P_online']:+.3f}  P_ls {r['P_lineshape']:+.3f} ± "
                      f"{r['P_lineshape_err']:.3f}  rms {r['fit_rms_V'] * 1e6:.0f} µV", flush=True)
        cache_file.write_text(yaml.safe_dump(sorted(cache.values(), key=lambda r: r["time"]), sort_keys=False))
    fits = [cache[(e["time"], e["buildup"])] for e in picks]

    # horn ratio of the data (chi''-driven part) and the MC expectation through each build-up's circuit
    for r in fits:
        z = np.load(CS.npz_path(r, OUT))
        r["horn_data"] = horn_ratio(z["f"], r["signal"]["f0"], z["raw"] - z["background"] - z["disperse"])

    fig, axes = plt.subplots(1, 3, figsize=(21, 6.4))
    ax, axh, axs = axes
    pg = np.linspace(-0.6, 0.6, 61)
    for k, (label, *_ ) in enumerate(BUILDUPS):
        rs = [r for r in fits if r["buildup"] == k]
        po = np.array([r["P_online"] for r in rs])
        ax.errorbar(po, [r["P_lineshape"] for r in rs], yerr=[r["P_lineshape_err"] for r in rs], fmt="o",
                    color=COLORS[k], ms=6, capsize=2, label=label)
        axh.plot(po, [r["horn_data"] for r in rs], "o", color=COLORS[k], ms=6, label=f"data: {label}")
        ref = max(rs, key=lambda r: abs(r["area_online"]))
        f = np.load(CS.npz_path(ref, OUT))["f"]
        c = [ref["circuit"][n] for n in M.CIRCUIT_NAMES]
        sg = np.sign(po[0])
        pp = pg[np.sign(pg) == sg] if sg != 0 else pg
        hr = []
        for P in pp:
            s = dict(ref["signal"], P=float(P))
            hr.append(horn_ratio(f, s["f0"], M.absorption_dispersion_parts(f, c, [s[n] for n in M.SIGNAL_NAMES])[0]))
        axh.plot(pp, hr, "-", color=COLORS[k], lw=1.4, alpha=0.8)
    lim = 0.62
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.9, label="P_lineshape = P_online")
    ax.axhspan(-0.3, 0.3, color="0.93", zorder=0)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("online P (cc trusted in these periods)")
    ax.set_ylabel("lineshape P (no cc)")
    ax.set_title("Lineshape P along polarization build-ups", fontsize=12, weight="bold")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5, loc="upper left")
    axh.set_xlabel("online P")
    axh.set_ylabel("horn ratio (upper / lower frequency horn), data")
    axh.set_title("Horn ratio of the data vs MC expectation (lines)", fontsize=12, weight="bold")
    axh.grid(alpha=0.3)
    axh.legend(fontsize=8, loc="upper left")

    # raw (unnormalized) signals along one build-up: same cc and tune, so size and shape should both follow P
    k0 = 0
    rs = sorted([r for r in fits if r["buildup"] == k0], key=lambda r: abs(r["P_online"]))
    cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rs)))
    for r, col in zip(rs, cmap):
        z = np.load(CS.npz_path(r, OUT))
        x = (z["f"] - r["signal"]["f0"]) * 1e3
        axs.plot(x, (z["raw"] - z["background"] - z["disperse"]) * 1e3, "-", color=col, lw=1,
                 label=f"P_online {r['P_online']:+.2f} · P_ls {r['P_lineshape']:+.2f}")
    axs.set_xlim(-250, 250)
    axs.set_xlabel("f − f0 [kHz]")
    axs.set_ylabel("χ″ signal, data [mV] (not normalized)")
    axs.set_title(f"Data along build-up: {BUILDUPS[k0][0]}", fontsize=12, weight="bold")
    axs.grid(alpha=0.3)
    axs.legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Build-up test: does the lineshape follow the polarization below 0.3?", fontsize=14, weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "buildup_test.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {PLOTS / 'buildup_test.png'}")
    for r in fits:
        print(f"  b{r['buildup']} {r['time']}  P_on {r['P_online']:+.3f}  P_ls {r['P_lineshape']:+.3f} ± "
              f"{r['P_lineshape_err']:.3f}  horn {r['horn_data']:.2f}  rms {r['fit_rms_V'] * 1e6:.0f}")


if __name__ == "__main__":
    sys.exit(main())
