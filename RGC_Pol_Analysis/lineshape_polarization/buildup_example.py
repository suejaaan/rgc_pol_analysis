"""Example plot: during a fast build-up the signal area grows ~10x while the lineshape P stays flat.

Uses the cached fits of buildup_test.py (results/buildup/). Each build-up is inside one online-cc period
(same cc, channel and tune). Top: P vs time from the online area, from our raw-sweep fit area (same cc) and
from the lineshape. Bottom: the measured chi'' signals (not normalized) at those times.
"""
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

import buildup_test as BT
import confirm_shapes as CS

CASES = [3, 0]  # Mar 1 2023 (cc -3.65), Sep 7 2022 (cc -1.34)


def main():
    fits = yaml.safe_load((BT.OUT / "buildup_fits.yaml").read_text())
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), gridspec_kw={"height_ratios": [1, 1.15]})
    for col, k in enumerate(CASES):
        label, cc, _, _ = BT.BUILDUPS[k]
        rs = sorted([r for r in fits if r["buildup"] == k and abs(r["P_online"]) > 0.01], key=lambda r: r["time"])
        t = [datetime.fromisoformat(r["time"]) for r in rs]
        ax = axes[0, col]
        ax.plot(t, [r["P_online"] for r in rs], "s-", color="0.5", ms=6, label=f"online P = cc × online area  (cc {cc:g})")
        ax.plot(t, [r["P_area"] for r in rs], "o-", color="tab:green", ms=6, label="P from our raw-sweep fit area (same cc)")
        ax.errorbar(t, [r["P_lineshape"] for r in rs], yerr=[r["P_lineshape_err"] for r in rs], fmt="D-",
                    color="tab:red", ms=6, capsize=2, label="P from lineshape (horn ratio, no cc)")
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_ylabel("polarization")
        ax.set_title(label, fontsize=12, weight="bold")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9, loc="lower right" if rs[-1]["P_online"] > 0 else "upper right")
        a0, a1 = abs(rs[0]["area_online"]), abs(rs[-1]["area_online"])
        ax.text(0.02, 0.96, f"area grows ×{a1 / a0:.0f}; lineshape P {rs[0]['P_lineshape']:+.2f} → {rs[-1]['P_lineshape']:+.2f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        ax.tick_params(axis="x", rotation=20)

        axs = axes[1, col]
        cmap = plt.cm.viridis(np.linspace(0, 0.9, len(rs)))
        for r, c in zip(rs, cmap):
            z = np.load(CS.npz_path(r, BT.OUT))
            x = (z["f"] - r["signal"]["f0"]) * 1e3
            y = (z["raw"] - z["background"] - z["disperse"]) * 1e3
            axs.plot(x, y, "-", color=c, lw=1.1,
                     label=f"{r['time'][11:16]}  P_area {r['P_area']:+.2f} · P_ls {r['P_lineshape']:+.2f}")
        axs.axhline(0, color="0.6", lw=0.6)
        axs.set_xlim(-250, 250)
        axs.set_xlabel("f − f0  [kHz]")
        axs.set_ylabel("measured χ″ signal  [mV]  (not normalized)")
        axs.set_title("Signals along the build-up: size grows, horn asymmetry already large at the start", fontsize=10.5)
        axs.grid(alpha=0.3)
        axs.legend(fontsize=7.8, loc="lower left" if rs[-1]["P_online"] > 0 else "upper left", ncol=1)
    fig.suptitle("Fast DNP build-ups at fixed cc: the signal area (∝ vector P) grows, the horn ratio (lineshape P) stays flat",
                 fontsize=13.5, weight="bold")
    fig.text(0.01, 0.003, "Each column: one online-cc period and channel. Mar 1: same stored baseline sweep 16:25–18:35 (clean case). "
             "Sep 7: baseline sweep and Q-meter phase change at ~20:00, first part fits worse (rms 70–110 µV). "
             "Measured χ″ = raw − fitted Q-meter background − fitted χ′ part. The area needs no spin-temperature assumption; "
             "the horn ratio does.", fontsize=8.5, color="0.3")
    fig.tight_layout(rect=(0, 0.015, 1, 0.97))
    out = CS.PLOTS / "buildup_example.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
