"""Stage 1: plateau selection (no cc used for any decision).

Inside each tune segment (stage 0) an event is "on a plateau" when, over the preceding WINDOW_H hours:
  * the online area changes by less than MAX_RATE per hour (linear-fit slope / area),
  * the area scatter is below MAX_SCATTER (relative std after removing the slope; catches tune jumps),
  * the microwave frequency moved by less than MAX_DUW GHz (no spin reversal / re-optimisation),
  * at least MIN_WINDOW_EVENTS events were taken, and the window does not cross a sign change,
  * at least SETTLE_H hours have passed since the segment start, a sign change or a microwave jump.
Large signal: |area| >= BIG_FRACTION x the largest |area| of that sign in the segment (relative, no cc).

Consecutive plateau events (gap < MERGE_GAP_H) form a plateau; plateaus shorter than MIN_PLATEAU_H are dropped.
Per plateau up to N_PER_PLATEAU events, spaced >= SPACING_H, are picked (at most N_PER_SEGMENT per segment).
The |P| > 0.3 cut is applied later, on the lineshape P from the stage-2 fits; online P is kept for reference.

Outputs: results/plateaus.csv, results/plateau_events.csv, plots/stage1_plateaus.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES, PLOTS = HERE / "results", HERE / "plots"

WINDOW_H = 1.0
MAX_RATE = 0.03          # |d area / dt| / |area| per hour
MAX_SCATTER = 0.02       # relative rms about the linear trend in the window
MAX_DUW = 0.05           # GHz
MIN_WINDOW_EVENTS = 8
SETTLE_H = 2.0
BIG_FRACTION = 0.5
MERGE_GAP_H = 0.5
MIN_PLATEAU_H = 1.0
N_PER_PLATEAU = 10
N_PER_SEGMENT = 50
SPACING_H = 0.5
MIN_SEGMENT_EVENTS = 100


def flag_segment(d):
    """d: candidate events of one segment, sorted by time. Adds plateau diagnostics."""
    t = (d["t"] - d["t"].iloc[0]).dt.total_seconds().to_numpy() / 3600
    a = d["area_online"].to_numpy()
    uw = d["uwave_freq"].fillna(0).to_numpy()
    sign = np.sign(a)
    # regime starts: segment start, sign change, microwave jump (> MAX_DUW between consecutive events)
    start = np.zeros(len(d), bool)
    start[0] = True
    start[1:] |= sign[1:] != sign[:-1]
    start[1:] |= np.abs(np.diff(uw)) > MAX_DUW
    t_regime = np.maximum.accumulate(np.where(start, t, -np.inf))
    rate = np.full(len(d), np.nan)
    scat = np.full(len(d), np.nan)
    nwin = np.zeros(len(d), int)
    ok = np.zeros(len(d), bool)
    j0 = 0
    for i in range(len(d)):
        while t[j0] < t[i] - WINDOW_H:
            j0 += 1
        w = slice(j0, i + 1)
        nwin[i] = i + 1 - j0
        if nwin[i] < MIN_WINDOW_EVENTS or np.any(sign[w] != sign[i]):
            continue
        k, c = np.polyfit(t[w] - t[i], a[w], 1)
        rate[i] = abs(k) / abs(c)
        scat[i] = np.std(a[w] - (k * (t[w] - t[i]) + c)) / abs(c)
        ok[i] = (rate[i] < MAX_RATE and scat[i] < MAX_SCATTER and np.ptp(uw[w]) < MAX_DUW
                 and t[i] - t_regime[i] >= SETTLE_H)
    amax = {s: np.abs(a[sign == s]).max() for s in (-1, 1) if np.any(sign == s)}
    big = np.array([abs(x) >= BIG_FRACTION * amax[np.sign(x)] if x != 0 else False for x in a])
    d = d.assign(rate_per_h=rate, rel_scatter=scat, n_window=nwin, settled=t - t_regime >= SETTLE_H,
                 big=big, plateau_ok=ok & big)
    return d


def main():
    ev = pd.read_csv(RES / "events.csv")
    ev = ev[ev["candidate"]].copy()
    ev["t"] = pd.to_datetime(ev["time"])
    sizes = ev.groupby("segment").size()
    ev = ev[ev["segment"].isin(sizes[sizes >= MIN_SEGMENT_EVENTS].index)]
    flagged = pd.concat([flag_segment(g.sort_values("t")) for _, g in ev.groupby("segment")])

    plateaus, picks, pid = [], [], 0
    for seg, g in flagged.groupby("segment"):
        g = g[g["plateau_ok"]]
        if g.empty:
            continue
        gap = g["t"].diff().dt.total_seconds().fillna(0).to_numpy() / 3600
        brk = (gap > MERGE_GAP_H) | (np.sign(g["area_online"]).diff().fillna(0).to_numpy() != 0)
        seg_picks = []
        for _, p in g.groupby(np.cumsum(brk)):
            dur = (p["t"].iloc[-1] - p["t"].iloc[0]).total_seconds() / 3600
            if dur < MIN_PLATEAU_H:
                continue
            plateaus.append({
                "plateau": pid, "segment": seg, "start": p["time"].iloc[0], "end": p["time"].iloc[-1],
                "hours": round(dur, 2), "n_events": len(p), "sign": int(np.sign(p["area_online"].iloc[0])),
                "area_median": p["area_online"].median(), "area_rel_spread": p["area_online"].std() / abs(p["area_online"].median()),
                "uwave_freq": p["uwave_freq"].median(), "beam_on_frac": float((p["beam_current"].fillna(0) > 1).mean()),
                "online_cc": ";".join(f"{c:g}" for c in sorted(p["cc"].unique())),
                "pol_online_median": p["pol_online"].median(), "period_verdict": ";".join(sorted(p["period_verdict"].unique())),
            })
            # evenly spaced picks across the plateau
            chosen = []
            for _, r in p.iloc[np.linspace(0, len(p) - 1, min(len(p), 4 * N_PER_PLATEAU)).astype(int)].iterrows():
                if all(abs((r["t"] - c["t"]).total_seconds()) / 3600 >= SPACING_H for c in chosen):
                    chosen.append(r)
            if len(chosen) > N_PER_PLATEAU:
                chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, N_PER_PLATEAU).astype(int)]
            seg_picks += [dict(r, plateau=pid) for r in chosen]
            pid += 1
        if len(seg_picks) > N_PER_SEGMENT:
            seg_picks = [seg_picks[i] for i in np.linspace(0, len(seg_picks) - 1, N_PER_SEGMENT).astype(int)]
        picks += seg_picks

    pl = pd.DataFrame(plateaus)
    pk = pd.DataFrame(picks)
    keep = ["time", "file", "index", "segment", "plateau", "channel", "baseline_id", "area_online", "sign",
            "rate_per_h", "rel_scatter", "uwave_freq", "beam_current", "cc", "pol_online", "period_verdict"]
    pl.to_csv(RES / "plateaus.csv", index=False)
    pk[keep].to_csv(RES / "plateau_events.csv", index=False)

    n_ok = int(flagged["plateau_ok"].sum())
    print(f"segments analysed {flagged['segment'].nunique()}, candidate events {len(flagged)}")
    print(f"events on a plateau with a large signal: {n_ok}  (rate ok {int((flagged['rate_per_h'] < MAX_RATE).sum())}, "
          f"settled {int(flagged['settled'].sum())}, big {int(flagged['big'].sum())})")
    print(f"plateaus {len(pl)} ({pl['hours'].sum():.0f} h) in {pl['segment'].nunique()} segments; "
          f"+ {int((pl['sign'] > 0).sum())} / - {int((pl['sign'] < 0).sum())}")
    print(f"picked {len(pk)} events: + {int((pk['sign'] > 0).sum())} / - {int((pk['sign'] < 0).sum())}")
    print("picked by period verdict:", pk["period_verdict"].value_counts().to_dict())
    print("online |P| of picks (reference only): quartiles",
          np.round(np.percentile(np.abs(pk["pol_online"]), [0, 25, 50, 75, 100]), 2).tolist())
    print(pl[["plateau", "segment", "start", "hours", "n_events", "sign", "area_median", "area_rel_spread",
              "online_cc", "pol_online_median", "period_verdict"]].round(3).to_string(index=False))

    # ---- plot: one panel per segment that has a plateau ----
    segs = sorted(pl["segment"].unique())
    ncol = 5
    nrow = int(np.ceil(len(segs) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 2.9 * nrow), squeeze=False)
    for ax, seg in zip(axes.flat, segs):
        g = flagged[flagged["segment"] == seg]
        ax.plot(g["t"], g["area_online"], ".", ms=1.5, color="0.6")
        for _, p in pl[pl["segment"] == seg].iterrows():
            ax.axvspan(pd.Timestamp(p["start"]), pd.Timestamp(p["end"]), color="tab:green", alpha=0.2, lw=0)
        s = pk[pk["segment"] == seg]
        ax.plot(s["t"], s["area_online"], "o", ms=4, color="tab:red", mfc="none")
        v = s["period_verdict"].mode().iloc[0]  # verdict of the picked events (segments can span cc periods)
        ax.set_title(f"seg {seg} · {g['time'].iloc[0][:10]} · {v}", fontsize=8.5,
                     color="tab:orange" if v == "online wrong" else "black")
        ax.axhline(0, color="0.8", lw=0.5)
        ax.tick_params(labelsize=6.5)
        ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
        ax.grid(alpha=0.3)
    for ax in axes.flat[len(segs):]:
        ax.set_axis_off()
    fig.suptitle(f"Stage 1: {len(pl)} plateaus (green) and {len(pk)} picked events (red) — online area vs time per tune segment",
                 fontsize=12, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(PLOTS / "stage1_plateaus.png", dpi=140)
    print(f"wrote {RES / 'plateaus.csv'}, {RES / 'plateau_events.csv'}, {PLOTS / 'stage1_plateaus.png'}")


if __name__ == "__main__":
    main()
