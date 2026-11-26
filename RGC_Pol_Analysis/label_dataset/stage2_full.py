"""Stage 2 (full): cc-free lineshape labels for all plateau events, with one shared circuit per tune segment.

Phase 1  segment-joint fit (segment_fit.py) on N_JOINT_EV events per segment -> shared circuit + sigma,
         best three circuit starts kept (their P spread = circuit systematic of the segment)
Phase 2  every picked event fitted with the best shared circuit and sigma fixed
         (free: U, phi, DC offset, A, P, f0, g; eta and wq fixed); start = nearest joint event,
         stuck events (signal rms > RESCUE x segment median) restarted from the other neighbour
Gate     fit_ok:   signal-region rms <= RESCUE x segment median, and no own parameter at a real bound
                   (U, phi, DC offset, f0; g at its lower bound is only flagged)
         label_ok: fit_ok and |P| >= P_MIN and segment P/area spread <= MAX_SEG_SPREAD
No online cc enters any fit or decision; online P and cc are stored only as reference columns.

Outputs: results/stage2/segments.yaml, results/stage2/labels.csv, results/stage2/npz_full/,
         plots/stage2_labels.png
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import model as M  # noqa: E402
import model_variants as V  # noqa: E402
import segment_fit as SF  # noqa: E402

OUT = HERE / "results" / "stage2"
NPZ = OUT / "npz_full"
N_JOINT_EV = 10
RESCUE = 2.0
P_MIN = 0.3
MAX_SEG_SPREAD = 0.05
N_WORKERS = SF.N_WORKERS


def parts(f, sh, own):
    """background, chi''-driven and chi'-driven parts of the model."""
    c = np.empty(9)
    c[SF.SHARED_C] = sh[:6]
    c[SF.OWN_C] = own[:3]
    A, P, f0, g = own[3:]
    chi2, chi1 = V.chi_gauss(f, A, P, f0, SF.WQ, SF.ETA, g, sh[6])
    bg = M.qmeter(f, *c)
    return bg, M.qmeter(f, *c, chi=-1j * chi2) - bg, M.qmeter(f, *c, chi=chi1 + 0j) - bg


def fit_event(job):
    ev, sh, starts = job
    t0 = time.time()
    fs, raws = SF.load([ev])
    f, raw = fs[0], raws[0]
    best = None
    for o0 in starts:
        o, cost = SF.fit_own(f, raw, sh, o0)
        if best is None or cost < best[1]:
            best = (o, cost)
    o = best[0]
    # error on P (shared part fixed) from the own-parameter Jacobian
    r = least_squares(lambda x: (SF.ev_model(f, sh, x) - raw) * SF.SCALE, o, bounds=(SF.OW_LO, SF.OW_HI),
                      x_scale="jac", max_nfev=1)
    J = r.jac / SF.SCALE
    res = raw - SF.ev_model(f, sh, o)
    cov = np.linalg.pinv(J.T @ J) * (np.sum(res**2) / max(len(f) - len(o), 1))
    bg, absorb, disperse = parts(f, sh, o)
    sreg = (f > V.SREG[0]) & (f < V.SREG[1])
    span = SF.OW_HI - SF.OW_LO
    names = ["U", "phi_deg", "DC_offset", "A", "P", "f0", "g"]
    at_bound = [n for k, n in enumerate(names) if np.isfinite(span[k]) and
                min(o[k] - SF.OW_LO[k], SF.OW_HI[k] - o[k]) < 1e-4 * max(abs(SF.OW_LO[k]), abs(SF.OW_HI[k]), 1e-3)]
    np.savez(NPZ / f"event_{ev['file'][:19]}_{int(ev['index'])}.npz", f=f, raw=raw, model=raw - res, background=bg,
             absorb=absorb, disperse=disperse, resid=res, own=o, shared=sh, cov=cov)
    return {
        "time": ev["time"], "file": ev["file"], "index": int(ev["index"]), "segment": int(ev["segment"]),
        "plateau": int(ev["plateau"]), "P": float(o[4]), "P_err_fit": float(np.sqrt(max(cov[4, 4], 0))),
        "A": float(o[3]), "U": float(o[0]), "phi_deg": float(o[1]), "DC_offset": float(o[2]), "f0": float(o[5]),
        "g": float(o[6]), "sigma_kHz": float(sh[6] * 1e3), "area_chi2": float(np.sum(absorb)),
        "area_online": float(ev["area_online"]),
        "rms_signal_uV": float(np.sqrt(np.mean(res[sreg] ** 2)) * 1e6),
        "rms_wings_uV": float(np.sqrt(np.mean(res[~sreg] ** 2)) * 1e6),
        "at_bound": ";".join(at_bound), "ref_online_cc": float(ev["cc"]), "ref_online_P": float(ev["pol_online"]),
        "ref_period_verdict": ev["period_verdict"], "fit_seconds": round(time.time() - t0, 1),
    }


def main():
    NPZ.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    picks = pd.read_csv(HERE / "results" / "plateau_events.csv")
    segs = sorted(picks["segment"].unique())
    SF.N_EV = N_JOINT_EV

    # ---- phase 1: shared circuit per segment ----
    seg_file = OUT / "segments.yaml"
    if seg_file.exists():
        seg_res = yaml.safe_load(seg_file.read_text())
        print(f"phase 1: reusing {seg_file}", flush=True)
    else:
        print(f"phase 1: segment-joint fits for {len(segs)} segments", flush=True)
        seg_res = SF.run_segments([int(s) for s in segs])
        seg_file.write_text(yaml.safe_dump(seg_res, sort_keys=False))
    print(f"phase 1 done at {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---- phase 2: every picked event with the shared part fixed ----
    jobs = []
    seg_info = {}
    for s in seg_res:
        best = s["solutions"][0]
        sh = np.array([best["shared"][M.CIRCUIT_NAMES[k]] for k in SF.SHARED_C] + [best["shared"]["sigma_kHz"] / 1e3])
        jev = best["events"]
        tj = np.array([np.datetime64(e["time"]) for e in jev])
        # circuit systematic: P spread over the kept solutions on the joint events
        Pm = np.array([[abs(e["P"]) for e in sol["events"]] for sol in s["solutions"]])
        ratio = np.array([e["P"] / e["area_online"] for e in jev])
        seg_info[s["segment"]] = {"P_circuit_syst": float(np.median(np.ptp(Pm, axis=0)) / 2) if len(Pm) > 1 else np.nan,
                                  "joint_ratio_spread": float(np.std(ratio) / abs(np.mean(ratio)))}
        for ev in picks[picks["segment"] == s["segment"]].to_dict("records"):
            order = np.argsort(np.abs(tj - np.datetime64(ev["time"])))[:2]
            starts = []
            for k in order:
                e = jev[k]
                starts.append(np.array([e["U"], e["phi_deg"], e["DC_offset"],
                                        e["A"] * ev["area_online"] / e["area_online"], e["P"],
                                        e.get("f0", 32.685), max(e["g"], 0.002)]))
            jobs.append((ev, sh, starts))
    print(f"phase 2: fitting {len(jobs)} events with the shared circuit fixed", flush=True)
    rows = []
    with Pool(N_WORKERS) as pool:
        for k, r in enumerate(pool.imap_unordered(fit_event, jobs), 1):
            rows.append(r)
            if k % 50 == 0 or k == len(jobs):
                print(f"  [{k}/{len(jobs)}] {(time.time() - t0) / 60:.1f} min", flush=True)
    d = pd.DataFrame(rows).sort_values("time")

    # ---- gate ----
    med = d.groupby("segment")["rms_signal_uV"].transform("median")
    hard = d["at_bound"].fillna("").apply(lambda s: any(n in s.split(";") for n in ("U", "phi_deg", "DC_offset", "f0")))
    d["g_at_lower"] = d["at_bound"].fillna("").str.contains("g")
    d["fit_ok"] = (d["rms_signal_uV"] <= RESCUE * med) & ~hard
    d["ratio"] = d["P"] / d["area_online"]
    spread = d[d["fit_ok"]].groupby("segment")["ratio"].agg(lambda x: np.std(x) / abs(np.mean(x)))
    d["seg_ratio_spread"] = d["segment"].map(spread)
    d["P_circuit_syst"] = d["segment"].map({k: v["P_circuit_syst"] for k, v in seg_info.items()})
    d["label_ok"] = d["fit_ok"] & (d["P"].abs() >= P_MIN) & (d["seg_ratio_spread"] <= MAX_SEG_SPREAD)
    d.to_csv(OUT / "labels.csv", index=False)

    print(f"\ndone in {(time.time() - t0) / 60:.1f} min: {len(d)} events, fit_ok {int(d['fit_ok'].sum())}, "
          f"label_ok {int(d['label_ok'].sum())} (+ {int((d['label_ok'] & (d['P'] > 0)).sum())} / "
          f"- {int((d['label_ok'] & (d['P'] < 0)).sum())})")
    print("label_ok by period verdict:", d[d["label_ok"]]["ref_period_verdict"].value_counts().to_dict())
    print(f"label |P| range {d[d['label_ok']]['P'].abs().min():.3f} .. {d[d['label_ok']]['P'].abs().max():.3f}")
    summ = d.groupby("segment").agg(n=("P", "size"), fit_ok=("fit_ok", "sum"), label_ok=("label_ok", "sum"),
                                    P_min=("P", lambda x: round(x.abs().min(), 3)), P_max=("P", lambda x: round(x.abs().max(), 3)),
                                    ratio=("ratio", "median"), spread=("seg_ratio_spread", "first"),
                                    syst=("P_circuit_syst", "first"), sigma=("sigma_kHz", "first"),
                                    rms_sig=("rms_signal_uV", "median"), online_cc=("ref_online_cc", lambda x: sorted(set(x))),
                                    verdict=("ref_period_verdict", "first"))
    print(summ.round(3).to_string())

    # ---- plot: P and P/area per segment ----
    segs = list(summ.index)
    ncol = 6
    nrow = int(np.ceil(len(segs) / ncol))
    fig, axes = plt.subplots(2 * nrow, ncol, figsize=(4 * ncol, 5.2 * nrow), squeeze=False)
    for i, s in enumerate(segs):
        g = d[d["segment"] == s]
        t = pd.to_datetime(g["time"])
        a1, a2 = axes[2 * (i // ncol), i % ncol], axes[2 * (i // ncol) + 1, i % ncol]
        for ok, c, lab in ((True, "tab:blue", "label ok"), (False, "tab:red", "rejected")):
            m = g["label_ok"] == ok
            a1.plot(t[m], g["P"][m].abs(), "o", ms=3, color=c, label=lab)
            a2.plot(t[m], g["ratio"][m], "o", ms=3, color=c)
        a1.axhline(P_MIN, color="0.6", lw=0.7, ls="--")
        a1.set_title(f"seg {s} · {g['time'].iloc[0][:10]} · {g['ref_period_verdict'].iloc[0]}", fontsize=8.5)
        a2.set_title(f"P/area spread {summ.loc[s, 'spread'] * 100:.1f} %, σ {summ.loc[s, 'sigma']:.1f} kHz", fontsize=8)
        for a in (a1, a2):
            a.tick_params(labelsize=6.5)
            a.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%m-%d"))
            a.grid(alpha=0.3)
        if i == 0:
            a1.legend(fontsize=7)
            a1.set_ylabel("|P| lineshape")
            a2.set_ylabel("P / online area")
    for i in range(len(segs), nrow * ncol):
        axes[2 * (i // ncol), i % ncol].set_axis_off()
        axes[2 * (i // ncol) + 1, i % ncol].set_axis_off()
    fig.suptitle(f"Stage 2: cc-free lineshape labels — {int(d['label_ok'].sum())} of {len(d)} events usable "
                 f"(segment-joint circuit; |P| ≥ {P_MIN}, P/area spread ≤ {MAX_SEG_SPREAD * 100:.0f} %)", weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / "plots" / "stage2_labels.png", dpi=130)
    print(f"wrote {OUT / 'labels.csv'}, {HERE / 'plots' / 'stage2_labels.png'}")


if __name__ == "__main__":
    main()
