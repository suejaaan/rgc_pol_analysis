"""Segment-joint fit: all events of one tune segment share one circuit (same stored baseline = same tune).

Shared per segment:  Cknob, trim, L0, Rcoil, alpha, Cstray  (coil, cable, tuning capacitor)
                     sigma                                   (Gaussian, inhomogeneous broadening)
Per event:           U, phi_deg, DC_offset                   (gain / phase / offset drift)
                     A, P, f0, g                             (signal; eta, wq fixed)
Signal model: Dulya powder lineshape (Lorentzian-type g) convolved with a Gaussian sigma (model_variants.py).

The circuit is strongly degenerate: single-event fits of the same tune land in different circuit families,
and P follows the family. Procedure per segment:
  1. screen: every start circuit (good single-event fits of ALL segments + this segment's pilot fits +
     the stored-baseline refit) is tried on N_SCREEN events with the shared part fixed
  2. joint: the N_JOINT best starts get the full joint fit (sparse finite-difference Jacobian)
  3. rescue: events whose signal-region rms is > RESCUE x the segment median are restarted from their
     neighbours' parameters (shared part fixed), then the joint fit is polished
The start with the lowest total cost wins; the other joint solutions give the circuit systematic on P.

Usage: python segment_fit.py [segment ...]   (default 112 119 125 133)
Output: results/stage2/segment_test.yaml, plots/stage2_segment_test.png
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
from scipy.sparse import lil_matrix

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
sys.path.insert(0, str(HERE.parent / "multi_period_fit"))
import fit_selected as FS  # noqa: E402
import model as M  # noqa: E402
import model_variants as V  # noqa: E402
import stage2_fit as S2  # noqa: E402

OUT = HERE / "results" / "stage2"
ETA, WQ = V.ETA_FIX, V.WQ_FIX
SHARED_C = [1, 2, 5, 6, 7, 8]   # Cknob, trim, L0, Rcoil, alpha, Cstray
OWN_C = [0, 3, 4]               # U, phi_deg, DC_offset
NS, NO = 7, 7                   # shared: 6 circuit + sigma; own: U, phi, DC, A, P, f0, g
N_EV = 8
N_SCREEN = 3
N_JOINT = 3
RESCUE = 2.0
GOOD_RMS_SIGNAL = 30e-6         # single-event fits used as global circuit starts
SIGMA0, G0 = 0.0045, 0.01
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
CLO, CHI = np.array(M.CIRCUIT_LO), np.array(M.CIRCUIT_HI)
SH_LO = np.concatenate([CLO[SHARED_C], [0.0]])
SH_HI = np.concatenate([CHI[SHARED_C], [V.SIG_HI]])
OW_LO = np.concatenate([CLO[OWN_C], [-np.inf, -0.95, 32.60, 0.001]])
OW_HI = np.concatenate([CHI[OWN_C], [np.inf, 0.95, 32.77, 0.08]])
SCALE = 1e4


def ev_model(f, sh, own):
    c = np.empty(9)
    c[SHARED_C] = sh[:6]
    c[OWN_C] = own[:3]
    A, P, f0, g = own[3:]
    return V.model(f, np.concatenate([c, [A, P, f0, WQ, ETA, g, sh[6]]]))


def fit_own(f, raw, sh, own0, max_nfev=400):
    r = least_squares(lambda o: (ev_model(f, sh, o) - raw) * SCALE, np.clip(own0, OW_LO + 1e-9, OW_HI - 1e-9),
                      bounds=(OW_LO, OW_HI), x_scale="jac", max_nfev=max_nfev)
    return r.x, r.cost


def joint_fit(fs, raws, sh0, owns0, max_nfev=200):
    n, nb = len(fs), len(fs[0])
    lo = np.concatenate([SH_LO] + [OW_LO] * n)
    hi = np.concatenate([SH_HI] + [OW_HI] * n)
    x0 = np.clip(np.concatenate([sh0] + list(owns0)), lo + 1e-9, hi - 1e-9)
    sp = lil_matrix((n * nb, len(x0)), dtype=int)
    sp[:, :NS] = 1
    for i in range(n):
        sp[i * nb:(i + 1) * nb, NS + NO * i:NS + NO * (i + 1)] = 1

    def resid(x):
        return np.concatenate([(ev_model(fs[i], x[:NS], x[NS + NO * i:NS + NO * (i + 1)]) - raws[i]) * SCALE
                               for i in range(n)])

    r = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", jac_sparsity=sp, max_nfev=max_nfev)
    return r.x[:NS], [r.x[NS + NO * i:NS + NO * (i + 1)] for i in range(n)], r.cost


def rms_parts(f, raw, sh, own):
    res = raw - ev_model(f, sh, own)
    s = (f > V.SREG[0]) & (f < V.SREG[1])
    return float(np.sqrt(np.mean(res[s] ** 2))), float(np.sqrt(np.mean(res[~s] ** 2)))


def load(evs):
    fs, raws = [], []
    for e in evs:
        if "mc_raw" in e:  # Monte Carlo event carried in memory (stage 3 closure)
            fs.append(np.asarray(e["mc_f"]))
            raws.append(np.asarray(e["mc_raw"]))
            continue
        d = S2.read_event(e["file"], int(e["index"]))
        fs.append(np.asarray(d["freq_list"]))
        raws.append(np.asarray(d["phase"]))
    return fs, raws


def own_start(circuit, s):
    return np.array([circuit[0], circuit[3], circuit[4], s["A"], s["P"], s["f0"], G0])


def screen(job):
    seg, label, circuit, sig, evs = job
    idx = np.linspace(0, len(evs) - 1, N_SCREEN).astype(int)
    fs, raws = load([evs[i] for i in idx])
    sh = np.concatenate([np.asarray(circuit)[SHARED_C], [SIGMA0]])
    cost = sum(fit_own(f, raw, sh, own_start(circuit, sig[evs[i]["time"]]), 200)[1]
               for i, f, raw in zip(idx, fs, raws))
    return seg, label, float(cost)


def joint_segment(job):
    seg, starts, sig, evs = job
    t0 = time.time()
    fs, raws = load(evs)
    sols = []
    for label, circuit in starts:
        sh = np.concatenate([np.asarray(circuit)[SHARED_C], [SIGMA0]])
        owns = [fit_own(f, raw, sh, own_start(circuit, sig[e["time"]]))[0] for e, f, raw in zip(evs, fs, raws)]
        sh, owns, cost = joint_fit(fs, raws, sh, owns)
        # rescue events stuck in a bad local minimum
        rs = np.array([rms_parts(f, raw, sh, o)[0] for f, raw, o in zip(fs, raws, owns)])
        bad = np.flatnonzero(rs > RESCUE * np.median(rs))
        for i in bad:
            best = None
            for j in (i - 1, i + 1):
                if 0 <= j < len(evs) and j not in bad:
                    o0 = owns[j].copy()
                    o0[3] *= evs[i]["area_online"] / evs[j]["area_online"]
                    o, c = fit_own(fs[i], raws[i], sh, o0)
                    if best is None or c < best[1]:
                        best = (o, c)
            if best is not None:
                owns[i] = best[0]
        if len(bad):
            sh, owns, cost = joint_fit(fs, raws, sh, owns, max_nfev=100)
        rows = []
        for e, f, raw, o in zip(evs, fs, raws, owns):
            r_s, r_w = rms_parts(f, raw, sh, o)
            rows.append({"time": e["time"], "P": float(o[4]), "A": float(o[3]), "U": float(o[0]),
                         "phi_deg": float(o[1]), "DC_offset": float(o[2]), "f0": float(o[5]), "g": float(o[6]),
                         "area_online": float(e["area_online"]), "rms_signal_uV": r_s * 1e6, "rms_wings_uV": r_w * 1e6})
        sols.append({"start": label, "cost": float(cost), "n_rescued": int(len(bad)),
                     "shared": dict(zip([M.CIRCUIT_NAMES[k] for k in SHARED_C] + ["sigma_kHz"],
                                        [float(v) for v in sh[:6]] + [float(sh[6] * 1e3)])),
                     "events": rows})
    sols.sort(key=lambda s: s["cost"])
    return {"segment": seg, "solutions": sols, "seconds": round(time.time() - t0)}


def segment_setup(seg, pilot, picks, base, global_starts):
    evs = picks[picks["segment"] == seg].sort_values("time")
    evs = evs.iloc[np.linspace(0, len(evs) - 1, min(N_EV, len(evs))).astype(int)].to_dict("records")
    pil = [p for p in pilot if p["segment"] == seg]
    starts = [(f"own pilot {p['time'][5:16]}", [p["circuit"][n] for n in M.CIRCUIT_NAMES]) for p in pil]
    starts += [s for s in global_starts if f"seg {seg} " not in s[0]]
    b = base.get(int(evs[0]["baseline_id"]))
    if b:
        starts.append(("baseline refit", [b["params"][n] for n in M.CIRCUIT_NAMES]))
    tp = np.array([np.datetime64(p["time"]) for p in pil])
    sig = {}
    for e in evs:  # signal start: nearest pilot fit in time, amplitude scaled by the online area
        p = pil[int(np.argmin(np.abs(tp - np.datetime64(e["time"]))))]
        sig[e["time"]] = dict(p["signal"], A=p["signal"]["A"] * e["area_online"] / p["area_online"])
    return evs, starts, sig


def global_circuit_starts(pilot):
    """Distinct circuits of good single-event fits (all segments)."""
    out, seen = [], []
    for p in sorted(pilot, key=lambda p: p["rms_signal_V"]):
        if p["rms_signal_V"] > GOOD_RMS_SIGNAL:
            continue
        c = [p["circuit"][n] for n in M.CIRCUIT_NAMES]
        key = np.array([np.log10(c[0]), np.log10(c[1]), c[2], np.log10(c[5]), c[8] / 10])
        if not any(np.allclose(key, k, atol=0.3) for k in seen):
            seen.append(key)
            out.append((f"good fit seg {p['segment']} {p['time'][5:16]}", c))
    return out


def run_segments(segs, picks_path=HERE / "results" / "plateau_events.csv"):
    pilot = yaml.safe_load((OUT / "fits_pilot.yaml").read_text())
    picks = pd.read_csv(picks_path)
    base = yaml.safe_load(FS.BASE_FITS.read_text())["baselines"]
    gstarts = global_circuit_starts(pilot)
    setup = {s: segment_setup(s, pilot, picks, base, gstarts) for s in segs}
    jobs = [(s, lab, c, setup[s][2], setup[s][0]) for s in segs for lab, c in setup[s][1]]
    print(f"{len(gstarts)} global circuit starts; screening {len(jobs)} (segment, start) pairs", flush=True)
    t0 = time.time()
    with Pool(N_WORKERS) as pool:
        scr = pool.map(screen, jobs, chunksize=1)
        print(f"screening done in {(time.time() - t0) / 60:.1f} min", flush=True)
        jjobs = []
        for s in segs:
            ranked = sorted([x for x in scr if x[0] == s], key=lambda x: x[2])[:N_JOINT]
            names = {lab for _, lab, _ in ranked}
            starts = [st for st in setup[s][1] if st[0] in names]
            print(f"  segment {s}: best starts {[(lab, round(c, 1)) for _, lab, c in ranked]}", flush=True)
            jjobs.append((s, starts, setup[s][2], setup[s][0]))
        res = pool.map(joint_segment, jjobs, chunksize=1)
    print(f"joint fits done in {(time.time() - t0) / 60:.1f} min", flush=True)
    return res


def main():
    segs = [int(s) for s in sys.argv[1:]] or [112, 119, 125, 133]
    res = run_segments(segs)
    (OUT / "segment_test.yaml").write_text(yaml.safe_dump(res, sort_keys=False))
    pilot = yaml.safe_load((OUT / "fits_pilot.yaml").read_text())
    fig, axes = plt.subplots(2, len(segs), figsize=(5 * len(segs), 8), squeeze=False)
    for k, r in enumerate(res):
        seg = r["segment"]
        print(f"\nsegment {seg}  [{r['seconds']} s]")
        for s in r["solutions"]:
            P = np.array([e["P"] for e in s["events"]])
            ratio = P / np.array([e["area_online"] for e in s["events"]])
            print(f"  {s['start']:26s} cost {s['cost']:8.1f}  σ {s['shared']['sigma_kHz']:.1f} kHz  rescued {s['n_rescued']}  "
                  f"|P| {np.round(np.abs(P), 3).tolist()}  P/area spread {np.std(ratio) / abs(np.mean(ratio)) * 100:.1f}%  "
                  f"rms sig {np.median([e['rms_signal_uV'] for e in s['events']]):.1f} µV")
        pil = sorted([p for p in pilot if p["segment"] == seg], key=lambda p: p["time"])
        best = r["solutions"][0]["cost"]
        for ax, key in ((axes[0, k], "P"), (axes[1, k], "ratio")):
            for i, s in enumerate(r["solutions"]):
                t = [np.datetime64(e["time"]) for e in s["events"]]
                y = [abs(e["P"]) if key == "P" else e["P"] / e["area_online"] for e in s["events"]]
                ax.plot(t, y, "o-" if i == 0 else "x--", ms=6 if i == 0 else 5, lw=1.4 if i == 0 else 0.8,
                        label=f"joint {'(best)' if i == 0 else ''} {s['start'][:24]}, cost ×{s['cost'] / best:.2f}")
            t = [np.datetime64(p["time"]) for p in pil]
            y = [abs(p["P_lineshape"]) if key == "P" else p["P_lineshape"] / p["area_online"] for p in pil]
            ax.plot(t, y, "s", color="0.5", mfc="none", ms=7, label="single-event pilot fits")
            ax.set_ylabel("|P| lineshape" if key == "P" else "P / online area (cc-free ratio)")
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=25, labelsize=7.5)
        axes[0, k].set_title(f"segment {seg}", weight="bold")
        axes[0, k].legend(fontsize=7)
    fig.suptitle("Segment-joint fit (one shared circuit + σ per tune) vs single-event fits", weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "stage2_segment_test.png", dpi=130)


if __name__ == "__main__":
    main()
