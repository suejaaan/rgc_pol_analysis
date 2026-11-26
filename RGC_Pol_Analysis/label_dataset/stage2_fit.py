"""Stage 2: high-quality Q-meter + Dulya fits of the plateau events (no cc used in the fit).

Per event (reuses the helpers of ../multi_period_fit/fit_selected.py):
  1. start values: chi''-only Dulya fit to the online fitsub, started at P = 0.3 with the sign of the
     signal (not the online P); circuit candidates fitted to the background (raw - fitsub), seeded
     with the refit of the event's stored baseline
  2. joint fit of the raw sweep, all 15 parameters free (eta and wq free, unlike run_lineshape.py),
     best of N_JOINT_STARTS circuit candidates
  3. stability: refit from the best solution with P restarted at P_ALT_STARTS -> P spread
  4. covariance from the Jacobian, noise sigma, residual
  5. quality gate (see GATE): rms vs noise, structure in the signal region, parameters at a bound, P stability

Usage:  python stage2_fit.py pilot    one event per plateau (the middle pick)
        python stage2_fit.py rest     all other picked events (skips events already fitted)
Outputs: results/stage2/fits_<mode>.yaml, results/stage2/npz/event_<file>_<index>.npz
"""
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "multi_period_fit"))
sys.path.insert(0, str(ROOT / "qmeter_dulya_fit"))
import fit_events as FE  # noqa: E402
import fit_selected as FS  # noqa: E402
import model as M  # noqa: E402

OUT = HERE / "results" / "stage2"
NPZ = OUT / "npz"
P_START = 0.3
P_ALT_STARTS = (0.12, 0.5)
N_JOINT_STARTS = 3
MAX_NFEV = 800
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
NC = len(M.CIRCUIT_NAMES)
GATE = dict(rms_over_noise=2.0,        # total rms < 2 x white-noise sigma
            signal_over_wings=1.5,     # residual rms in the signal region < 1.5 x that in the wings
            p_spread=0.02,             # |P| spread over restarts
            bound_frac=1e-3)           # parameter within 0.1 % of its range from a bound (A and P excluded)


def read_event(file, index):
    with open(FS.DATA / file) as fh:
        for i, line in enumerate(fh):
            if i == index:
                return json.loads(line)


def joint(f, raw, x0, lo, hi):
    return least_squares(lambda x: (M.full_model(f, x[:NC], x[NC:]) - raw) * FE.SCALE,
                         np.clip(x0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), x_scale="jac", max_nfev=MAX_NFEV)


def fit_event(job):
    ev, seed = job
    t0 = time.time()
    d = read_event(ev["file"], int(ev["index"]))
    f, raw = np.asarray(d["freq_list"]), np.asarray(d["phase"])
    fitsub = np.asarray(d["fitsub"])
    lo = np.array(list(M.CIRCUIT_LO) + list(M.SIGNAL_LO))
    hi = np.array(list(M.CIRCUIT_HI) + list(M.SIGNAL_HI))

    # P sign guess from the signal only: online area and P have opposite signs in every period (cc < 0)
    p0 = -np.sign(ev["area_online"]) * P_START
    sig_fs = FS.shape_fit_fitsub(f, fitsub, p0)
    cands = FS.circuit_candidates(f, raw - fitsub, seed, np.random.default_rng(int(ev["index"])))
    starts = []
    for c in cands:
        A0 = FE.initial_amplitude(f, raw, c, list(sig_fs[1:]))
        x0 = np.clip(np.concatenate([c, [A0, *sig_fs[1:]]]), lo + 1e-9, hi - 1e-9)
        starts.append((float(np.sum((M.full_model(f, x0[:NC], x0[NC:]) - raw) ** 2)), x0))
    starts.sort(key=lambda t: t[0])
    res = None
    for _, x0 in starts[:N_JOINT_STARTS]:
        r = joint(f, raw, x0, lo, hi)
        if res is None or r.cost < res.cost:
            res = r

    # stability: restart P from other values, keep the rest of the best solution
    p_restarts = []
    for pa in P_ALT_STARTS:
        x0 = res.x.copy()
        x0[NC + 1] = np.sign(res.x[NC + 1] or p0) * pa
        x0[NC] = FE.initial_amplitude(f, raw, x0[:NC], list(x0[NC + 1:]))
        r = joint(f, raw, x0, lo, hi)
        p_restarts.append((float(r.x[NC + 1]), float(r.cost)))
        if r.cost < res.cost * (1 - 1e-6):
            res = r
    x = res.x
    circuit, signal = x[:NC], x[NC:]

    model = M.full_model(f, circuit, signal)
    bg = M.qmeter(f, *circuit)
    absorb, disperse = M.absorption_dispersion_parts(f, circuit, signal)
    resid = raw - model
    noise = float(np.std(np.diff(resid)) / np.sqrt(2))
    J = res.jac / FE.SCALE
    dof = max(len(f) - J.shape[1], 1)
    cov = np.linalg.pinv(J.T @ J) * (np.sum(resid**2) / dof)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    sreg = (f > FS.SIGNAL_REGION[0]) & (f < FS.SIGNAL_REGION[1])
    rms, rms_sig, rms_wing = (float(np.sqrt(np.mean(r**2))) for r in (resid, resid[sreg], resid[~sreg]))
    span = hi - lo
    at_bound = [n for k, n in enumerate(list(M.CIRCUIT_NAMES) + list(M.SIGNAL_NAMES))
                if n not in ("A", "P") and min(x[k] - lo[k], hi[k] - x[k]) < GATE["bound_frac"] * span[k]]
    p_all = [float(signal[1])] + [p for p, c in p_restarts if c < 1.05 * res.cost]  # restarts that fit as well
    p_spread = float(np.ptp(np.abs(p_all)))
    gate = {
        "rms_ok": rms < GATE["rms_over_noise"] * noise,
        "residual_structure_ok": rms_sig < GATE["signal_over_wings"] * rms_wing,
        "no_bound": not at_bound,
        "p_stable": p_spread < GATE["p_spread"],
    }
    out = {
        "time": ev["time"], "file": ev["file"], "index": int(ev["index"]), "segment": int(ev["segment"]),
        "plateau": int(ev["plateau"]), "baseline_id": int(ev["baseline_id"]),
        "P_lineshape": float(signal[1]), "P_lineshape_err": float(perr[NC + 1]),
        "P_restarts": p_restarts, "P_spread": p_spread,
        "area_chi2": float(np.sum(absorb)), "area_online": float(ev["area_online"]),
        "rms_V": rms, "rms_signal_V": rms_sig, "rms_wings_V": rms_wing, "noise_sigma_V": noise,
        "at_bound": at_bound, "gate": gate, "pass": bool(all(gate.values())),
        "circuit": dict(zip(M.CIRCUIT_NAMES, map(float, circuit))),
        "signal": dict(zip(M.SIGNAL_NAMES, map(float, signal))),
        "errors": dict(zip(list(M.CIRCUIT_NAMES) + list(M.SIGNAL_NAMES), map(float, perr))),
        "ref_online_cc": float(ev["cc"]), "ref_online_P": float(ev["pol_online"]),
        "ref_period_verdict": ev["period_verdict"],
        "fit_seconds": round(time.time() - t0, 1),
    }
    np.savez(NPZ / f"event_{ev['file'][:19]}_{int(ev['index'])}.npz", f=f, raw=raw, model=model, background=bg,
             absorb=absorb, disperse=disperse, resid=resid, cov=cov, x=x)
    return out


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "pilot"
    NPZ.mkdir(parents=True, exist_ok=True)
    picks = pd.read_csv(HERE / "results" / "plateau_events.csv")
    if mode == "pilot":
        sel = picks.groupby("plateau", group_keys=False).apply(lambda g: g.iloc[[len(g) // 2]])
    else:
        done = set()
        for p in OUT.glob("fits_*.yaml"):
            done |= {(r["file"], r["index"]) for r in yaml.safe_load(p.read_text())}
        sel = picks[[(r.file, r.index) not in done for r in picks.itertuples()]]
    base = yaml.safe_load(FS.BASE_FITS.read_text())["baselines"]
    jobs = []
    for ev in sel.to_dict("records"):
        b = base.get(int(ev["baseline_id"]))
        jobs.append((ev, [b["params"][n] for n in M.CIRCUIT_NAMES] if b else None))
    t0 = time.time()
    print(f"stage 2 ({mode}): fitting {len(jobs)} events on {N_WORKERS} workers", flush=True)
    results = []
    with Pool(N_WORKERS) as pool:
        for k, r in enumerate(pool.imap_unordered(fit_event, jobs), 1):
            results.append(r)
            print(f"  [{k:4d}/{len(jobs)}] {r['time']} seg {r['segment']:3d}  P_ls {r['P_lineshape']:+.3f} ± "
                  f"{r['P_lineshape_err']:.3f} (spread {r['P_spread']:.3f})  rms/noise {r['rms_V'] / r['noise_sigma_V']:.2f}  "
                  f"{'PASS' if r['pass'] else 'fail ' + ','.join(k for k, v in r['gate'].items() if not v)}  "
                  f"[{r['fit_seconds']:.0f} s, {(time.time() - t0) / 60:.1f} min]", flush=True)
            if k % 20 == 0:  # checkpoint
                (OUT / f"fits_{mode}.yaml").write_text(yaml.safe_dump(sorted(results, key=lambda r: r["time"]), sort_keys=False))
    results.sort(key=lambda r: r["time"])
    (OUT / f"fits_{mode}.yaml").write_text(yaml.safe_dump(results, sort_keys=False))
    n_pass = sum(r["pass"] for r in results)
    print(f"done in {(time.time() - t0) / 60:.1f} min: {n_pass}/{len(results)} pass the gate -> {OUT / f'fits_{mode}.yaml'}")


if __name__ == "__main__":
    main()
