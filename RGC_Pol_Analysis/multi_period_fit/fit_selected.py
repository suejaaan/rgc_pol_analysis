"""Q-meter + Dulya global fit of the raw sweep for the selected events (parallel), plus regeneration.

Per event:
  1. start values: Dulya (chi'' only) fit to the online fitsub, and the physical Q-meter fit to the
     online background (raw - fitsub), seeded with the refit of the event's own baseline
  2. joint circuit + Dulya fit of the full raw sweep from those start values
  3. P_area = online cc x sum(chi''-driven signal)   -> compare with online pol
  4. regenerate a raw sweep from the fitted parameters + white noise with the measured sigma,
     then run the online chain on it (stored baseline, cubic wing polynomial, sum 25-75 %, x cc)
     -> closure test against the online pol

Writes results/fits.yaml and results/event_<file>_<index>.npz. Run select_events.py first.
"""
import os
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import fit_events as FE  # noqa: E402
import model as M  # noqa: E402

DATA = HERE.parents[1] / "data_creation/data-d"
BASE_FITS = HERE.parent / "baseline_refit/results/baseline_fits_physical.yaml"
OUT = HERE / "results"
N_RANDOM = 30
MAX_NFEV_JOINT = 600
N_CIRCUIT_STARTS = 80
CANDIDATE_COST_FACTOR = 4.0
N_JOINT_STARTS = 3
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 4) - 2)))  # all cores but 2
WING_POLY_DEG = 3
SIGNAL_REGION = (32.48, 32.90)
SHAPE0 = dict(f0=32.685, wq=0.082, eta_q=0.11, g=0.03)


def online_chain(f, sweep, baseline, wings, cc):
    """Reproduce the online processing: baseline subtraction, cubic wing fit, 25-75 % sum, x cc."""
    n = len(f)
    idx = np.arange(n)
    m = ((idx >= int(wings[0] * n)) & (idx < int(wings[1] * n))) | ((idx >= int(wings[2] * n)) & (idx < int(wings[3] * n)))
    basesub = sweep - baseline
    fitsub = basesub - np.polyval(np.polyfit(f[m], basesub[m], WING_POLY_DEG), f)
    area = float(np.sum(fitsub[int(wings[1] * n): int(wings[2] * n)]))
    return fitsub, area, cc * area


def circuit_candidates(f, y, seed, rng):
    """All distinct circuit fits to the background whose cost is within CANDIDATE_COST_FACTOR of the best."""
    lo, hi = np.array(M.CIRCUIT_LO), np.array(M.CIRCUIT_HI)
    FE.rng = rng
    starts = ([np.asarray(seed, float)] if seed is not None else []) + [FE.random_circuit(0.0) for _ in range(N_CIRCUIT_STARTS)]
    sols = []
    for p0 in starts:
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        p0[4] = np.clip(p0[4] + np.mean(y - M.qmeter(f, *p0)), lo[4] + 1e-6, hi[4] - 1e-6)
        try:
            r = least_squares(lambda p: (M.qmeter(f, *p) - y) * FE.SCALE, p0, bounds=(lo, hi), max_nfev=400)
        except ValueError:
            continue
        sols.append((r.cost, r.x))
    best = min(c for c, _ in sols)
    keep = []
    for c, x in sorted(sols, key=lambda t: t[0]):
        if c <= CANDIDATE_COST_FACTOR * best and not any(np.allclose(x, k, rtol=1e-3) for k in keep):
            keep.append(x)
    return keep


def shape_fit_fitsub(f, y, p_start):
    """chi''-only Dulya fit (free amplitude) to the online fitsub: start values for the joint fit."""
    def mdl(s):
        return M.chi_components(f, *s)[0]

    s0 = [1.0, p_start, SHAPE0["f0"], SHAPE0["wq"], SHAPE0["eta_q"], SHAPE0["g"]]
    u = mdl(s0)
    s0[0] = np.dot(y, u) / np.dot(u, u)
    lo, hi = np.array(M.SIGNAL_LO), np.array(M.SIGNAL_HI)
    r = least_squares(lambda s: (mdl(s) - y) * FE.SCALE, np.clip(s0, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                      x_scale="jac", max_nfev=400)
    return r.x


def area_error(f, circuit, signal, res):
    J = res.jac / FE.SCALE
    cov = np.linalg.pinv(J.T @ J) * (2 * res.cost / FE.SCALE**2 / max(len(f) - J.shape[1], 1))
    theta = np.concatenate([circuit, signal])
    nc = len(M.CIRCUIT_NAMES)
    a0 = np.sum(M.absorption_dispersion_parts(f, circuit, signal)[0])
    g = np.zeros(len(theta))
    for k in range(len(theta)):
        h = 1e-6 * max(abs(theta[k]), 1e-3)
        tp = theta.copy()
        tp[k] += h
        g[k] = (np.sum(M.absorption_dispersion_parts(f, tp[:nc], tp[nc:])[0]) - a0) / h
    return float(np.sqrt(g @ cov @ g)), np.sqrt(np.clip(np.diag(cov), 0, None))


def fit_one(job):
    ev, seed_circuit = job
    t0 = time.time()
    with open(DATA / ev["file"]) as fh:
        for i, line in enumerate(fh):
            if i == ev["index"]:
                d = json.loads(line)
                break
    f, raw = np.asarray(d["freq_list"]), np.asarray(d["phase"])
    wing = (f < SIGNAL_REGION[0]) | (f > SIGNAL_REGION[1])
    fitsub = np.asarray(d["fitsub"])
    online_bg = raw - fitsub
    sig_fs = shape_fit_fitsub(f, fitsub, np.clip(ev["pol"], -0.6, 0.6))
    nc = len(M.CIRCUIT_NAMES)
    lo = np.array(list(M.CIRCUIT_LO) + list(M.SIGNAL_LO))
    hi = np.array(list(M.CIRCUIT_HI) + list(M.SIGNAL_HI))

    # Many circuits reproduce the same background but map chi into the output with different phases.
    # Keep every near-best background solution, let the signal pick: start cost with the fitsub shape.
    cands = circuit_candidates(f, online_bg, seed_circuit, np.random.default_rng(ev["index"]))
    starts = []
    for c in cands:
        A0 = FE.initial_amplitude(f, raw, c, list(sig_fs[1:]))
        x0 = np.clip(np.concatenate([c, [A0, *sig_fs[1:]]]), lo + 1e-9, hi - 1e-9)
        cost = float(np.sum((M.full_model(f, x0[:nc], x0[nc:]) - raw) ** 2))
        starts.append((cost, x0))
    starts.sort(key=lambda t: t[0])
    res = None
    for _, x0 in starts[:N_JOINT_STARTS]:
        r = least_squares(lambda x: (M.full_model(f, x[:nc], x[nc:]) - raw) * FE.SCALE, x0, bounds=(lo, hi),
                          x_scale="jac", max_nfev=MAX_NFEV_JOINT)
        if res is None or r.cost < res.cost:
            res = r
    circuit, signal = res.x[:nc], res.x[nc:]

    model = M.full_model(f, circuit, signal)
    bg = M.qmeter(f, *circuit)
    absorb, disperse = M.absorption_dispersion_parts(f, circuit, signal)
    resid = raw - model
    area_abs = float(np.sum(absorb))
    area_err, perr = area_error(f, circuit, signal, res)
    noise_sigma = float(np.std(np.diff(resid)) / np.sqrt(2))

    rng = np.random.default_rng(1000 + ev["index"])
    regenerated = model + rng.normal(0, noise_sigma, len(f))
    baseline = np.asarray(d["baseline"])
    fitsub_regen, area_regen, pol_regen = online_chain(f, regenerated, baseline, d["wings"], ev["cc"])
    _, _, pol_chain_on_data = online_chain(f, raw, baseline, d["wings"], ev["cc"])

    leg = (f > 32.9) & (f < 33.08)
    out = {
        **ev,
        "channel": d["channel"]["name"],
        "P_online": ev["pol"],
        "P_area": ev["cc"] * area_abs,
        "P_area_err": abs(ev["cc"]) * area_err,
        "mismatch_rel": (ev["cc"] * area_abs - ev["pol"]) / abs(ev["pol"]),
        "P_lineshape": float(signal[1]),
        "P_lineshape_err": float(perr[len(M.CIRCUIT_NAMES) + 1]),
        "area_online": float(d["area"]), "area_chi2": area_abs, "area_chi2_err": area_err,
        "fit_rms_V": float(np.sqrt(np.mean(resid**2))),
        "fit_rms_signal_region_V": float(np.sqrt(np.mean(resid[~wing] ** 2))),
        "noise_sigma_V": noise_sigma,
        "leg_chi1_mV": float(np.mean(disperse[leg]) * 1e3),
        "leg_data_mV": float(np.mean((raw - bg)[leg]) * 1e3),
        "P_regen_online_chain": pol_regen,
        "P_online_chain_on_data": pol_chain_on_data,
        "regen_rms_diff_V": float(np.sqrt(np.mean((regenerated - raw) ** 2))),
        "circuit": dict(zip(M.CIRCUIT_NAMES, map(float, circuit))),
        "signal": dict(zip(M.SIGNAL_NAMES, map(float, signal))),
        "fit_seconds": round(time.time() - t0, 1),
        "flag_sign": bool(np.sign(area_abs) != np.sign(d["area"])),
        "joint_fit_converged": bool(res.status > 0),
    }
    np.savez(OUT / f"event_{ev['file'][:19]}_{ev['index']}.npz", f=f, raw=raw, model=model, background=bg,
             absorb=absorb, disperse=disperse, regenerated=regenerated, fitsub_online=np.asarray(d["fitsub"]),
             fitsub_regen=fitsub_regen, baseline=baseline)
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    events = yaml.safe_load((HERE / "selected_events.yaml").read_text())
    base = yaml.safe_load(BASE_FITS.read_text())["baselines"]
    jobs = []
    for ev in events:
        b = base.get(ev["baseline_id"])
        seed = [b["params"][n] for n in M.CIRCUIT_NAMES] if b else None
        jobs.append((ev, seed))
    t0 = time.time()
    print(f"fitting {len(jobs)} events on {N_WORKERS} workers ...", flush=True)
    with Pool(N_WORKERS) as pool:
        results = []
        for r in pool.imap_unordered(fit_one, jobs):
            results.append(r)
            print(f"  {r['time']} cc {r['cc']:7.3f}  P_online {r['P_online']:+.3f}  P_area {r['P_area']:+.3f}  "
                  f"({r['mismatch_rel'] * 100:+.1f} %)  P_lineshape {r['P_lineshape']:+.3f}  "
                  f"rms {r['fit_rms_V'] * 1e6:.0f} µV  [{r['fit_seconds']:.0f} s]", flush=True)
    results.sort(key=lambda r: r["time"])
    (OUT / "fits.yaml").write_text(yaml.safe_dump(results, sort_keys=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {OUT / 'fits.yaml'}")


if __name__ == "__main__":
    main()
