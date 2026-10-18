"""Fit raw Q-meter sweeps (no baseline, no polynomial) with Q-meter + Dulya and extract P by the area method.

  P_area = cc * sum(chi''-driven part of the fitted output signal)   (online cc, same units as online area)

The lineshape asymmetry parameter P is fitted (it shapes chi'') but is not reported as a
polarization estimate.

Usage: python fit_events.py [event indices...]     (default 0 250 500)
"""
import json
import sys
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

import model as M

HERE = Path(__file__).resolve().parent
DATA_FILE = HERE.parents[1] / "data_creation/data-test/2022-09-23_00-54-02__2022-09-23_14-13-02.txt"
OUT_DIR = HERE / "results"
EVENTS = [0, 250, 500]

SIGNAL_REGION_MHZ = (32.48, 32.90)
N_STARTS_FIRST = 300
N_STARTS_NEXT = 60
SIGNAL_P0 = dict(P=-0.3, f0=32.685, wq=0.085, eta_q=0.12, g=0.015)
SCALE = 1e4

rng = np.random.default_rng(0)


def load_events(indices):
    wanted, out = set(indices), {}
    with open(DATA_FILE) as f:
        for i, line in enumerate(f):
            if i in wanted:
                out[i] = json.loads(line)
            if len(out) == len(wanted):
                break
    return out


def random_circuit(y_mean):
    return [
        10 ** rng.uniform(-1.5, 1), 10 ** rng.uniform(0, 2.3), rng.uniform(0, 7), rng.uniform(-180, 180), y_mean,
        10 ** rng.uniform(0, 1.3), 10 ** rng.uniform(-1.5, 0.5), 10 ** rng.uniform(-2.5, -0.5), rng.uniform(0, 20),
    ]


def fit_circuit_wings(f, y, wing, seeds, n_random):
    lo, hi = np.array(M.CIRCUIT_LO), np.array(M.CIRCUIT_HI)
    starts = [np.asarray(s, float) for s in seeds] + [random_circuit(0.0) for _ in range(n_random)]
    best = None
    for p0 in starts:
        p0 = np.clip(p0, lo + 1e-9, hi - 1e-9)
        p0[4] = np.clip(np.mean(y[wing] - M.qmeter(f[wing], *p0)) + p0[4], lo[4] + 1e-6, hi[4] - 1e-6)
        try:
            res = least_squares(lambda p: (M.qmeter(f[wing], *p) - y[wing]) * SCALE, p0, bounds=(lo, hi), max_nfev=400)
        except ValueError:
            continue
        if best is None or res.cost < best.cost:
            best = res
    return best.x


def initial_amplitude(f, y, circuit, sig):
    bg = M.qmeter(f, *circuit)
    unit = M.signal_component(f, circuit, [1.0, *sig])
    return float(np.dot(y - bg, unit) / np.dot(unit, unit))


def joint_fit(f, y, circuit0, signal0):
    nc = len(M.CIRCUIT_NAMES)
    lo = np.array(list(M.CIRCUIT_LO) + list(M.SIGNAL_LO))
    hi = np.array(list(M.CIRCUIT_HI) + list(M.SIGNAL_HI))
    x0 = np.clip(np.concatenate([circuit0, signal0]), lo + 1e-9, hi - 1e-9)

    def resid(x):
        return (M.full_model(f, x[:nc], x[nc:]) - y) * SCALE

    res = least_squares(resid, x0, bounds=(lo, hi), x_scale="jac", max_nfev=3000)
    return res.x[:nc], res.x[nc:], res


def param_errors(res, n_data):
    J = res.jac / SCALE
    dof = max(n_data - J.shape[1], 1)
    s2 = 2 * res.cost / SCALE**2 / dof
    try:
        cov = np.linalg.pinv(J.T @ J) * s2
    except np.linalg.LinAlgError:
        return np.full(J.shape[1], np.nan)
    return np.sqrt(np.clip(np.diag(cov), 0, None))


def main():
    indices = [int(a) for a in sys.argv[1:]] or EVENTS
    records = load_events(indices)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    seeds = []
    summary = {}

    for idx in indices:
        r = records[idx]
        f = np.asarray(r["freq_list"])
        raw = np.asarray(r["phase"])
        wing = (f < SIGNAL_REGION_MHZ[0]) | (f > SIGNAL_REGION_MHZ[1])
        print(f"\n=== event {idx}  (online pol {r['pol']:+.4f}, cc {r['cc']}) ===", flush=True)

        circuit0 = fit_circuit_wings(f, raw, wing, seeds, N_STARTS_FIRST if not seeds else N_STARTS_NEXT)
        wing_rms = float(np.sqrt(np.mean((M.qmeter(f[wing], *circuit0) - raw[wing]) ** 2)))
        print(f"stage 1 circuit-on-wings rms = {wing_rms:.3g} V", flush=True)

        sig0 = [SIGNAL_P0[k] for k in ("P", "f0", "wq", "eta_q", "g")]
        A0 = initial_amplitude(f, raw, circuit0, sig0)
        circuit, signal, res = joint_fit(f, raw, circuit0, np.array([A0, *sig0]))
        seeds.append(circuit)
        err = param_errors(res, len(f))
        resid = M.full_model(f, circuit, signal) - raw
        sigma = float(np.sqrt(np.sum(resid**2) / (len(f) - len(res.x))))
        print(f"stage 2 joint fit rms = {np.sqrt(np.mean(resid**2)):.3g} V   (wings {np.sqrt(np.mean(resid[wing]**2)):.3g})", flush=True)

        sig_out = M.signal_component(f, circuit, signal)
        absorb, disperse = M.absorption_dispersion_parts(f, circuit, signal)
        area_absorb = float(np.sum(absorb))
        area_total = float(np.sum(sig_out))

        # propagate the joint-fit covariance to both areas numerically
        J = res.jac / SCALE
        dof = max(len(f) - J.shape[1], 1)
        cov = np.linalg.pinv(J.T @ J) * (2 * res.cost / SCALE**2 / dof)
        theta = np.concatenate([circuit, signal])
        nc = len(M.CIRCUIT_NAMES)
        g_abs, g_tot = np.zeros(len(theta)), np.zeros(len(theta))
        for k in range(len(theta)):
            h = 1e-6 * max(abs(theta[k]), 1e-3)
            tp = theta.copy()
            tp[k] += h
            a_k, _ = M.absorption_dispersion_parts(f, tp[:nc], tp[nc:])
            g_abs[k] = (np.sum(a_k) - area_absorb) / h
            g_tot[k] = (np.sum(M.signal_component(f, tp[:nc], tp[nc:])) - area_total) / h
        area_absorb_err = float(np.sqrt(g_abs @ cov @ g_abs))
        area_total_err = float(np.sqrt(g_tot @ cov @ g_tot))

        cc = float(r["cc"])
        summary[idx] = {
            "start_time": r["start_time"],
            "cc_online": cc,
            "P_online": float(r["pol"]),
            "area_online": float(r["area"]),
            "area_absorb": area_absorb,
            "area_absorb_err_stat": area_absorb_err,
            "area_total_output": area_total,
            "area_total_output_err_stat": area_total_err,
            "P_area": cc * area_absorb,
            "P_area_err_stat": abs(cc) * area_absorb_err,
            "P_area_total_output": cc * area_total,
            "fit_rms_V": float(np.sqrt(np.mean(resid**2))),
            "fit_rms_wings_V": float(np.sqrt(np.mean(resid[wing] ** 2))),
            "stage1_wing_rms_V": wing_rms,
            "circuit": {k: float(v) for k, v in zip(M.CIRCUIT_NAMES, circuit)},
            "circuit_err": {k: float(v) for k, v in zip(M.CIRCUIT_NAMES, err[:nc])},
            "signal": {k: float(v) for k, v in zip(M.SIGNAL_NAMES, signal)},
            "signal_err": {k: float(v) for k, v in zip(M.SIGNAL_NAMES, err[nc:])},
        }
        np.savez(
            OUT_DIR / f"event_{idx}_arrays.npz",
            f=f, raw=raw, model=M.full_model(f, circuit, signal), background=M.qmeter(f, *circuit),
            signal=sig_out, absorb=absorb, disperse=disperse,
            online_basesub=np.asarray(r["basesub"]), online_fitcurve=np.asarray(r["fitcurve"]),
            online_baseline=np.asarray(r["baseline"]), online_fitsub=np.asarray(r["fitsub"]),
            sigma=sigma,
        )
        print(f"P_online = {r['pol']:+.4f}   P_area = {cc * area_absorb:+.4f} ± {abs(cc) * area_absorb_err:.4f}", flush=True)

    out_yaml = OUT_DIR / "results.yaml"
    merged = yaml.safe_load(out_yaml.read_text()) if out_yaml.exists() else {}
    merged.update({int(k): v for k, v in summary.items()})
    with open(out_yaml, "w") as fh:
        yaml.safe_dump(dict(sorted(merged.items())), fh, sort_keys=False)
    print(f"\nwrote {OUT_DIR / 'results.yaml'}")


if __name__ == "__main__":
    main()
