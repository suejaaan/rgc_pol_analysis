"""Refit every unique data-d baseline sweep with the bounded physical Q-meter circuit.

Replaces data_creation/fitting/baseline_fits_data_d.yaml, where all 21 circuit
parameters were fitted unbounded and every fit has negative L0/Rcoil/r/M/Cstray.
Here the hardware constants are fixed and only 9 parameters are fitted, all inside
physical (positive) bounds; see ../qmeter_dulya_fit/model.py.

Pass 1: independent multi-start fit per baseline (parallel).
Pass 2: refit baselines whose rms is well above the median, seeded with the best
        solutions of the baselines nearest in time.

Usage: python refit_baselines.py [baseline ids...]     (default: all non-zero baselines)
Run collect_baselines.py first.
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import yaml
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import model as M  # noqa: E402

CACHE = HERE / "cache"
OUT_YAML = HERE / "results" / "baseline_fits_physical.yaml"
N_RANDOM = 200
N_WORKERS = 14
SCALE = 1e4
RETRY_FACTOR = 3.0
N_SEED_NEIGHBORS = 6
AT_BOUND_FRAC = 1e-3
EDGE_BINS = 3  # sweep-edge artifacts at the first/last bins are excluded from the fit

LO, HI = np.array(M.CIRCUIT_LO), np.array(M.CIRCUIT_HI)


def random_start(rng):
    return np.array([
        10 ** rng.uniform(-1.5, 1), 10 ** rng.uniform(-1.3, 2.3), rng.uniform(0, 7), rng.uniform(-180, 180), 0.0,
        10 ** rng.uniform(0, 1.3), 10 ** rng.uniform(-1.5, 0.5), 10 ** rng.uniform(-2.5, -0.5), rng.uniform(0, 20),
    ])


def fit_one(args):
    bid, f_all, y_all, seeds, n_random = args
    f, y = f_all[EDGE_BINS:-EDGE_BINS], y_all[EDGE_BINS:-EDGE_BINS]
    rng = np.random.default_rng(bid)
    starts = [np.asarray(s, float) for s in seeds] + [random_start(rng) for _ in range(n_random)]
    best = None
    t0 = time.time()
    for p0 in starts:
        p0 = np.clip(p0, LO + 1e-9, HI - 1e-9)
        p0[4] = np.clip(p0[4] + np.mean(y - M.qmeter(f, *p0)), LO[4] + 1e-6, HI[4] - 1e-6)
        try:
            res = least_squares(lambda p: (M.qmeter(f, *p) - y) * SCALE, p0, bounds=(LO, HI), max_nfev=400)
        except ValueError:
            continue
        if best is None or res.cost < best.cost:
            best = res
    resid = M.qmeter(f, *best.x) - y
    J = best.jac / SCALE
    dof = max(len(y) - J.shape[1], 1)
    cov = np.linalg.pinv(J.T @ J) * (np.sum(resid**2) / dof)
    return {
        "id": bid,
        "params": best.x,
        "errors": np.sqrt(np.clip(np.diag(cov), 0, None)),
        "rms": float(np.sqrt(np.mean(resid**2))),
        "max_abs_resid": float(np.max(np.abs(resid))),
        "ptp": float(np.ptp(y)),
        "seconds": time.time() - t0,
        "n_starts": len(starts),
    }


def load(ids=None):
    meta = yaml.safe_load((CACHE / "baselines_meta.yaml").read_text())
    z = np.load(CACHE / "baselines.npz")
    out = []
    for m in meta:
        b, f = z["baseline"][m["id"]], z["freq"][m["id"]]
        ok = ~np.isnan(b)
        b, f = b[ok], f[ok]
        if not np.any(b) or (ids and m["id"] not in ids):
            continue
        out.append((m, f, b))
    return out


def at_bound(p):
    """Positive-bounded params: within 2 % of the bound value (they span decades); others: 0.1 % of the range."""
    out = []
    for n, v, lo, hi in zip(M.CIRCUIT_NAMES, p, LO, HI):
        if lo > 0:
            hit = v < lo * 1.02 or v > hi * 0.98
        else:
            hit = min(v - lo, hi - v) < AT_BOUND_FRAC * (hi - lo)
        if hit:
            out.append(n)
    return out


def main():
    ids = {int(a) for a in sys.argv[1:]} or None
    items = load(ids)
    print(f"fitting {len(items)} baselines with {N_WORKERS} workers, {N_RANDOM} random starts each", flush=True)
    t0 = time.time()

    with Pool(N_WORKERS) as pool:
        results = {r["id"]: r for r in pool.imap_unordered(
            fit_one, [(m["id"], f, y, [], N_RANDOM) for m, f, y in items])}
        print(f"pass 1 done in {time.time() - t0:.0f} s", flush=True)

        rel = {i: r["rms"] / max(r["ptp"], 1e-12) for i, r in results.items()}
        med = np.median(list(rel.values()))
        order = sorted(items, key=lambda it: it[0]["base_stamp"] or 0)
        pos = {m["id"]: k for k, (m, _, _) in enumerate(order)}
        retry = []
        for m, f, y in items:
            if rel[m["id"]] > RETRY_FACTOR * med:
                k = pos[m["id"]]
                near = [order[j][0]["id"] for j in range(max(0, k - N_SEED_NEIGHBORS), min(len(order), k + N_SEED_NEIGHBORS + 1)) if j != k]
                retry.append((m["id"], f, y, [results[n]["params"] for n in near], N_RANDOM // 2))
        if retry:
            print(f"pass 2: retrying {len(retry)} baselines with neighbour seeds", flush=True)
            for r in pool.imap_unordered(fit_one, retry):
                if r["rms"] < results[r["id"]]["rms"]:
                    r["improved_in_pass2"] = True
                    results[r["id"]] = r
    elapsed = time.time() - t0

    out = {}
    for m, f, y in items:
        r = results[m["id"]]
        out[m["id"]] = {
            "base_file": m["base_file"], "base_time": m["base_time"], "channel": m["channel"],
            "n_events": m["n_events"], "first_event": m["first_event"], "last_event": m["last_event"],
            "labels": m["labels"], "cc_values": m["cc_values"],
            "freq_min_mhz": float(f[0]), "freq_max_mhz": float(f[-1]), "n_bins": len(f),
            "rms_V": r["rms"], "edge_bins_excluded": EDGE_BINS, "max_abs_resid_V": r["max_abs_resid"], "ptp_V": r["ptp"],
            "nrmse": r["rms"] / r["ptp"] if r["ptp"] else None,
            "params": {k: float(v) for k, v in zip(M.CIRCUIT_NAMES, r["params"])},
            "param_errors": {k: float(v) for k, v in zip(M.CIRCUIT_NAMES, r["errors"])},
            "params_at_bound": at_bound(r["params"]),
            "fit_seconds": round(r["seconds"], 1),
            "improved_in_pass2": bool(r.get("improved_in_pass2", False)),
        }
    OUT_YAML.parent.mkdir(parents=True, exist_ok=True)
    header = {
        "model": "bounded Liverpool Q-meter, ../qmeter_dulya_fit/model.py (qmeter, chi = 0)",
        "fitted_params": list(M.CIRCUIT_NAMES),
        "bounds_lo": [float(v) for v in LO], "bounds_hi": [float(v) for v in HI],
        "n_baselines": len(out), "wall_seconds": round(elapsed, 1),
    }
    OUT_YAML.write_text(yaml.safe_dump({"info": header, "baselines": out}, sort_keys=False))

    rms = np.array([v["rms_V"] for v in out.values()])
    print(f"\nwrote {OUT_YAML}  ({elapsed / 60:.1f} min wall)")
    print(f"rms: median {np.median(rms) * 1e6:.1f} µV, 90% {np.percentile(rms, 90) * 1e6:.1f} µV, max {rms.max() * 1e6:.1f} µV")
    print(f"baselines with a parameter at a bound: {sum(bool(v['params_at_bound']) for v in out.values())}")


if __name__ == "__main__":
    main()
