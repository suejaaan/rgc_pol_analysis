"""Stage 2 model test: does extra Gaussian broadening remove the systematic horn residual, and how much does P move?

chi'' = A [r I+(x) + I-(x)] convolved with a Gaussian of width sigma (MHz, inhomogeneous broadening),
chi' = its Hilbert partner, both inside the Q-meter (as in ../qmeter_dulya_fit/model.py).
Variants, each warm-started from the event's pilot fit:
  A  eta, wq fixed (ETA_FIX, WQ_FIX), sigma = 0      (current model, fixed shape)
  B  eta, wq fixed, sigma free
  C  eta, wq, sigma free
Output: results/stage2/model_variants.yaml, plots/stage2_model_variants.png
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares
from scipy.signal import hilbert

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import model as M  # noqa: E402

OUT = HERE / "results" / "stage2"
ETA_FIX, WQ_FIX = 0.111, 0.0825
N_EVENTS = 20
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
NC = len(M.CIRCUIT_NAMES)
SIG_LO, SIG_HI = 0.0, 0.03     # MHz
SCALE = 1e4
SREG = (32.48, 32.90)


def chi_gauss(f, A, P, f0, wq, eta, g, sigma):
    df = float(np.median(np.diff(f)))
    grid = np.arange(f[0] - M.PAD_MHZ, f[-1] + M.PAD_MHZ, df)
    x = (grid - f0) / wq
    chi2 = A * (M.p_to_r(P) * M.PowderLineshape(x, 1, eta, g, nphi=M.POWDER_NPHI)
                + M.PowderLineshape(x, -1, eta, g, nphi=M.POWDER_NPHI))
    if sigma > 1e-5:
        k = np.arange(-int(5 * sigma / df) - 1, int(5 * sigma / df) + 2) * df
        ker = np.exp(-0.5 * (k / sigma) ** 2)
        chi2 = np.convolve(chi2, ker / ker.sum(), mode="same")
    chi1 = -np.imag(hilbert(chi2))
    return np.interp(f, grid, chi2), np.interp(f, grid, chi1)


def model(f, theta):
    """theta = 9 circuit + (A, P, f0, wq, eta, g, sigma)."""
    chi2, chi1 = chi_gauss(f, *theta[NC:])
    return M.qmeter(f, *theta[:NC], chi=chi1 - 1j * chi2)


def fit_variant(f, raw, theta0, free):
    lo = np.array(list(M.CIRCUIT_LO) + list(M.SIGNAL_LO) + [SIG_LO])
    hi = np.array(list(M.CIRCUIT_HI) + list(M.SIGNAL_HI) + [SIG_HI])
    idx = np.flatnonzero(free)

    def resid(p):
        th = theta0.copy()
        th[idx] = p
        return (model(f, th) - raw) * SCALE

    best = None
    for s0 in ([theta0[-1]] if not free[-1] else [0.002, 0.008]):  # sigma starts
        th = theta0.copy()
        th[-1] = s0
        p0 = np.clip(th[idx], lo[idx] + 1e-9, hi[idx] - 1e-9)
        r = least_squares(lambda p: resid(p), p0, bounds=(lo[idx], hi[idx]), x_scale="jac", max_nfev=800)
        if best is None or r.cost < best.cost:
            best = r
    th = theta0.copy()
    th[idx] = best.x
    res = raw - model(f, th)
    J = best.jac / SCALE
    cov = np.linalg.pinv(J.T @ J) * (np.sum(res**2) / max(len(f) - len(idx), 1))
    k_p = list(idx).index(NC + 1)
    return th, res, float(np.sqrt(cov[k_p, k_p]))


def run(e):
    z = np.load(OUT / "npz" / f"event_{e['file'][:19]}_{e['index']}.npz")
    f, raw = z["f"], z["raw"]
    theta = np.concatenate([z["x"], [0.0]])
    theta[NC + 3], theta[NC + 4] = WQ_FIX, ETA_FIX
    sreg = (f > SREG[0]) & (f < SREG[1])
    out = {"time": e["time"], "segment": e["segment"], "P_pilot": e["P_lineshape"], "area_online": e["area_online"],
           "ref_online_P": e["ref_online_P"]}
    free_base = np.ones(len(theta), bool)
    for name, fix in (("A", {NC + 3, NC + 4, NC + 6}), ("B", {NC + 3, NC + 4}), ("C", set())):
        free = free_base.copy()
        free[list(fix)] = False
        th, res, perr = fit_variant(f, raw, theta, free)
        out[name] = {"P": float(th[NC + 1]), "P_err": perr, "eta": float(th[NC + 4]), "wq": float(th[NC + 3]),
                     "g": float(th[NC + 5]), "sigma_kHz": float(th[NC + 6] * 1e3),
                     "rms_signal_uV": float(np.sqrt(np.mean(res[sreg] ** 2)) * 1e6),
                     "rms_wings_uV": float(np.sqrt(np.mean(res[~sreg] ** 2)) * 1e6)}
        out[name + "_resid"] = res
        if name == "B":
            theta = th.copy()  # C starts from B
    out["f"], out["f0"] = f, float(theta[NC + 2])
    out["amp"] = float(np.max(np.abs(z["absorb"])))
    return out


def main():
    pilot = yaml.safe_load((OUT / "fits_pilot.yaml").read_text())
    good = [e for e in pilot if e["ref_period_verdict"] == "consistent"]
    good.sort(key=lambda e: e["time"])
    sel = [good[i] for i in np.linspace(0, len(good) - 1, N_EVENTS).astype(int)]
    with Pool(N_WORKERS) as pool:
        res = pool.map(run, sel)
    rows = [{k: v for k, v in r.items() if not (k.endswith("_resid") or k in ("f", "amp", "f0"))} for r in res]
    (OUT / "model_variants.yaml").write_text(yaml.safe_dump(rows, sort_keys=False))

    print(f"{'time':20s} seg  P_pilot |   A: P   s/w  |   B: P   s/w  sigma |   C: P   s/w  sigma   eta    wq")
    for r in rows:
        a, b, c = r["A"], r["B"], r["C"]
        print(f"{r['time']:20s} {r['segment']:3d} {r['P_pilot']:+.3f} | {a['P']:+.3f} {a['rms_signal_uV'] / a['rms_wings_uV']:4.2f} | "
              f"{b['P']:+.3f} {b['rms_signal_uV'] / b['rms_wings_uV']:4.2f} {b['sigma_kHz']:5.1f} | "
              f"{c['P']:+.3f} {c['rms_signal_uV'] / c['rms_wings_uV']:4.2f} {c['sigma_kHz']:5.1f} {c['eta']:.3f} {c['wq']:.4f}")
    for v in "ABC":
        sw = np.median([r[v]["rms_signal_uV"] / r[v]["rms_wings_uV"] for r in rows])
        rs = np.median([r[v]["rms_signal_uV"] for r in rows])
        print(f"variant {v}: median rms signal {rs:.1f} µV, signal/wings {sw:.2f}")
    dBA = np.array([abs(r["B"]["P"]) - abs(r["A"]["P"]) for r in rows])
    dCA = np.array([abs(r["C"]["P"]) - abs(r["A"]["P"]) for r in rows])
    print(f"|P| shift B-A: median {np.median(dBA):+.3f} (range {dBA.min():+.3f} .. {dBA.max():+.3f}); "
          f"C-A: median {np.median(dCA):+.3f} (range {dCA.min():+.3f} .. {dCA.max():+.3f})")

    fig, axes = plt.subplots(1, 3, figsize=(19, 4.8), sharey=True)
    X = np.linspace(-0.3, 0.3, 601)
    for ax, v in zip(axes, "ABC"):
        stack = []
        for r in res:
            s = np.sign(r[v if v != "A" else "A"]["P"])
            y = np.interp(X, r["f"] - r["f0"], r[v + "_resid"] * s / r["amp"]) * 100
            stack.append(y)
            ax.plot(X * 1e3, y, color="0.75", lw=0.4)
        ax.plot(X * 1e3, np.median(stack, axis=0), "r", lw=1.8, label="median")
        sw = np.median([r[v]["rms_signal_uV"] / r[v]["rms_wings_uV"] for r in rows])
        title = {"A": "A: η, wq fixed, no Gaussian", "B": "B: η, wq fixed + Gaussian σ", "C": "C: η, wq, σ free"}[v]
        ax.set_title(f"{title}\nmedian rms signal/wings {sw:.2f}", fontsize=10.5)
        ax.set_xlabel("f − f0 [kHz]")
        ax.grid(alpha=0.3)
        ax.set_ylim(-8, 8)
    axes[0].set_ylabel("residual / signal peak [%]")
    axes[0].legend()
    fig.suptitle(f"Fit residuals of {len(res)} plateau events for three lineshape models", weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "stage2_model_variants.png", dpi=130)


if __name__ == "__main__":
    main()
