"""Sanity check of the lineshape P for one event (default: Mar 1 2023 16:25, online P 0.05, lineshape P 0.38).

1. Model-free: horn depths of the measured chi'' signal (raw - fitted background - fitted chi').
2. Data vs fixed-P fits: P = online value vs P = lineshape value (A, f0, g refitted).
3. Closure: MC through this event's circuit at known P (0.05 and the lineshape P) + its measured noise,
   refitted with free P (same bounds as run_lineshape.py) -> is the true P recovered?
"""
import sys
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares

import buildup_test as BT
import confirm_hypotheses as CH
import confirm_shapes as CS
import run_lineshape as RL
from run_lineshape import M

TIME = sys.argv[1] if len(sys.argv) > 1 else "2023-03-01 16:25:34"
N_SEEDS = 6


def horns(f, f0, y):
    x = f - f0
    s = np.sign(np.sum(y))  # make the signal positive
    yy = y * s
    lo = yy[(x > -0.095) & (x < -0.055)]
    hi = yy[(x > 0.055) & (x < 0.095)]
    centre = np.median(yy[np.abs(x) < 0.02])
    return np.sort(lo)[-3:].mean(), np.sort(hi)[-3:].mean(), centre


def closure(job):
    f, circuit, signal, p_true, sigma, seed = job
    RL.init_worker()
    s = dict(signal, P=p_true)
    truth = [s[n] for n in M.SIGNAL_NAMES]
    raw = M.full_model(f, circuit, truth) + np.random.default_rng(seed).normal(0, sigma, len(f))
    nc = len(M.CIRCUIT_NAMES)
    lo = np.array(list(M.CIRCUIT_LO) + list(M.SIGNAL_LO))
    hi = np.array(list(M.CIRCUIT_HI) + list(M.SIGNAL_HI))
    best = None
    for p0 in (0.0, 0.2, 0.5):  # start P away from the truth on purpose
        x0 = np.clip(np.array(list(circuit) + [s["A"], p0, s["f0"], s["wq"], s["eta_q"], s["g"]]), lo + 1e-9, hi - 1e-9)
        r = least_squares(lambda x: (M.full_model(f, x[:nc], x[nc:]) - raw) * 1e4, x0, bounds=(lo, hi),
                          x_scale="jac", max_nfev=600)
        if best is None or r.cost < best.cost:
            best = r
    return p_true, seed, float(best.x[nc + 1])


def main():
    fits = yaml.safe_load((BT.OUT / "buildup_fits.yaml").read_text())
    e = next(r for r in fits if r["time"] == TIME)
    z = np.load(CS.npz_path(e, BT.OUT))
    f, raw, bg, disp = z["f"], z["raw"], z["background"], z["disperse"]
    f0 = e["signal"]["f0"]
    meas = raw - bg - disp
    hl, hr, hc = horns(f, f0, meas)
    print(f"event {TIME}: online P {e['P_online']:+.3f}, P_area {e['P_area']:+.3f}, P_lineshape {e['P_lineshape']:+.3f}")
    print(f"1) measured horn depths: lower {hl * 1e3:.3f} mV, upper {hr * 1e3:.3f} mV, centre {hc * 1e3:.3f} mV; "
          f"ratio {hr / hl:.2f}; above centre: {(hl - hc) * 1e3:.3f} vs {(hr - hc) * 1e3:.3f} mV; noise {e['noise_sigma_V'] * 1e6:.0f} µV")

    circuit = [e["circuit"][n] for n in M.CIRCUIT_NAMES]
    exp_ratio = {}
    for P in (0.0, abs(e["P_online"]), abs(e["P_lineshape"])):
        s = dict(e["signal"], P=P)
        a = M.absorption_dispersion_parts(f, circuit, [s[n] for n in M.SIGNAL_NAMES])[0]
        l, h, _ = horns(f, f0, a)
        exp_ratio[P] = h / l
    print("   model horn ratio through this circuit: " + ", ".join(f"P={p:.2f}: {v:.2f}" for p, v in exp_ratio.items()))

    fixed = {}
    for kind, P in (("online", e["P_online"]), ("lineshape", e["P_lineshape"])):
        model, _ = CH.fit_fixed_p((f, raw, circuit, e["signal"], P))
        reg = (f > CH.SIGNAL_REGION[0]) & (f < CH.SIGNAL_REGION[1])
        fixed[kind] = (P, model, float(np.sqrt(np.mean((raw - bg - model)[reg] ** 2))))
        print(f"2) fit with P fixed at {P:+.3f} ({kind}): rms {fixed[kind][2] * 1e6:.1f} µV")

    jobs = [(f, circuit, e["signal"], p, e["noise_sigma_V"], s)
            for p in (abs(e["P_online"]), abs(e["P_lineshape"])) for s in range(N_SEEDS)]
    with Pool(RL.N_WORKERS) as pool:
        res = pool.map(closure, jobs)
    for p in (abs(e["P_online"]), abs(e["P_lineshape"])):
        got = [r[2] for r in res if r[0] == p]
        print(f"3) closure: true P {p:.3f} -> fitted {np.mean(got):.3f} ± {np.std(got):.3f}  ({', '.join(f'{g:.3f}' for g in got)})")

    fig, axes = plt.subplots(1, 3, figsize=(20, 5.8))
    x = (f - f0) * 1e3
    ax = axes[0]
    ax.plot(x, meas * 1e3, ".-", color="0.3", ms=3, lw=0.6, label="measured χ″ (raw − background − χ′)")
    ax.axhline(0, color="0.6", lw=0.6)
    ax.set_xlim(-200, 200)
    ax.set_title(f"1) Data {TIME[:16]}: horn ratio {hr / hl:.2f}\n(model: P=0 → {exp_ratio[0.0]:.2f}, "
                 f"P={abs(e['P_online']):.2f} → {exp_ratio[abs(e['P_online'])]:.2f}, "
                 f"P={abs(e['P_lineshape']):.2f} → {exp_ratio[abs(e['P_lineshape'])]:.2f})", fontsize=10.5)
    ax.set_xlabel("f − f0 [kHz]")
    ax.set_ylabel("mV")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    ax = axes[1]
    ax.plot(x, (raw - bg) * 1e3, ".", color="0.35", ms=3, label="data (raw − background)")
    for kind, col in (("online", "tab:red"), ("lineshape", "tab:blue")):
        P, model, rms = fixed[kind]
        ax.plot(x, model * 1e3, color=col, lw=1.6, label=f"fit with P fixed = {P:+.2f}: rms {rms * 1e6:.0f} µV")
    ax.set_xlim(-200, 200)
    ax.set_title("2) Data vs fits at the two P values", fontsize=10.5)
    ax.set_xlabel("f − f0 [kHz]")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    ax = axes[2]
    for p, col in ((abs(e["P_online"]), "tab:red"), (abs(e["P_lineshape"]), "tab:blue")):
        got = [r[2] for r in res if r[0] == p]
        ax.plot([p] * len(got), got, "o", color=col, ms=7, alpha=0.7, label=f"MC true P {p:.2f} → fit {np.mean(got):.3f}")
    ax.plot([0, 0.5], [0, 0.5], "k--", lw=0.8)
    ax.set_xlabel("true P (MC)")
    ax.set_ylabel("fitted P")
    ax.set_title("3) Closure: MC through this event's circuit + its noise, refitted", fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8.5)
    fig.tight_layout()
    out = CS.PLOTS / "lineshape_check_event.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
