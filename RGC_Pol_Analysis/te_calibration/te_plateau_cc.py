"""Calibrate cc from the one data-d TE run that (nearly) reaches equilibrium: 2023-03-21.

1. Average the online `fitsub` sweeps in 1-hour bins (microwaves off, label TE).
2. Measure each bin's signal with a shape-weighted area: Dulya doublet (P ~ 0, shape fitted on the
   strong early bins, then fixed) + quadratic background. A smooth background drift cannot mimic
   the narrow horns, so this is more robust at TE level than summing bins.
3. Fit the late relaxation (after the fast component) with A(t) = A_eq + B exp(-t / tau) and
   extrapolate A_eq. Compare with the last bins directly.
4. cc_TE = P_TE(T_vp) / A_eq. Systematics: thermometer (vapor pressure vs bath resistors),
   tail start time, area method (shape-weighted vs online sum).
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
DATA = HERE.parents[1] / "data_creation/data-d"
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import model as M  # noqa: E402

FILES = ["current_2023-03-21_14-13-19.txt", "2023-03-21_18-51-38__2023-03-22_03-05-10.txt",
         "2023-03-22_03-05-24__2023-03-22_14-40-07.txt"]
OUT = HERE / "results" / "2023-03-21"
H, KB, NU = 6.62607015e-34, 1.380649e-23, 32.7e6
BIN_H = 1.0
SHAPE_FROM_H = (0.5, 4.0)
TAIL_STARTS_H = [6.0, 8.0, 10.0, 12.0]
GLITCH = 0.02


def p_te(T):
    x = H * NU / (KB * T)
    return 2 * np.sinh(x) / (1 + 2 * np.cosh(x))


def load():
    ev = []
    for f in FILES:
        for line in open(DATA / f):
            d = json.loads(line)
            if d["label"] == "TE" and abs(d["area"]) < GLITCH and not d.get("uwave_power"):
                e = d["epics"]
                ev.append(dict(t=datetime.fromisoformat(d["start_time"][:19]), fitsub=np.asarray(d["fitsub"]),
                               area=d["area"], cc=d["cc"], f=np.asarray(d["freq_list"]),
                               T_vp=e["TGT:PT12:VaporPressure_T"],
                               T_bath=0.5 * (e["TGT:PT12:Bath_Top_T"] + e["TGT:PT12:Bath_Bottom_T"])))
    ev.sort(key=lambda e: e["t"])
    seen, out = set(), []
    for e in ev:  # current_* file overlaps the run file at the start
        if e["t"] not in seen:
            seen.add(e["t"])
            out.append(e)
    return out


def template(f, f0, wq, eta, g):
    chi2, _ = M.chi_components(f, 1.0, 0.0, f0, wq, eta, g)
    return chi2 / chi2.sum()


def fit_bin(f, y, shape, free_shape=False):
    x = (f - f.mean()) / np.ptp(f)

    def model(p):
        s = shape if not free_shape else p[4:]
        return p[0] * template(f, *s) + p[1] + p[2] * x + p[3] * x**2

    p0 = [y.sum(), 0, 0, 0] + (list(shape) if free_shape else [])
    lo = [-np.inf] * 4 + ([32.6, 0.05, 0.0, 0.002] if free_shape else [])
    hi = [np.inf] * 4 + ([32.77, 0.15, 0.4, 0.08] if free_shape else [])
    r = least_squares(lambda p: (model(p) - y) * 1e6, p0, bounds=(lo, hi), x_scale="jac")
    J = r.jac / 1e6
    s2 = np.sum((r.fun / 1e6) ** 2) / (len(y) - len(p0))
    err = np.sqrt(np.diag(np.linalg.pinv(J.T @ J)) * s2)
    return r.x, err, model(r.x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ev = load()
    t0 = ev[0]["t"]
    hrs = np.array([(e["t"] - t0).total_seconds() / 3600 for e in ev])
    f = ev[0]["f"]
    edges = np.arange(0, hrs.max() + BIN_H, BIN_H)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = [e for e, h in zip(ev, hrs) if lo <= h < hi]
        if len(sel) >= 5:
            bins.append(dict(t=0.5 * (lo + hi), n=len(sel), y=np.mean([e["fitsub"] for e in sel], axis=0),
                             area_online=np.mean([e["area"] for e in sel]),
                             T_vp=np.mean([e["T_vp"] for e in sel]), T_bath=np.mean([e["T_bath"] for e in sel])))

    sig250 = yaml.safe_load((HERE.parent / "qmeter_dulya_fit/results/results.yaml").read_text())[250]["signal"]
    shape0 = [sig250["f0"], sig250["wq"], sig250["eta_q"], sig250["g"]]
    strong = [b for b in bins if SHAPE_FROM_H[0] <= b["t"] <= SHAPE_FROM_H[1]]
    p, _, _ = fit_bin(f, np.mean([b["y"] for b in strong], axis=0), shape0, free_shape=True)
    shape = list(p[4:])

    for b in bins:
        p, e, m = fit_bin(f, b["y"], shape)
        b["area_shape"], b["area_shape_err"], b["model"] = p[0], e[0], m

    t = np.array([b["t"] for b in bins])
    a = np.array([b["area_shape"] for b in bins])
    ae = np.array([b["area_shape_err"] for b in bins])
    ao = np.array([b["area_online"] for b in bins])
    T_end = float(np.mean([b["T_vp"] for b in bins[-3:]]))
    T_end_bath = float(np.mean([b["T_bath"] for b in bins[-3:]]))
    pte, pte_bath = p_te(T_end), p_te(T_end_bath)

    def tail_fit(y, ye, start):
        m = t >= start
        r = least_squares(lambda q: (q[0] + q[1] * np.exp(-(t[m] - start) / q[2]) - y[m]) / ye[m],
                          [y[m][-1], y[m][0] - y[m][-1], 4.0], bounds=([-1, -1, 0.2], [1, 1, 200]))
        J = r.jac
        cov = np.linalg.pinv(J.T @ J) * max(np.sum(r.fun**2) / (m.sum() - 3), 1)
        return r.x, np.sqrt(np.diag(cov))

    tails = {s: tail_fit(a, ae, s) for s in TAIL_STARTS_H}
    tails_online = {s: tail_fit(ao, np.full_like(ao, np.std(np.diff(ao[-6:])) + 1e-7), s) for s in TAIL_STARTS_H}
    aeq = np.array([tails[s][0][0] for s in TAIL_STARTS_H])
    aeq_err = np.array([tails[s][1][0] for s in TAIL_STARTS_H])
    best = TAIL_STARTS_H.index(8.0)
    A_eq, A_eq_stat = aeq[best], aeq_err[best]
    A_eq_sys_tail = float(np.ptp(aeq) / 2)
    A_eq_online = tails_online[8.0][0][0]
    A_last = float(np.mean(a[-3:]))

    cc = pte / A_eq
    rel = np.sqrt((A_eq_stat / A_eq) ** 2 + (A_eq_sys_tail / A_eq) ** 2)
    res = {
        "run": "2023-03-21 TE (microwaves off)", "n_events": len(ev), "hours": float(hrs.max()),
        "shape_params_f0_wq_eta_g": [float(v) for v in shape],
        "T_vp_end_K": T_end, "T_bath_end_K": T_end_bath, "P_TE_percent_vp": float(pte * 100),
        "A_eq_shape_e3": float(A_eq * 1e3), "A_eq_stat_e3": float(A_eq_stat * 1e3),
        "A_eq_tail_start_spread_e3": float(A_eq_sys_tail * 1e3),
        "A_eq_by_tail_start_e3": {float(s): float(v * 1e3) for s, v in zip(TAIL_STARTS_H, aeq)},
        "tau_slow_h": float(tails[8.0][0][2]),
        "A_last_3h_shape_e3": A_last * 1e3, "A_eq_online_sum_e3": float(A_eq_online * 1e3),
        "cc_TE": float(cc),
        "cc_TE_err_fit": float(abs(cc) * rel),
        "cc_TE_err_thermometer": float(abs(cc - pte_bath / A_eq)),
        "cc_TE_err_area_method": float(abs(cc - pte / A_eq_online)),
        "cc_TE_if_last_bins_were_equilibrium": float(pte / A_last),
        "cc_online_before": -3.67, "cc_online_after": -0.816887,
    }
    res["cc_TE_err_total"] = float(np.sqrt(res["cc_TE_err_fit"] ** 2 + res["cc_TE_err_thermometer"] ** 2
                                           + res["cc_TE_err_area_method"] ** 2))
    (OUT / "cc_result.yaml").write_text(yaml.safe_dump(res, sort_keys=False))

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.2))
    ax = axes[0]
    ax.errorbar(t, a * 1e3, yerr=ae * 1e3, fmt="o", ms=4, color="k", label="shape-weighted area (1 h bins)")
    ax.plot(t, ao * 1e3, "s", ms=3, color="0.6", label="online area (bin mean)")
    q = tails[8.0][0]
    tt = np.linspace(8.0, t[-1] + 20, 300)
    ax.plot(tt, (q[0] + q[1] * np.exp(-(tt - 8.0) / q[2])) * 1e3, "tab:red", lw=1.5,
            label=f"tail fit from 8 h: A_eq = {A_eq * 1e3:.3f} ± {A_eq_stat * 1e3:.3f} e-3, τ = {q[2]:.1f} h")
    ax.axhline(A_eq * 1e3, color="tab:red", ls=":", lw=1)
    ax.axhline(pte / -3.67 * 1e3, color="tab:purple", ls="--", lw=1, label="equilibrium if cc = −3.67 (online, before)")
    ax.axhline(pte / -0.816887 * 1e3, color="tab:blue", ls="--", lw=1, label="equilibrium if cc = −0.817 (online, after)")
    ax.set_ylim(-3.2, 0)
    ax.axvspan(t[-1], tt[-1], color="0.93", zorder=0)
    ax.set_xlabel("hours since TE start (2023-03-21 14:44)")
    ax.set_ylabel("area ×10³")
    ax.set_title(f"Relaxation to TE at {T_end:.2f} K (P_TE = {pte * 100:.3f} %)")
    ax.legend(fontsize=7.5, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    for b, c in [(bins[2], "tab:orange"), (bins[len(bins) // 2], "tab:green"), (bins[-1], "k")]:
        ax.plot(f, b["y"] * 1e6, color=c, lw=0.8, alpha=0.8, label=f"{b['t']:.1f} h: mean fitsub ({b['n']} ev)")
        ax.plot(f, b["model"] * 1e6, color=c, lw=1.6, ls="--")
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xlabel("frequency [MHz]")
    ax.set_ylabel("µV")
    ax.set_title("Hour-averaged TE sweeps (solid) and Dulya + quadratic bg fit (dashed)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    labels = ["online\nbefore", "online\nafter", "this TE\nanalysis"]
    vals = [-3.67, -0.816887, cc]
    errs = [0, 0, res["cc_TE_err_total"]]
    ax.bar(labels, vals, yerr=errs, color=["tab:purple", "tab:blue", "tab:red"], capsize=6, alpha=0.8)
    for i, v in enumerate(vals):
        ax.text(i, v - 0.12, f"{v:.3f}", ha="center", va="top", fontsize=10)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_ylabel("cc")
    ax.set_title("Calibration constant for this period")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(OUT / "te_2023-03-21_calibration.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(yaml.safe_dump(res, sort_keys=False))


if __name__ == "__main__":
    main()
