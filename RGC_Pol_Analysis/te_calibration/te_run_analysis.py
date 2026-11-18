"""Analyse one TE (thermal-equilibrium) run: relaxation curve, equilibrium extrapolation, signal shape.

For a TE run, cc = P_TE / A_eq, with P_TE the spin-1 Boltzmann polarization at the measured
temperature and A_eq the equilibrium NMR area. A_eq can only be read off directly if the run
reaches a plateau; otherwise it has to be extrapolated from the relaxation curve, and the result
depends on the relaxation model (single vs double exponential). This script fits both and shows
which cc values the data can and cannot distinguish.

Usage: python te_run_analysis.py <run-name>      (runs defined in RUNS below)
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

H, KB, NU = 6.62607015e-34, 1.380649e-23, 32.7e6
T_KEY = "TGT:PT12:VaporPressure_T"

RUNS = {
    "2022-09-20": dict(
        files=["2022-09-20_14-47-16__2022-09-20_19-08-04.txt", "2022-09-20_19-38-30__2022-09-20_19-54-22.txt",
               "2022-09-20_19-55-15__2022-09-21_08-55-14.txt"],
        fit_start="2022-09-20 15:45", t_min_k=1.4, cc_online=-1.39,
        cc_test=[-1.39, -0.7, -0.3],
        shape_windows=[("2022-09-20 18:23", "2022-09-20 19:06"), ("2022-09-20 19:39", "2022-09-20 20:09")],
    ),
}


def p_te(T):
    x = H * NU / (KB * T)
    return 2 * np.sinh(x) / (1 + 2 * np.cosh(x))


def load(run):
    ev = []
    for f in run["files"]:
        for line in open(DATA / f):
            d = json.loads(line)
            if d["label"] == "TE":
                ev.append(d)
    return ev


def fit_models(th, y, aeq_for_cc):
    out = {}

    def run(name, f, p0, lo, hi):
        r = least_squares(lambda p: f(p) - y, p0, bounds=(lo, hi))
        out[name] = dict(chi2=float(np.sum(r.fun**2)), npar=len(p0), params=r.x, f=f)

    run("1-exp, A_eq free", lambda p: p[0] + p[1] * np.exp(-th / p[2]), [-1, -15, 1.2], [-50, -100, 0.05], [50, 100, 50])
    run("2-exp, A_eq free", lambda p: p[0] + p[1] * np.exp(-th / p[2]) + p[3] * np.exp(-th / p[4]),
        [-1, -12, 0.8, -8, 5], [-50, -100, 0.05, -100, 0.05], [50, 0, 50, 0, 500])
    for cc, aeq in aeq_for_cc.items():
        run(f"2-exp, A_eq fixed (cc = {cc})", lambda p, a=aeq: a + p[0] * np.exp(-th / p[1]) + p[2] * np.exp(-th / p[3]),
            [-12, 0.8, -8, 5], [-100, 0.05, -100, 0.05], [0, 50, 0, 500])
    return out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "2022-09-20"
    run = RUNS[name]
    ev = load(run)
    t = np.array([datetime.fromisoformat(d["start_time"][:19]).timestamp() for d in ev])
    area = np.array([d["area"] for d in ev]) * 1e3
    T = np.array([d["epics"][T_KEY] for d in ev])
    t0 = datetime.fromisoformat(run["fit_start"]).timestamp()
    sel = (t > t0) & (T > run["t_min_k"]) & (area < -1) & (area > -50)
    th, y = (t[sel] - t0) / 3600, area[sel]
    T_end = float(np.mean(T[sel][-50:]))
    pte = p_te(T_end)
    aeq_for_cc = {cc: pte / cc * 1e3 for cc in run["cc_test"]}
    models = fit_models(th, y, aeq_for_cc)
    ref = models["1-exp, A_eq free"]["chi2"] / (len(y) - 3)

    out = HERE / "results" / name
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "run": name, "n_te_events": len(ev), "n_used": int(sel.sum()), "hours_fitted": float(th[-1]),
        "T_vp_end_K": T_end, "P_TE_percent": float(pte * 100), "cc_online": run["cc_online"],
        "expected_A_eq_online_cc_e3": float(pte / run["cc_online"] * 1e3),
        "models": {k: {"chi2_rel": float(v["chi2"] / (len(y) - v["npar"]) / ref),
                       "params": [float(p) for p in v["params"]]} for k, v in models.items()},
    }
    summary["cc_from_1exp_plateau"] = float(pte / (models["1-exp, A_eq free"]["params"][0] * 1e-3))
    (out / "summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))

    fig, axes = plt.subplots(2, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={"height_ratios": [3, 1.2]})
    tt = (t - t0) / 3600
    ax = axes[0]
    ax.plot(tt, area, ".", color="0.6", ms=3, label="online area, all TE events")
    ax.plot(th, y, ".", color="k", ms=3, label="used in fit")
    grid = np.linspace(0, max(th[-1] * 2.5, 10), 400)
    styles = {"1-exp, A_eq free": ("tab:orange", "-"), "2-exp, A_eq free": ("tab:purple", "-")}
    for k, v in models.items():
        color, ls = styles.get(k, (None, "--"))
        p = v["params"]
        if k.startswith("1-exp"):
            curve = p[0] + p[1] * np.exp(-grid / p[2])
        elif "free" in k:
            curve = p[0] + p[1] * np.exp(-grid / p[2]) + p[3] * np.exp(-grid / p[4])
        else:
            aeq = float(k.split("cc = ")[1].rstrip(")"))
            curve = aeq_for_cc[aeq] + p[0] * np.exp(-grid / p[1]) + p[2] * np.exp(-grid / p[3])
        ax.plot(grid, curve, color=color, ls=ls, lw=1.3,
                label=f"{k}: χ²/dof = {summary['models'][k]['chi2_rel']:.2f} (rel.)")
    for cc, aeq in aeq_for_cc.items():
        ax.axhline(aeq, color="tab:blue" if cc == run["cc_online"] else "0.7", lw=0.8, ls=":")
    ax.axvspan(th[-1], grid[-1], color="0.93", zorder=0)
    ax.set_ylim(min(y.min() * 1.1, -1), 1)
    ax.text(th[-1] + 0.15, 0.55, "extrapolation\n(no data)", fontsize=10, color="0.4",
            transform=ax.get_xaxis_transform())
    ax.set_ylabel("online area ×10³")
    ax.set_title(f"TE run {name}: relaxation after microwaves off (T_vp ≈ {T_end:.2f} K, P_TE = {pte * 100:.3f} %)\n"
                 f"dotted: equilibrium area for cc = {', '.join(str(c) for c in run['cc_test'])} (blue = online cc)")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.plot(tt, T, ".", ms=2, color="tab:red")
    ax.set_ylabel("T_vp [K]")
    ax.set_xlabel(f"hours since {run['fit_start']}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "relaxation_fits.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    sig = yaml.safe_load((HERE.parent / "qmeter_dulya_fit/results/results.yaml").read_text())[250]["signal"]
    fig, axes = plt.subplots(1, len(run["shape_windows"]), figsize=(7.5 * len(run["shape_windows"]), 4.8), squeeze=False)
    for ax, (lo, hi) in zip(axes[0], run["shape_windows"]):
        lo_t, hi_t = datetime.fromisoformat(lo).timestamp(), datetime.fromisoformat(hi).timestamp()
        w = [d for d, ti, a in zip(ev, t, area) if lo_t <= ti <= hi_t and -50 < a < -1]
        f = np.asarray(w[0]["freq_list"])
        mean_fs = np.mean([d["fitsub"] for d in w], axis=0)
        mean_area = np.mean([d["area"] for d in w])
        chi2, _ = M.chi_components(f, 1.0, pte, sig["f0"], sig["wq"], sig["eta_q"], sig["g"])
        tmpl = chi2 / chi2[len(f) // 4: 3 * len(f) // 4].sum()
        ax.plot(f, mean_fs * 1e6, "k", lw=0.9, label=f"mean online fitsub, {len(w)} TE events")
        ax.plot(f, mean_area * tmpl * 1e6, "tab:red", lw=1.3, label=f"Dulya scaled to observed area ({mean_area * 1e3:.2f}e-3)")
        ax.plot(f, pte / run["cc_online"] * tmpl * 1e6, "tab:blue", lw=1.3,
                label=f"Dulya at true TE for cc = {run['cc_online']} ({pte / run['cc_online'] * 1e3:.2f}e-3)")
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_title(f"{lo[11:]}–{hi[11:]}")
        ax.set_xlabel("frequency [MHz]")
        ax.set_ylabel("µV")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    fig.suptitle("Late-TE signal is a real Pake doublet, ~10× larger than equilibrium for the online cc → still relaxing", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "late_te_signal_shape.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    print(yaml.safe_dump({k: v for k, v in summary.items() if k != "models"}, sort_keys=False))
    for k, v in summary["models"].items():
        print(f"  {k:32s} χ²/dof rel = {v['chi2_rel']:.2f}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
