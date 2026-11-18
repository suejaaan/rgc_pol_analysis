"""Problem periods: does the measured signal look like P_online or like P_lineshape?

For each problem event, two Monte Carlo signals are generated from the event's own fit: the fitted Q-meter
circuit (background) and eta, wq are kept, P is fixed to the hypothesis (P_online or P_lineshape) and only the
amplitude A, f0 and g are refitted. Signals are compared to the data in volts (no normalization), with residuals.
"""
from multiprocessing import Pool

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.gridspec import GridSpec
from scipy.optimize import least_squares

import compare_methods as CM
import confirm_shapes as CS
import run_lineshape as RL
from run_lineshape import M

RES, PLOTS = CS.RES, CS.PLOTS
SIGNAL_REGION = (32.48, 32.90)


def fit_fixed_p(job):
    """Refit A, f0, g with P fixed; circuit, eta, wq from the free fit."""
    f, raw, circuit, signal, P = job
    s0 = dict(signal)
    p_fixed = float(np.clip(P, -0.95, 0.95))

    def sig(x):
        return [x[0], p_fixed, x[1], s0["wq"], s0["eta_q"], x[2]]

    x0 = [s0["A"], s0["f0"], s0["g"]]
    r = least_squares(lambda x: (M.full_model(f, circuit, sig(x)) - raw) * 1e4, x0,
                      bounds=([-np.inf, 32.60, 0.001], [np.inf, 32.77, 0.08]), x_scale="jac", max_nfev=300)
    return M.full_model(f, circuit, sig(r.x)) - M.qmeter(f, *circuit), dict(zip(("A", "f0", "g"), map(float, r.x)))


def main():
    events, by = CM.load()
    rows = CM.period_rows(by)
    problems = [r for r in rows if r["status"] in ("online wrong", "unresolved", "no result")]
    plan = []
    for r in problems:
        rs = [e for e in by[r["cc"]] if e["use"]] or by[r["cc"]]
        plan.append((r, max(rs, key=lambda e: abs(e["area_online"]))))

    jobs, meta = [], []
    for r, e in plan:
        z = np.load(CS.npz_path(e, RES))
        circuit = [e["circuit"][n] for n in M.CIRCUIT_NAMES]
        for kind, P in (("online", e["P_online"]), ("lineshape", e["P_lineshape"])):
            jobs.append((z["f"], z["raw"], circuit, e["signal"], P))
            meta.append((r["cc"], kind))
    with Pool(RL.N_WORKERS) as p:
        out = dict(zip(meta, p.map(fit_fixed_p, jobs)))

    summary = []
    for group, title, fname in (
            (("online wrong",), "Online cc inconsistent", "hypotheses_online_wrong.png"),
            (("unresolved", "no result"), "Unresolved", "hypotheses_unresolved.png")):
        sel = [p for p in plan if p[0]["status"] in group]
        ncol = min(3, len(sel))
        nrow = int(np.ceil(len(sel) / ncol))
        fig = plt.figure(figsize=(6.6 * ncol, 6.6 * nrow))
        outer = GridSpec(nrow, ncol, figure=fig, hspace=0.28, wspace=0.28)
        for k, (r, e) in enumerate(sel):
            row, col = divmod(k, ncol)
            inner = outer[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.06)
            ax = fig.add_subplot(inner[0])
            axr = fig.add_subplot(inner[1], sharex=ax)
            z = np.load(CS.npz_path(e, RES))
            f, raw, bg = z["f"], z["raw"], z["background"]
            x = (f - e["signal"]["f0"]) * 1e3
            data = (raw - bg) * 1e3
            reg = (f > SIGNAL_REGION[0]) & (f < SIGNAL_REGION[1])
            ax.plot(x, data, ".", color="0.35", ms=3, label=f"data {e['time'][:16]} ({e['channel'].replace('Deuteron', '')})")
            ent = {"period_cc": r["cc"], "event": e["time"], "P_online": e["P_online"], "P_lineshape": e["P_lineshape"]}
            for kind, P, color in (("online", e["P_online"], "tab:red"), ("lineshape", e["P_lineshape"], "tab:blue")):
                model, pars = out[(r["cc"], kind)]
                resid = (raw - bg - model) * 1e3
                rms = float(np.sqrt(np.mean(resid[reg] ** 2)) * 1e3)  # µV, signal region
                cc_txt = f"cc {e['cc']:g}" if kind == "online" else f"cc_ls {e['P_lineshape'] / e['area_online']:+.2f}"
                ax.plot(x, model * 1e3, "-", color=color, lw=1.8,
                        label=f"MC at {kind} P = {P:+.2f} ({cc_txt}),  rms {rms:.0f} µV")
                axr.plot(x, resid, "-", color=color, lw=1)
                ent[f"rms_uV_{kind}"] = rms
            summary.append(ent)
            ax.axhline(0, color="0.6", lw=0.6)
            axr.axhline(0, color="0.6", lw=0.6)
            ax.set_xlim(-330, 330)
            ax.set_title(f"cc {r['cc']:g}  ({r['dates']})  — {r['status']}", fontsize=11, weight="bold")
            ax.set_ylabel("signal  (raw − fitted background)  [mV]")
            ax.legend(fontsize=7.8, loc="upper left")
            ax.grid(alpha=0.3)
            ax.tick_params(labelbottom=False)
            axr.set_ylabel("data − MC [mV]")
            axr.set_xlabel("f − f0  [kHz]")
            axr.grid(alpha=0.3)
        fig.suptitle(f"{title}: MC from each event's own fit at the online P (red) and the lineshape P (blue)",
                     fontsize=13, weight="bold", y=0.995)
        fig.text(0.01, 0.0, "Each event: fitted Q-meter circuit, η, wq kept; P fixed to the hypothesis; amplitude A, f0, g "
                 "refitted. No normalization: amplitude is whatever fits the data best at that P. rms over 32.48–32.90 MHz.",
                 fontsize=8.5, color="0.3")
        fig.savefig(PLOTS / fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {PLOTS / fname}")
    (RES / "hypotheses_rms.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    for s in summary:
        print(f"  cc {s['period_cc']:9.4f}  P_on {s['P_online']:+.3f} rms {s['rms_uV_online']:6.0f} µV | "
              f"P_ls {s['P_lineshape']:+.3f} rms {s['rms_uV_lineshape']:6.0f} µV")


if __name__ == "__main__":
    main()
