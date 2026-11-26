"""Stage 3 report: error budget per label and labels_final.csv (no cc used anywhere).

Terms per label (absolute, on |P|):
  fit        P error from the event fit (covariance, shared circuit fixed)
  consist    spread of P / online area within the label's segment and sign (relative x |P|): label-to-label
             consistency, includes slow drifts of the gain/processing not modelled
  circuit    half the P spread over the best three shared-circuit solutions of the segment (stage 2)
  model      eta, wq, powder grid, g bound variations (stage3_checks.py), per tested segment; other segments
             get the median relative model systematic. The no-Gaussian model is excluded (fit 2x worse).
  closure    rms of (P_fit - P_true) in the MC closure with real residual noise (bias itself < 0.001)
Total = quadrature sum. Sign asymmetry vs the online cc is reported, not added (see README).

Outputs: results/stage3/labels_final.csv, results/stage3/error_budget.yaml, plots/stage3_error_budget.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
R3 = HERE / "results" / "stage3"
MODEL_VARIANTS = (("eta_lo", "eta_hi"), ("wq_lo", "wq_hi"), ("nphi_128",), ("g_lo_1e-4",))


def main():
    d = pd.read_csv(HERE / "results/stage2/labels.csv")
    L = d[d["label_ok"]].copy()
    L["sign"] = np.sign(L["P"]).astype(int)
    L["absP"] = L["P"].abs()

    # consistency: relative spread of P/area within segment and sign
    sp = L.groupby(["segment", "sign"])["ratio"].agg(lambda x: np.std(x) / abs(np.mean(x)))
    L["err_consist"] = [sp[(s, g)] * p for s, g, p in zip(L["segment"], L["sign"], L["absP"])]
    L["err_fit"] = L["P_err_fit"]
    L["err_circuit"] = L["P_circuit_syst"]

    # model systematic per tested segment (relative), median for the others
    v = yaml.safe_load((R3 / "variants.yaml").read_text())
    nom = {o["segment"]: dict(zip(o["times"], np.abs(o["P"]))) for o in v if o["variant"] == "nominal"}
    rel_model = {}
    for seg in nom:
        terms = []
        for group in MODEL_VARIANTS:
            dd = []
            for name in group:
                o = next(x for x in v if x["segment"] == seg and x["variant"] == name)
                dd.append(np.median(np.abs(o["P"]) - np.array([nom[seg][t] for t in o["times"]])))
            terms.append(max(abs(x) for x in dd))
        rel_model[seg] = float(np.sqrt(np.sum(np.square(terms))) / np.median(list(nom[seg].values())))
    rel_med = float(np.median(list(rel_model.values())))
    L["model_tested"] = L["segment"].isin(rel_model)
    L["err_model"] = [rel_model.get(s, rel_med) * p for s, p in zip(L["segment"], L["absP"])]

    c = yaml.safe_load((R3 / "closure.yaml").read_text())
    diff = np.concatenate([np.abs(o["P_fit"]) - np.abs(o["P_true"]) for o in c if o["noise"] == "real"])
    closure_rms, closure_bias = float(np.sqrt(np.mean(diff**2))), float(np.median(diff))
    L["err_closure"] = closure_rms

    comps = ["err_fit", "err_consist", "err_circuit", "err_model", "err_closure"]
    L["err_total"] = np.sqrt(np.sum(L[comps] ** 2, axis=1))
    L["err_total_rel"] = L["err_total"] / L["absP"]
    L["rel_to_online_cc"] = L["ratio"] / L["ref_online_cc"]
    keep = ["time", "file", "index", "segment", "plateau", "P", "err_total", "err_total_rel"] + comps + [
        "model_tested", "A", "U", "phi_deg", "DC_offset", "f0", "g", "sigma_kHz", "area_chi2", "area_online",
        "rms_signal_uV", "rms_wings_uV", "g_at_lower", "ref_online_cc", "ref_online_P", "rel_to_online_cc"]
    L[keep].to_csv(R3 / "labels_final.csv", index=False)

    budget = {
        "n_labels": int(len(L)), "n_pos": int((L["P"] > 0).sum()), "n_neg": int((L["P"] < 0).sum()),
        "absP_range": [float(L["absP"].min()), float(L["absP"].max())],
        "median_abs_error": {k: float(L[k].median()) for k in comps + ["err_total"]},
        "median_rel_total": float(L["err_total_rel"].median()),
        "p90_rel_total": float(L["err_total_rel"].quantile(0.9)),
        "closure": {"bias_median": closure_bias, "rms_real_noise": closure_rms},
        "model_rel_per_tested_segment": {int(k): round(v, 4) for k, v in rel_model.items()},
        "sign_vs_online_cc": {int(s): float(L[L["sign"] == s].groupby("segment")["rel_to_online_cc"].median().median())
                              for s in (1, -1)},
    }
    (R3 / "error_budget.yaml").write_text(yaml.safe_dump(budget, sort_keys=False))
    print(yaml.safe_dump(budget, sort_keys=False))

    # ---- plot ----
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    a = ax[0]
    for o in c:
        col = "tab:blue" if o["noise"] == "real" else "tab:gray"
        a.plot(np.abs(o["P_true"]), np.abs(o["P_fit"]) - np.abs(o["P_true"]), "o", ms=3, color=col, alpha=0.6)
    a.plot([], [], "o", color="tab:blue", label="real residual added")
    a.plot([], [], "o", color="tab:gray", label="white noise")
    a.axhline(0, color="k", lw=0.8)
    a.set_xlabel("true |P| (MC)")
    a.set_ylabel("fitted − true |P|")
    a.set_title(f"MC closure, 6 segments × 4 MC sets: bias {closure_bias:+.4f}, rms {closure_rms:.4f}", fontsize=10.5)
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3)
    a = ax[1]
    names = [n for g in MODEL_VARIANTS for n in g] + ["no_gauss"]
    segs = sorted(nom)
    w = 0.8 / len(segs)
    for i, seg in enumerate(segs):
        vals = []
        for n in names:
            o = next(x for x in v if x["segment"] == seg and x["variant"] == n)
            vals.append(np.median(np.abs(o["P"]) - np.array([nom[seg][t] for t in o["times"]])))
        a.bar(np.arange(len(names)) + i * w - 0.4 + w / 2, vals, w, label=f"seg {seg}")
    a.set_xticks(np.arange(len(names)))
    a.set_xticklabels(names, rotation=30, fontsize=8.5)
    a.axhline(0, color="k", lw=0.8)
    a.set_ylabel("Δ|P| vs nominal model (median per segment)")
    a.set_title("Model variation (no_gauss excluded from the budget: fit ×2 worse)", fontsize=10.5)
    a.legend(fontsize=7.5, ncol=2)
    a.grid(alpha=0.3, axis="y")
    a = ax[2]
    order = np.argsort(L["absP"].to_numpy())
    labels = {"err_fit": "fit", "err_consist": "segment consistency", "err_circuit": "circuit choice",
              "err_model": "model (η, wq, grid, g)", "err_closure": "MC closure", "err_total": "total"}
    for k, lab in labels.items():
        y = (L[k] / L["absP"]).to_numpy()[order] * 100
        a.plot(L["absP"].to_numpy()[order], pd.Series(y).rolling(25, center=True, min_periods=5).median(),
               lw=2.4 if k == "err_total" else 1.4, color="k" if k == "err_total" else None, label=lab)
    a.set_ylim(0, 6)
    a.set_xlabel("|P| label")
    a.set_ylabel("relative error [%] (running median)")
    a.set_title(f"Error budget of {len(L)} labels: total median {budget['median_rel_total'] * 100:.1f} % "
                f"(90 %: {budget['p90_rel_total'] * 100:.1f} %)", fontsize=10.5)
    a.legend(fontsize=8.5, ncol=3, loc="upper center")
    a.grid(alpha=0.3)
    fig.suptitle("Stage 3: validation of the cc-free lineshape labels", weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "stage3_error_budget.png", dpi=140)


if __name__ == "__main__":
    main()
