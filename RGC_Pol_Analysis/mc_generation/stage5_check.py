"""Stage 5 check: does the MC look like real data?

Twin test: every stage-3 label event (real raw sweep) gets an MC twin made by the generator from the SAME
library record (circuit, P_fit, f0, g, sigma, k_seg; no jitter) but with a DIFFERENT real noise sample of the
same segment (sign flipped at random). Then:
  1. twin difference: rms(real - twin) / (sqrt(2) * noise rms)  -> ~1 if the MC reproduces everything but noise
  2. model-free features (online-style processing: cubic polynomial on the wings, sum over the signal region):
     wing rms, wing lag-1 autocorrelation, high-frequency noise, signal peak, area, horn ratio, roughness in the
     signal region, background shape (cubic coefficients)
  3. classifier: L2 logistic regression on the standardised features, 5-fold cross-validated AUC, real vs twin
     (0.5 = indistinguishable)
Also compares the population of the MC batch (rows with |P| in the label range) with the real labels.

Usage: python stage5_check.py [MC_NAME]     (default test10k)
Output: plots/stage5_check.png, mc_data/<MC_NAME>/check.yaml
"""
import glob
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import minimize

import stage5_generate as G

HERE = Path(__file__).resolve().parent
LD = HERE.parent / "label_dataset"
SREG = (32.48, 32.90)
FEATS = ["wing_rms", "wing_ac1", "hf_noise", "peak", "area", "horn_ratio", "roughness", "bg_c1", "bg_c2", "bg_c3"]


def features(f, y, f0=32.685):
    wing = (f < SREG[0]) | (f > SREG[1])
    x = f - f0
    c = np.polyfit(x[wing], y[wing], 3)
    sub = y - np.polyval(c, x)
    w = sub[wing]
    reg = sub[~wing]
    s = np.sign(np.sum(reg)) or 1.0
    yy = sub * s
    lo = np.sort(yy[(x > -0.095) & (x < -0.055)])[-3:].mean()
    hi = np.sort(yy[(x > 0.055) & (x < 0.095)])[-3:].mean()
    k = 7
    smooth = np.convolve(reg, np.ones(k) / k, mode="same")
    return {
        "wing_rms": np.sqrt(np.mean(w**2)), "wing_ac1": np.corrcoef(w[:-1], w[1:])[0, 1],
        "hf_noise": np.std(np.diff(w)) / np.sqrt(2), "peak": np.max(np.abs(reg)), "area": np.sum(reg),
        "horn_ratio": max(lo, hi) / max(min(lo, hi), 1e-9), "roughness": np.sqrt(np.mean((reg - smooth)[k:-k] ** 2)),
        "bg_c1": c[2], "bg_c2": c[1], "bg_c3": c[0],
    }, sub


def auc(score, label):
    o = np.argsort(score)
    r = np.empty(len(score))
    r[o] = np.arange(1, len(score) + 1)
    n1 = label.sum()
    n0 = len(label) - n1
    return (r[label == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1)


def logreg_auc(X, y, folds=5, lam=1e-2, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(y))
    out = np.empty(len(y))
    for k in range(folds):
        te = idx[k::folds]
        tr = np.setdiff1d(idx, te)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-12
        A, B = (X[tr] - mu) / sd, (X[te] - mu) / sd
        A1, B1 = np.c_[A, np.ones(len(A))], np.c_[B, np.ones(len(B))]

        def loss(w):
            z = A1 @ w
            return np.sum(np.logaddexp(0, z) - y[tr] * z) + lam * np.sum(w[:-1] ** 2)

        def grad(w):
            p = 1 / (1 + np.exp(-(A1 @ w)))
            g = A1.T @ (p - y[tr])
            g[:-1] += 2 * lam * w[:-1]
            return g

        w = minimize(loss, np.zeros(A1.shape[1]), jac=grad, method="L-BFGS-B").x
        out[te] = B1 @ w
    return auc(out, y), out


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "test10k"
    L = G.load_library()
    f = L["freq"][0]
    rng = np.random.default_rng(3)
    lab = np.flatnonzero(L["quality"] == "label")
    real_feat, twin_feat, ratio, ex = [], [], [], []
    for r in lab:
        t, seg = str(L["time"][r]), L["segment"][r]
        fl = pd.read_csv(LD / "results/stage3/labels_final.csv", usecols=["time", "file", "index"]) if r == lab[0] else fl
        row = fl[fl["time"] == t].iloc[0]
        z = np.load(LD / f"results/stage2/npz_full/event_{row['file'][:19]}_{row['index']}.npz")
        raw = z["raw"]
        clean, *_ = G.make_sweep(f, L["circuit"][r], L["P_fit"][r], L["f0"][r], L["g"][r], L["sigma"][r],
                                 L["k_seg"][r], L["template_x"], L["template"])
        nz, _, _ = G.draw_noise(rng, L, seg, exclude=r)
        twin = clean + nz
        noise_ref = np.sqrt(np.mean(L["noise"][r] ** 2))
        ratio.append(np.sqrt(np.mean((raw - twin) ** 2)) / (np.sqrt(2) * noise_ref))
        fr, sr = features(f, raw, L["f0"][r])
        ft, st = features(f, twin, L["f0"][r])
        real_feat.append(fr)
        twin_feat.append(ft)
        if len(ex) < 3 and r in lab[[0, len(lab) // 2, -1]]:
            ex.append((t, L["P_fit"][r], sr, st))
    R, T = pd.DataFrame(real_feat), pd.DataFrame(twin_feat)
    ratio = np.array(ratio)
    X = np.vstack([R[FEATS].to_numpy(), T[FEATS].to_numpy()])
    y = np.r_[np.ones(len(R)), np.zeros(len(T))]
    a_all, _ = logreg_auc(X, y)
    per_feat = {k: float(logreg_auc(X[:, [i]], y)[0]) for i, k in enumerate(FEATS)}

    # population: MC batch rows with |P| in the label range vs real labels
    files = sorted(glob.glob(str(HERE / "mc_data" / name / "*.parquet")))
    mc = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    lo, hi = np.abs(L["P_fit"][lab]).min(), np.abs(L["P_fit"][lab]).max()
    mc = mc[(mc["P"].abs() >= lo) & (mc["P"].abs() <= hi)].sample(min(3000, len(mc)), random_state=0)
    cols = [str(i) for i in range(len(f))]
    M_feat = pd.DataFrame([features(f, row, f0)[0] for row, f0 in zip(mc[cols].to_numpy(), mc["f0"].to_numpy())])

    res = {
        "n_twins": int(len(lab)),
        "twin_diff_over_sqrt2_noise_median": float(np.median(ratio)),
        "twin_diff_over_sqrt2_noise_q90": float(np.quantile(ratio, 0.9)),
        "classifier_auc_all_features": float(a_all),
        "classifier_auc_single_feature": {k: round(v, 3) for k, v in per_feat.items()},
        "feature_median_real_vs_twin": {k: [float(R[k].median()), float(T[k].median())] for k in FEATS},
        "feature_median_real_vs_mc_population": {k: [float(R[k].median()), float(M_feat[k].median())] for k in FEATS},
    }
    (HERE / "mc_data" / name / "check.yaml").write_text(yaml.safe_dump(res, sort_keys=False))
    print(yaml.safe_dump(res, sort_keys=False))

    fig = plt.figure(figsize=(20, 11))
    gs = fig.add_gridspec(2, 4)
    for j, (t, P, sr, st) in enumerate(ex):
        a = fig.add_subplot(gs[0, j])
        a.plot((f - 32.685) * 1e3, sr * 1e3, color="0.3", lw=1, label="real (raw − cubic on wings)")
        a.plot((f - 32.685) * 1e3, st * 1e3, color="tab:red", lw=0.9, alpha=0.8, label="MC twin")
        a.set_xlim(-300, 300)
        a.set_title(f"{t}  P = {P:+.3f}", fontsize=10)
        a.set_xlabel("f − 32.685 MHz [kHz]")
        a.set_ylabel("mV")
        a.grid(alpha=0.3)
        a.legend(fontsize=8)
    a = fig.add_subplot(gs[0, 3])
    a.hist(ratio, bins=40, color="tab:purple")
    a.axvline(1, color="k", ls="--")
    a.set_xlabel("rms(real − twin) / (√2 · noise rms)")
    a.set_title(f"Twin difference: median {np.median(ratio):.2f} (1 = only noise differs)", fontsize=10)
    a.grid(alpha=0.3)
    for j, k in enumerate(["wing_rms", "wing_ac1", "roughness", "horn_ratio"]):
        a = fig.add_subplot(gs[1, j])
        v = np.r_[R[k], T[k], M_feat[k]]
        bins = np.linspace(np.quantile(v, 0.01), np.quantile(v, 0.99), 40)
        a.hist(R[k], bins=bins, histtype="step", lw=2, color="k", density=True, label="real labels")
        a.hist(T[k], bins=bins, histtype="step", lw=1.5, color="tab:red", density=True, label="MC twins")
        a.hist(M_feat[k], bins=bins, histtype="step", lw=1.2, color="tab:blue", density=True, label=f"MC {name} (|P| matched)")
        a.set_title(f"{k}: single-feature AUC {per_feat[k]:.2f}", fontsize=10)
        a.grid(alpha=0.3)
        a.legend(fontsize=7.5)
    fig.suptitle(f"Stage 5 check: real vs MC — classifier AUC (all features, real vs twin) = {a_all:.2f} (0.5 = indistinguishable)",
                 weight="bold")
    fig.tight_layout()
    fig.savefig(HERE / "plots" / "stage5_check.png", dpi=130)


if __name__ == "__main__":
    main()
