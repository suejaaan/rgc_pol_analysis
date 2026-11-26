"""Metrics and plots for P predictions (MC held-out tunes and real lineshape labels).

report(df, out_dir, tag): df needs P (truth or label), P_pred, sigma_pred; optional segment, P_err (label error).
Metrics: bias and rms of P_pred - P (absolute, in units of P), MAE, for all and for |P| >= 0.3 (physics range);
binned in |P|; per segment; pull = (P_pred - P) / sqrt(sigma_pred^2 + P_err^2) -> rms ~ 1 if sigma is calibrated.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

BINS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.65]


def _stats(d):
    r = d["P_pred"] - d["P"]
    sig = np.sqrt(d["sigma_pred"] ** 2 + (d["P_err"] ** 2 if "P_err" in d else 0))
    pull = r / sig
    return {"n": int(len(d)), "bias": float(r.mean()), "rms": float(np.sqrt(np.mean(r**2))),
            "mae": float(r.abs().mean()), "median_sigma_pred": float(d["sigma_pred"].median()),
            "pull_rms": float(np.sqrt(np.mean(pull**2)))}


def report(d, out_dir, tag, title=""):
    d = d.copy()
    d["absP"] = d["P"].abs()
    met = {"all": _stats(d)}
    if (d["absP"] >= 0.3).any():
        met["absP>=0.3"] = _stats(d[d["absP"] >= 0.3])
    met["by_absP"] = {f"{lo:.1f}-{hi:.2f}": _stats(g) for (lo, hi), g in
                      ((b, d[(d["absP"] >= b[0]) & (d["absP"] < b[1])]) for b in zip(BINS[:-1], BINS[1:])) if len(g)}
    if "segment" in d:
        met["by_segment"] = {int(s): _stats(g) for s, g in d.groupby("segment")}
    (out_dir / f"{tag}_metrics.yaml").write_text(yaml.safe_dump(met, sort_keys=False))
    keep = [c for c in ("P", "P_pred", "sigma_pred", "P_err", "segment", "SNR", "time") if c in d]
    d[keep].to_csv(out_dir / f"{tag}_predictions.csv", index=False)
    a = met["all"]
    print(f"[{tag}] n {a['n']}  bias {a['bias']:+.4f}  rms {a['rms']:.4f}  mae {a['mae']:.4f}  pull rms {a['pull_rms']:.2f}"
          + (f"  | |P|>=0.3: bias {met['absP>=0.3']['bias']:+.4f} rms {met['absP>=0.3']['rms']:.4f}" if "absP>=0.3" in met else ""),
          flush=True)

    fig, ax = plt.subplots(1, 3, figsize=(17, 5))
    s = d.sample(min(len(d), 20000), random_state=0)
    ax[0].scatter(s["P"], s["P_pred"], s=2 if len(s) > 2000 else 12, alpha=0.4)
    lim = max(0.7, s["P"].abs().max() * 1.05)
    ax[0].plot([-lim, lim], [-lim, lim], "r--", lw=1)
    ax[0].set_xlabel("true / label P")
    ax[0].set_ylabel("predicted P")
    ax[0].set_title(f"bias {a['bias']:+.4f}, rms {a['rms']:.4f}", fontsize=10)
    ax[0].grid(alpha=0.3)
    cen, bias, rms = [], [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        g = d[(d["absP"] >= lo) & (d["absP"] < hi)]
        if len(g):
            cen.append((lo + hi) / 2)
            r = g["P_pred"] - g["P"]
            bias.append(r.mean())
            rms.append(np.sqrt(np.mean(r**2)))
    ax[1].plot(cen, bias, "o-", label="bias")
    ax[1].plot(cen, rms, "s-", label="rms")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xlabel("|P|")
    ax[1].set_ylabel("P_pred − P")
    ax[1].set_title("Bias and rms vs |P|", fontsize=10)
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    sig = np.sqrt(d["sigma_pred"] ** 2 + (d["P_err"] ** 2 if "P_err" in d else 0))
    pull = (d["P_pred"] - d["P"]) / sig
    ax[2].hist(pull.clip(-6, 6), bins=60, density=True, alpha=0.7)
    xx = np.linspace(-6, 6, 200)
    ax[2].plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), "r", lw=1.5, label="N(0,1)")
    ax[2].set_xlabel("pull = (P_pred − P) / σ")
    ax[2].set_title(f"Uncertainty calibration: pull rms {a['pull_rms']:.2f} (1 = calibrated)", fontsize=10)
    ax[2].legend()
    ax[2].grid(alpha=0.3)
    fig.suptitle(title or tag, weight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / f"{tag}.png", dpi=130)
    plt.close(fig)
    return met


def plot_loss(csv, png):
    h = pd.read_csv(csv)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(h["epoch"], h["train_loss"], label="train")
    ax[0].plot(h["epoch"], h["val_loss"], label="val")
    ax[0].set_ylabel("loss")
    ax[1].plot(h["epoch"], h["train_mae"], label="train")
    ax[1].plot(h["epoch"], h["val_mae"], label="val")
    ax[1].set_ylabel("MAE in P")
    ax[1].set_yscale("log")
    for a in ax:
        a.set_xlabel("epoch")
        a.legend()
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png, dpi=120)
    plt.close(fig)
