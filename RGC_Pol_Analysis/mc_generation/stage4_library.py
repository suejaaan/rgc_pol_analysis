"""Stage 4: parameter library for the Monte Carlo (no cc anywhere).

Unit of the library = one fitted real event (stage 2, segment-joint circuit). Everything that is correlated is
kept together and must be drawn together:
  circuit (9: shared Cknob, trim, L0, Rcoil, alpha, Cstray + own U, phi, DC offset), sigma (segment),
  g and f0 (event), k_seg = A (r+1) / P of the segment (amplitude per unit polarization, cc-free),
  noise = the event's own fit residual minus the signal-correlated misfit template (see below).
The circuit is degenerate (stage 2) and k_seg differs 70x between segments because it belongs to the
segment's circuit solution: never mix circuit, sigma and k from different segments.

Misfit template: median fit residual of the labels in the signal frame (x = (f - f0) * sign(P), signal made
positive), in units of the absorption peak. About half of the residual power at the horns is this repeatable
shape; the MC adds template * peak (mirrored with the sign of P) to the model, the rest comes from the noise.

Quality classes per record:
  label    stage-3 label (|P| >= 0.3, consistent segment), all parameters trusted
  circuit  fit ok, P not a label (cc -14.5 segments, |P| < 0.3): circuit and noise usable, k/g less certain
Stored baseline refits are NOT used: they are a different circuit family (the joint fits never chose them).

Outputs: library/param_library.npz, library/library_summary.yaml, plots/stage4_library.png
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
LD = HERE.parent / "label_dataset"
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
sys.path.insert(0, str(LD))
import model as M  # noqa: E402
import segment_fit as SF  # noqa: E402

LIB, PLOTS = HERE / "library", HERE / "plots"
X = np.linspace(-0.30, 0.30, 769)          # signal-frame grid for the template [MHz]
FAIL_SIGMA_KHZ = 29.9                      # new segment failed: sigma ran to its 30 kHz bound
FAIL_WINGS_UV = 40.0                       # ... or background (wing) residual far above the ~10 uV of good segments
EXTRA_RMS_FACTOR = 1.5                     # extra record kept if its rms <= 1.5 x its segment's median


def main():
    LIB.mkdir(exist_ok=True)
    PLOTS.mkdir(exist_ok=True)
    d = pd.read_csv(LD / "results/stage2/labels.csv")
    d = d[d["fit_ok"]].sort_values("time").reset_index(drop=True)
    final = pd.read_csv(LD / "results/stage3/labels_final.csv")
    d["is_label"] = d["time"].isin(set(final["time"]))
    segs = {s["segment"]: s["solutions"][0] for s in yaml.safe_load((LD / "results/stage2/segments.yaml").read_text())}

    # amplitude per unit polarization, per segment (labels if available, else all fitted events)
    d["k"] = d["A"] * (M.p_to_r(d["P"]) + 1) / d["P"]
    k_seg = {}
    for s, g in d.groupby("segment"):
        gg = g[g["is_label"]] if g["is_label"].any() else g
        k_seg[s] = (float(gg["k"].median()), float(gg["k"].std() / abs(gg["k"].mean())))

    # load per-event arrays, residual in the signal frame
    F, RES, PK, R_sig = [], [], [], []
    for r in d.itertuples():
        z = np.load(LD / f"results/stage2/npz_full/event_{r.file[:19]}_{r.index}.npz")
        F.append(z["f"])
        RES.append(z["resid"])
        pk = float(np.abs(z["absorb"]).max())
        PK.append(pk)
        x = (z["f"] - r.f0) * np.sign(r.P)
        s = np.sign(np.sum(z["absorb"]))
        o = np.argsort(x)
        R_sig.append(np.interp(X, x[o], (z["resid"] * s / pk)[o], left=np.nan, right=np.nan))
    F, RES, PK, R_sig = np.array(F), np.array(RES), np.array(PK), np.array(R_sig)
    lab = d["is_label"].to_numpy()
    template = np.nanmedian(R_sig[lab], axis=0)
    template[np.isnan(template)] = 0.0
    # templates in |P| bins (to check P-dependence)
    tbins = {}
    for lo, hi in ((0.30, 0.36), (0.36, 0.42), (0.42, 0.56)):
        m = lab & (d["P"].abs() >= lo).to_numpy() & (d["P"].abs() < hi).to_numpy()
        tbins[f"{lo:.2f}-{hi:.2f}"] = np.nanmedian(R_sig[m], axis=0)

    # noise = residual - template mapped back to the event's frequency grid
    NOISE = np.empty_like(RES)
    expl = []
    for i, r in enumerate(d.itertuples()):
        x = (F[i] - r.f0) * np.sign(r.P)
        s = np.sign(np.sum(np.load(LD / f"results/stage2/npz_full/event_{r.file[:19]}_{r.index}.npz")["absorb"]))
        tmpl_ev = np.interp(x, X, template) * PK[i] * s
        NOISE[i] = RES[i] - tmpl_ev
        reg = np.abs(F[i] - r.f0) < 0.12
        expl.append(1 - np.mean(NOISE[i][reg] ** 2) / np.mean(RES[i][reg] ** 2))
    expl = np.array(expl)

    circuit = np.zeros((len(d), 9))
    for i, r in enumerate(d.itertuples()):
        sh = segs[r.segment]["shared"]
        circuit[i, SF.SHARED_C] = [sh[M.CIRCUIT_NAMES[k]] for k in SF.SHARED_C]
        circuit[i, SF.OWN_C] = [r.U, r.phi_deg, r.DC_offset]
    sigma = np.array([segs[s]["shared"]["sigma_kHz"] / 1e3 for s in d["segment"]])
    klib = np.array([k_seg[s][0] for s in d["segment"]])
    quality = np.where(lab, "label", "circuit")
    seg_of = d["segment"].to_numpy()
    g_arr, f0_arr, A_arr, P_arr = d["g"].to_numpy(), d["f0"].to_numpy(), d["A"].to_numpy(), d["P"].to_numpy()
    time_arr = d["time"].to_numpy().astype(str)
    P_err = final.set_index("time").reindex(d["time"])["err_total"].to_numpy()

    # ---- extension (stage4_extend.py): new tune segments + extra records with the shared circuit fixed ----
    n_new_seg = 0
    if (LIB / "extra_records.csv").exists():
        new = yaml.safe_load((LIB / "segments_extra.yaml").read_text())
        # failed new segments: sigma at its upper bound or poor background (Jun-Jul 2022 hardware state)
        failed = {s["segment"] for s in new
                  if s["solutions"][0]["shared"]["sigma_kHz"] >= FAIL_SIGMA_KHZ or s["median_rms_wings_uV"] > FAIL_WINGS_UV}
        new = [s for s in new if s["segment"] not in failed]
        n_new_seg = len(new)
        ref = d.groupby("segment")[["rms_signal_uV", "rms_wings_uV"]].median()
        for s in new:
            b = s["solutions"][0]
            segs[s["segment"]] = b
            kk = np.array([e["A"] * (M.p_to_r(e["P"]) + 1) / e["P"] for e in b["events"] if abs(e["P"]) > 0.02])
            k_seg[s["segment"]] = (float(np.median(kk)), float(np.std(kk) / abs(np.mean(kk))))
            ref.loc[s["segment"]] = [np.median([e["rms_signal_uV"] for e in b["events"]]),
                                     np.median([e["rms_wings_uV"] for e in b["events"]])]
        x = pd.read_csv(LIB / "extra_records.csv")
        x = x[~x["segment"].isin(failed)]
        x = x[(x["rms_signal_uV"] <= EXTRA_RMS_FACTOR * x["segment"].map(ref["rms_signal_uV"]))
              & (x["rms_wings_uV"] <= EXTRA_RMS_FACTOR * x["segment"].map(ref["rms_wings_uV"]))]
        x = x.sort_values("time").reset_index(drop=True)
        print(f"extension: {n_new_seg} new tune segments kept, failed {sorted(failed)}; {len(x)} extra records kept")
        Fx, Rx, PKx, Nx, Cx = [], [], [], [], []
        for r in x.itertuples():
            z = np.load(LIB / f"npz_extra/event_{r.file[:19]}_{r.index}.npz")
            pk = float(np.abs(z["absorb"]).max())
            s = np.sign(np.sum(z["absorb"]))
            xx = (z["f"] - r.f0) * np.sign(r.P)
            Fx.append(z["f"])
            Rx.append(z["resid"])
            PKx.append(pk)
            Nx.append(z["resid"] - np.interp(xx, X, template) * pk * s)
            c = np.empty(9)
            sh = segs[r.segment]["shared"]
            c[SF.SHARED_C] = [sh[M.CIRCUIT_NAMES[k]] for k in SF.SHARED_C]
            c[SF.OWN_C] = [r.U, r.phi_deg, r.DC_offset]
            Cx.append(c)
        F, RES, NOISE = np.vstack([F, Fx]), np.vstack([RES, Rx]), np.vstack([NOISE, Nx])
        PK = np.concatenate([PK, PKx])
        circuit = np.vstack([circuit, Cx])
        sigma = np.concatenate([sigma, [segs[s]["shared"]["sigma_kHz"] / 1e3 for s in x["segment"]]])
        klib = np.concatenate([klib, [k_seg[s][0] for s in x["segment"]]])
        quality = np.concatenate([quality, np.where(x["new_segment"], "extra_new_tune", "extra")])
        seg_of = np.concatenate([seg_of, x["segment"]])
        g_arr, f0_arr = np.concatenate([g_arr, x["g"]]), np.concatenate([f0_arr, x["f0"]])
        A_arr, P_arr = np.concatenate([A_arr, x["A"]]), np.concatenate([P_arr, x["P"]])
        time_arr = np.concatenate([time_arr, x["time"].astype(str)])
        P_err = np.concatenate([P_err, np.full(len(x), np.nan)])
        lab = quality == "label"

    np.savez_compressed(
        LIB / "param_library.npz",
        circuit=circuit, circuit_names=np.array(M.CIRCUIT_NAMES), sigma=sigma, g=g_arr,
        f0=f0_arr, k_seg=klib, A=A_arr, P_fit=P_arr, P_err=P_err,
        segment=seg_of, time=time_arr, quality=quality,
        freq=F, noise=NOISE, resid=RES, peak=PK, template_x=X, template=template,
        eta=SF.ETA, wq=SF.WQ)

    noise_rms = np.sqrt(np.mean(NOISE**2, axis=1))
    # spectral colour of the noise: lag-1 autocorrelation (0 = white)
    ac1 = np.array([np.corrcoef(n[:-1], n[1:])[0, 1] for n in NOISE])
    summary = {
        "n_records": int(len(quality)), "by_quality": {str(q): int((quality == q).sum()) for q in np.unique(quality)},
        "n_segments": int(len(np.unique(seg_of))), "n_new_tune_segments": n_new_seg,
        "segments_label": sorted(int(s) for s in np.unique(seg_of[lab])),
        "eta_fixed": SF.ETA, "wq_fixed": SF.WQ,
        "sigma_kHz_per_segment": {int(s): round(segs[s]["shared"]["sigma_kHz"], 2) for s in segs},
        "k_seg": {int(s): {"k": round(v[0], 5), "rel_spread": round(v[1], 4)} for s, v in k_seg.items()},
        "g_label_quantiles_10_50_90": [float(np.quantile(g_arr[lab], q)) for q in (0.1, 0.5, 0.9)],
        "noise_rms_uV_quantiles_10_50_90": [float(np.quantile(noise_rms, q) * 1e6) for q in (0.1, 0.5, 0.9)],
        "noise_lag1_autocorr_median": float(np.median(ac1)),
        "template_peak_abs_pct": float(np.max(np.abs(template)) * 100),
        "template_explains_horn_residual_power_median": float(np.median(expl[d["is_label"].to_numpy()])),
        "P_fit_range_label": [float(np.abs(P_arr[lab]).min()), float(np.abs(P_arr[lab]).max())],
    }
    (LIB / "library_summary.yaml").write_text(yaml.safe_dump(summary, sort_keys=False))
    print(yaml.safe_dump(summary, sort_keys=False))

    # ---- plot ----
    fig, ax = plt.subplots(2, 3, figsize=(19, 10))
    a = ax[0, 0]
    for q, c in (("extra", "tab:orange"), ("extra_new_tune", "tab:green"), ("circuit", "tab:gray"), ("label", "tab:blue")):
        m = quality == q
        if m.any():
            a.scatter(np.log10(circuit[m, 0]), circuit[m, 3], s=6, c=c, alpha=0.5, label=f"{q} ({m.sum()})")
    a.set_xlabel("log10 U")
    a.set_ylabel("phase [deg]")
    a.set_title("Circuit states: U and phase per event (clusters = tune segments)", fontsize=10.5)
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3)
    a = ax[0, 1]
    ss = sorted(segs)
    a.bar(range(len(ss)), [segs[s]["shared"]["sigma_kHz"] for s in ss],
          color=["tab:blue" if s in summary["segments_label"] else "tab:gray" for s in ss])
    a.set_xticks(range(len(ss)))
    a.set_xticklabels(ss, rotation=60, fontsize=8)
    a.set_ylabel("σ (Gaussian broadening) [kHz]")
    a.set_title("σ per segment (blue: label segments, grey: others)", fontsize=10.5)
    a.grid(alpha=0.3, axis="y")
    a = ax[0, 2]
    for q, c in (("label", "tab:blue"), ("circuit", "tab:gray")):
        m = quality == q
        a.scatter(np.abs(P_arr[m]), g_arr[m], s=8, c=c, alpha=0.6, label=q)
    a.set_yscale("log")
    a.set_xlabel("|P| (fit)")
    a.set_ylabel("g (Lorentzian-type width)")
    a.set_title("g vs |P|: drawn together with its event", fontsize=10.5)
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3)
    a = ax[1, 0]
    for k, v in tbins.items():
        a.plot(X * 1e3, v * 100, lw=1, label=f"|P| {k}")
    a.plot(X * 1e3, template * 100, "k", lw=2, label="template (all labels)")
    a.set_xlim(-250, 250)
    a.set_xlabel("(f − f0) · sign(P) [kHz]  (larger horn on the right)")
    a.set_ylabel("residual / absorption peak [%]")
    a.set_title(f"Signal-correlated misfit template (explains {np.median(expl[d["is_label"].to_numpy()]) * 100:.0f} % of horn residual power)",
                fontsize=10.5)
    a.legend(fontsize=8)
    a.grid(alpha=0.3)
    a = ax[1, 1]
    i = int(np.flatnonzero(lab)[len(np.flatnonzero(lab)) // 2])
    a.plot((F[i] - f0_arr[i]) * 1e3, RES[i] * 1e6, color="0.6", lw=0.8, label="fit residual")
    a.plot((F[i] - f0_arr[i]) * 1e3, NOISE[i] * 1e6, color="tab:red", lw=0.8, label="noise = residual − template")
    a.set_xlabel("f − f0 [kHz]")
    a.set_ylabel("µV")
    a.set_title(f"Example {time_arr[i]} (P {P_arr[i]:+.2f})", fontsize=10.5)
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3)
    a = ax[1, 2]
    fr = np.fft.rfftfreq(F.shape[1])
    ps = np.median(np.abs(np.fft.rfft(NOISE[lab], axis=1)) ** 2, axis=0)
    a.loglog(fr[1:], ps[1:], color="tab:red", label="noise library (median)")
    white = np.median(noise_rms[lab]) ** 2 * F.shape[1] / 2
    a.axhline(white, color="k", ls="--", lw=0.8, label="white noise, same rms")
    a.set_xlabel("frequency [1 / bin]")
    a.set_ylabel("power")
    a.set_title(f"Noise is coloured (lag-1 autocorr {np.median(ac1):.2f}): use real samples, not white", fontsize=10.5)
    a.legend(fontsize=8.5)
    a.grid(alpha=0.3, which="both")
    fig.suptitle(f"Stage 4: MC parameter library — {len(quality)} fitted events from {len(np.unique(seg_of))} tune segments "
                 f"({lab.sum()} labels)", weight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS / "stage4_library.png", dpi=130)
    print(f"wrote {LIB / 'param_library.npz'}, {PLOTS / 'stage4_library.png'}")


if __name__ == "__main__":
    main()
