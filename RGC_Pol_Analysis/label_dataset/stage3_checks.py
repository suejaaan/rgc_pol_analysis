"""Stage 3, compute part: MC closure and model variation of the segment-joint lineshape labels.

Closure: for each test segment, the 10 joint events are regenerated through the segment's fitted circuit and
sigma with their own fitted U/phi/DC/f0/g, at a known random P_true in [P_LO, P_HI] (sign of the data). The
signal size follows P_true (absorption area scaled with P_true, as in a real segment). Noise variants:
  white      Gaussian noise with the event's wing-residual rms
  real       the event's own fit residual added (real noise + unmodelled structure)
The MC segment is then fitted exactly like data: the segment's original start circuits (pilot / global ones,
never the truth) -> joint fit -> best cost.  Bias = P_fit - P_true.

Model variation: the same real joint events refitted (warm start from the segment's best shared solution) with
one modelling choice changed at a time (VARIANTS). Delta P vs the nominal refit = model systematic.

Usage: python stage3_checks.py [closure] [variants]      (default: both)
Output: results/stage3/closure.yaml, results/stage3/variants.yaml
"""
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
import model as M  # noqa: E402
import segment_fit as SF  # noqa: E402

OUT = HERE / "results" / "stage3"
SEGMENTS = [96, 103, 110, 119, 125, 133]
P_LO, P_HI = 0.28, 0.56
N_SEEDS = 2
NOMINAL = dict(eta=0.111, wq=0.0825, nphi=32, g_lo=0.001, sigma_on=True)
VARIANTS = {
    "nominal": {},
    "eta_lo": dict(eta=0.105), "eta_hi": dict(eta=0.117),
    "wq_lo": dict(wq=0.0818), "wq_hi": dict(wq=0.0832),
    "no_gauss": dict(sigma_on=False),
    "nphi_128": dict(nphi=128),
    "g_lo_1e-4": dict(g_lo=1e-4),
}


def configure(cfg):
    """Set the model globals in this worker (jobs run one after another in the same process)."""
    c = dict(NOMINAL, **cfg)
    SF.ETA, SF.WQ = c["eta"], c["wq"]
    M.POWDER_NPHI = c["nphi"]
    SF.OW_LO[6] = c["g_lo"]
    SF.SH_HI[6] = SF.V.SIG_HI if c["sigma_on"] else 1e-7
    SF.SIGMA0 = 0.0045 if c["sigma_on"] else 0.0


def segment_inputs(seg_res, picks):
    """Best solution, its event dicts (file/index from the picks) and own-parameter starts."""
    best = seg_res["solutions"][0]
    p = picks[picks["segment"] == seg_res["segment"]].set_index("time")
    evs = [dict(p.loc[e["time"]].to_dict(), time=e["time"]) for e in best["events"]]
    sh = np.array([best["shared"][M.CIRCUIT_NAMES[k]] for k in SF.SHARED_C] + [best["shared"]["sigma_kHz"] / 1e3])
    return best, evs, sh


def circuit_of(sh, e):
    c = np.empty(9)
    c[SF.SHARED_C] = sh[:6]
    c[SF.OWN_C] = [e["U"], e["phi_deg"], e["DC_offset"]]
    return c


# ---------------- closure ----------------
def closure_job(job):
    seg, noise, seed, best, evs, sh, starts, sig_real = job
    configure({})
    rng = np.random.default_rng(1000 * seg + 10 * seed + (noise == "real"))
    mc_evs, truth = [], []
    for e_fit, ev in zip(best["events"], evs):
        npz = np.load(HERE / "results/stage2/npz_full" / f"event_{ev['file'][:19]}_{int(ev['index'])}.npz")
        f = npz["f"]
        p_true = float(np.sign(e_fit["P"]) * rng.uniform(P_LO, P_HI))
        r0, r1 = M.p_to_r(e_fit["P"]), M.p_to_r(p_true)
        A = e_fit["A"] * (p_true / e_fit["P"]) * (r0 + 1) / (r1 + 1)  # absorption area follows P_true
        own = np.array([e_fit["U"], e_fit["phi_deg"], e_fit["DC_offset"], A, p_true, e_fit["f0"], e_fit["g"]])
        clean = SF.ev_model(f, sh, own)
        if noise == "white":
            s = (f > SF.V.SREG[0]) & (f < SF.V.SREG[1])
            raw = clean + rng.normal(0, np.sqrt(np.mean(npz["resid"][~s] ** 2)), len(f))
        else:
            raw = clean + npz["resid"]
        area_scale = p_true / e_fit["P"]  # absorption area follows P (A (r+1) scaled above)
        mc_evs.append(dict(ev, mc_f=f, mc_raw=raw, area_online=ev["area_online"] * area_scale))
        truth.append(p_true)
    sig = {e["time"]: sig_real[e["time"]] for e in mc_evs}
    res = SF.joint_segment((seg, starts, sig, mc_evs))
    fit = res["solutions"][0]
    return {"segment": seg, "noise": noise, "seed": seed, "P_true": truth,
            "P_fit": [e["P"] for e in fit["events"]],
            "P_fit_other_starts": [[e["P"] for e in s["events"]] for s in res["solutions"][1:]],
            "start_best": fit["start"], "seconds": res["seconds"]}


def run_closure(segres, picks):
    pilot = yaml.safe_load((HERE / "results/stage2/fits_pilot.yaml").read_text())
    base = yaml.safe_load(SF.FS.BASE_FITS.read_text())["baselines"]
    gstarts = SF.global_circuit_starts(pilot)
    jobs = []
    for s in segres:
        if s["segment"] not in SEGMENTS:
            continue
        best, evs, sh = segment_inputs(s, picks)
        _, starts_all, _ = SF.segment_setup(s["segment"], pilot, picks, base, gstarts)
        names = [sol["start"] for sol in s["solutions"]]
        starts = [st for st in starts_all if st[0] in names]       # the data's start circuits, not the truth
        sig = {}
        for e in best["events"]:
            p = min(pilot, key=lambda q: abs(np.datetime64(q["time"]) - np.datetime64(e["time"])) if q["segment"] == s["segment"] else np.timedelta64(10**9, "s"))
            sig[e["time"]] = dict(p["signal"])
        for noise in ("white", "real"):
            for seed in range(N_SEEDS):
                jobs.append((s["segment"], noise, seed, best, evs, sh, starts, sig))
    print(f"closure: {len(jobs)} MC segments", flush=True)
    with Pool(SF.N_WORKERS) as pool:
        out = pool.map(closure_job, jobs, chunksize=1)
    (OUT / "closure.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    d = np.concatenate([np.abs(o["P_fit"]) - np.abs(o["P_true"]) for o in out])
    print(f"closure: bias |P_fit|-|P_true| median {np.median(d):+.4f}, rms {np.sqrt(np.mean(d**2)):.4f}", flush=True)
    for noise in ("white", "real"):
        dd = np.concatenate([np.abs(o["P_fit"]) - np.abs(o["P_true"]) for o in out if o["noise"] == noise])
        print(f"  {noise:5s}: median {np.median(dd):+.4f}  rms {np.sqrt(np.mean(dd**2)):.4f}  max |.| {np.abs(dd).max():.4f}")


# ---------------- model variation ----------------
def variant_job(job):
    seg, name, cfg, best, evs, sh = job
    configure(cfg)
    circuit = circuit_of(sh, best["events"][0])
    sig = {e["time"]: {"A": e["A"], "P": e["P"], "f0": e["f0"]} for e in best["events"]}
    res = SF.joint_segment((seg, [(f"warm {name}", circuit)], sig, evs))
    sol = res["solutions"][0]
    return {"segment": seg, "variant": name, "cost": sol["cost"], "sigma_kHz": sol["shared"]["sigma_kHz"],
            "times": [e["time"] for e in sol["events"]], "P": [e["P"] for e in sol["events"]],
            "g": [e["g"] for e in sol["events"]],
            "rms_signal_uV": [e["rms_signal_uV"] for e in sol["events"]], "seconds": res["seconds"]}


def run_variants(segres, picks):
    jobs = []
    for s in segres:
        if s["segment"] not in SEGMENTS:
            continue
        best, evs, sh = segment_inputs(s, picks)
        for name, cfg in VARIANTS.items():
            jobs.append((s["segment"], name, cfg, best, evs, sh))
    print(f"variants: {len(jobs)} joint refits", flush=True)
    with Pool(SF.N_WORKERS) as pool:
        out = pool.map(variant_job, jobs, chunksize=1)
    (OUT / "variants.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    nom = {o["segment"]: np.abs(o["P"]) for o in out if o["variant"] == "nominal"}
    for name in VARIANTS:
        if name == "nominal":
            continue
        d = np.concatenate([np.abs(o["P"]) - nom[o["segment"]] for o in out if o["variant"] == name])
        c = np.median([o["cost"] / next(x["cost"] for x in out if x["variant"] == "nominal" and x["segment"] == o["segment"])
                       for o in out if o["variant"] == name])
        print(f"  {name:10s}: Δ|P| median {np.median(d):+.4f}, range {d.min():+.4f} .. {d.max():+.4f}, cost ×{c:.2f}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    what = sys.argv[1:] or ["closure", "variants"]
    segres = yaml.safe_load((HERE / "results/stage2/segments.yaml").read_text())
    picks = pd.read_csv(HERE / "results/plateau_events.csv")
    t0 = time.time()
    if "variants" in what:
        run_variants(segres, picks)
        print(f"variants done at {(time.time() - t0) / 60:.1f} min", flush=True)
    if "closure" in what:
        run_closure(segres, picks)
        print(f"closure done at {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
