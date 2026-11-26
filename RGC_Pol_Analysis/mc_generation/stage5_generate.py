"""Stage 5: Monte Carlo NMR sweeps for ML training (no cc anywhere).

Per MC sweep:
  1. P drawn from [-0.65, +0.65], 50/50 sign; |P| mixture P_MIX (most weight on the physics range 0.25-0.55)
  2. one library record drawn whole from the sweep's split (circuit, sigma, g, f0, k_seg of its segment);
     U, phase, DC offset jittered by JITTER x their within-segment spread
  3. signal amplitude A = k_seg * P / (r(P) + 1)  (signal size per unit P of that tune, from the cc-free labels)
  4. sweep = Q-meter(circuit, chi'(f) - i chi''(f))  with chi'' = Dulya powder lineshape (*) Gaussian sigma
            + misfit template * absorption peak (mirrored with the sign of P)
            + real noise of the SAME tune segment: with prob. P_PLAIN a plain library sample (no sign flip), else
              the segment's mean noise pattern + (n_b - n_c)/sqrt(2) * random sign  (two other samples of the
              segment: the systematic part cancels in the difference, only the random part varies)
  The noise is mostly a systematic background pattern of each tune (stage5_check: noise from another segment
  or a sign-flipped sample is recognisable, AUC 0.66-0.74; the two modes above give AUC 0.43-0.46).
Splits are by circuit group (segments that share a shared circuit stay together): no tune, record or noise
sample of the val/test tunes is ever used in train.
Only segments with a reliable k (lineshape labels or clean fits) generate signals: K_SEGMENTS.

Output (parquet, upstream layout): columns "0".."511" = sweep [V] (float32), then P, Pzz, split, segment,
record_id, quality, noise_id, noise_aug, A, k_seg, sigma, g, f0, circuit (9), signal_peak, area_chi2,
noise_rms, SNR.  Shards of SHARD rows; freq.npy (common frequency grid) and manifest.yaml alongside.

Usage: python stage5_generate.py N NAME          e.g.  python stage5_generate.py 10000 test10k
                                                        python stage5_generate.py 1000000 mc1m
"""
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "qmeter_dulya_fit"))
sys.path.insert(0, str(HERE.parent / "label_dataset"))
import model as M  # noqa: E402
import model_variants as V  # noqa: E402

LIB = HERE / "library" / "param_library.npz"
OUTROOT = HERE / "mc_data"
SHARD = 50_000
N_WORKERS = int(os.environ.get("N_WORKERS", max(1, (os.cpu_count() or 4) - 2)))
P_MAX = 0.65
P_MIX = [((0.00, 0.15), 0.12), ((0.15, 0.25), 0.23), ((0.25, 0.55), 0.55), ((0.55, 0.65), 0.10)]
JITTER = 0.5
P_PLAIN = 0.5
SPLIT_FRACTION = {"train": 0.8, "val": 0.1, "test": 0.1}
K_SEGMENTS = {93, 96, 97, 101, 103, 105, 110, 112, 115, 116, 118, 119, 122, 123, 125, 133, 135, 95, 124, 150, 154}
# circuit groups held out (segments sharing a shared circuit stay together)
TEST_GROUPS = [[96], [118, 119]]
VAL_GROUPS = [[105, 135]]
SHARED_C, OWN_C = [1, 2, 5, 6, 7, 8], [0, 3, 4]

_L = {}


def load_library():
    z = np.load(LIB, allow_pickle=True)
    L = {k: z[k] for k in z.files}
    seg = L["segment"]
    split = np.full(len(seg), "train", dtype=object)
    for g in TEST_GROUPS:
        split[np.isin(seg, g)] = "test"
    for g in VAL_GROUPS:
        split[np.isin(seg, g)] = "val"
    L["split"] = split
    L["k_ok"] = np.isin(seg, list(K_SEGMENTS))
    # within-segment spread of the own circuit parameters (for the jitter)
    sd = np.zeros((len(seg), 3))
    for s in np.unique(seg):
        m = seg == s
        sd[m] = L["circuit"][m][:, OWN_C].std(axis=0)
    L["own_sd"] = sd
    L["noise_mean_seg"] = {int(s): L["noise"][seg == s].mean(axis=0) for s in np.unique(seg)}
    L["noise_pool_seg"] = {int(s): np.flatnonzero(seg == s) for s in np.unique(seg)}
    return L


def init_worker():
    _L.update(load_library())


def draw_absP(rng, n):
    w = np.array([p for _, p in P_MIX])
    comp = rng.choice(len(P_MIX), n, p=w / w.sum())
    lo = np.array([P_MIX[c][0][0] for c in comp])
    hi = np.array([P_MIX[c][0][1] for c in comp])
    return rng.uniform(lo, hi)


def draw_noise(rng, L, seg, exclude=-1):
    """Real noise of tune segment seg (see the module docstring). Returns noise, id string, augmentation."""
    pool = L["noise_pool_seg"][int(seg)]
    pool = pool[pool != exclude]
    if rng.random() < P_PLAIN:
        a = rng.choice(pool)
        return L["noise"][a], str(a), "plain"
    b, c = rng.choice(pool, 2, replace=False)
    sgn = rng.choice([-1.0, 1.0])
    return L["noise_mean_seg"][int(seg)] + sgn * (L["noise"][b] - L["noise"][c]) / np.sqrt(2), f"mean+{b}-{c}", "mean+diff"


def make_sweep(f, circuit, P, f0, g, sigma, k, template_x, template):
    """Clean MC sweep (background + signal + misfit template) and its absorption part."""
    A = k * P / (M.p_to_r(P) + 1)
    chi2, chi1 = V.chi_gauss(f, A, P, f0, V.WQ_FIX, V.ETA_FIX, g, sigma)
    bg = M.qmeter(f, *circuit)
    full = M.qmeter(f, *circuit, chi=chi1 - 1j * chi2)
    absorb = M.qmeter(f, *circuit, chi=-1j * chi2) - bg
    pk = float(np.max(np.abs(absorb)))
    s = np.sign(np.sum(absorb)) if pk > 0 else 1.0
    x = (f - f0) * (np.sign(P) if P != 0 else 1.0)
    misfit = np.interp(x, template_x, template) * pk * s
    return full + misfit, A, pk, float(np.sum(absorb))


def gen_chunk(job):
    split, n, seed, path = job
    L = _L
    rng = np.random.default_rng(seed)
    f = L["freq"][0]
    rec_pool = np.flatnonzero((L["split"] == split) & L["k_ok"])
    absP = draw_absP(rng, n)
    P = absP * rng.choice([-1.0, 1.0], n)
    X = np.empty((n, len(f)), np.float32)
    meta = {k: np.empty(n) for k in ("A", "k_seg", "sigma", "g", "f0", "signal_peak", "area_chi2", "noise_rms")}
    circ = np.empty((n, 9))
    rec = rng.choice(rec_pool, n)
    nid = np.empty(n, dtype=object)
    naug = np.empty(n, dtype=object)
    for i in range(n):
        r = rec[i]
        c = L["circuit"][r].copy()
        c[OWN_C] += rng.normal(0, JITTER, 3) * L["own_sd"][r]
        c[0] = max(c[0], 1e-3)
        clean, A, pk, area = make_sweep(f, c, P[i], L["f0"][r], L["g"][r], L["sigma"][r], L["k_seg"][r],
                                        L["template_x"], L["template"])
        noise, nid[i], naug[i] = draw_noise(rng, L, L["segment"][r])
        X[i] = clean + noise
        circ[i] = c
        meta["A"][i], meta["k_seg"][i], meta["sigma"][i] = A, L["k_seg"][r], L["sigma"][r]
        meta["g"][i], meta["f0"][i], meta["signal_peak"][i], meta["area_chi2"][i] = L["g"][r], L["f0"][r], pk, area
        meta["noise_rms"][i] = float(np.sqrt(np.mean(noise**2)))
    cols = {str(j): X[:, j] for j in range(X.shape[1])}
    cols["P"] = P.astype(np.float32)
    cols["Pzz"] = (2 - np.sqrt(4 - 3 * P**2)).astype(np.float32)
    cols["split"] = np.full(n, split)
    cols["segment"] = L["segment"][rec].astype(np.int32)
    cols["record_id"] = rec.astype(np.int32)
    cols["quality"] = L["quality"][rec].astype(str)
    cols["noise_id"] = nid.astype(str)
    cols["noise_aug"] = naug.astype(str)
    for k, v in meta.items():
        cols[k] = v.astype(np.float32)
    for j, name in enumerate(L["circuit_names"]):
        cols[str(name)] = circ[:, j].astype(np.float32)
    cols["SNR"] = (meta["signal_peak"] / meta["noise_rms"]).astype(np.float32)
    pd.DataFrame(cols).to_parquet(path, compression="snappy", index=False)
    return split, n, str(path)


def main():
    n_total = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000
    name = sys.argv[2] if len(sys.argv) > 2 else f"mc{n_total}"
    out = OUTROOT / name
    out.mkdir(parents=True, exist_ok=True)
    L = load_library()
    np.save(out / "freq.npy", L["freq"][0])
    # jobs: chunks of at most SHARD // 10 rows (load balancing), merged per shard afterwards
    chunk = max(1000, min(SHARD // 10, n_total // (4 * N_WORKERS) or 1000))
    jobs, k = [], 0
    for split, frac in SPLIT_FRACTION.items():
        n_split = int(round(n_total * frac))
        for start in range(0, n_split, chunk):
            jobs.append((split, min(chunk, n_split - start), 12345 + k, out / f"part_{k:05d}_{split}.parquet"))
            k += 1
    print(f"{name}: {n_total} sweeps in {len(jobs)} chunks on {N_WORKERS} workers -> {out}", flush=True)
    t0 = time.time()
    with Pool(N_WORKERS, initializer=init_worker) as pool:
        for i, _ in enumerate(pool.imap_unordered(gen_chunk, jobs), 1):
            if i % max(1, len(jobs) // 10) == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} chunks, {(time.time() - t0) / 60:.1f} min", flush=True)
    # merge chunks into shards of SHARD rows per split
    shards = []
    for split in SPLIT_FRACTION:
        parts = sorted(out.glob(f"part_*_{split}.parquet"))
        buf, nbuf, s = [], 0, 0
        for p in parts:
            buf.append(pd.read_parquet(p))
            nbuf += len(buf[-1])
            p.unlink()
            if nbuf >= SHARD or p == parts[-1]:
                df = pd.concat(buf, ignore_index=True)
                fn = out / f"{split}_{s:03d}.parquet"
                df.to_parquet(fn, compression="snappy", index=False)
                shards.append({"file": fn.name, "rows": len(df)})
                buf, nbuf, s = [], 0, s + 1
    seg_split = {sp: sorted(int(x) for x in np.unique(L["segment"][(L["split"] == sp) & L["k_ok"]])) for sp in SPLIT_FRACTION}
    manifest = {
        "name": name, "n_total": n_total, "created_seconds": round(time.time() - t0),
        "library": str(LIB.relative_to(HERE)), "columns": "0..511 sweep [V] float32, then P, Pzz, split, segment, "
        "record_id, quality, noise_id, noise_aug, A, k_seg, sigma, g, f0, signal_peak, area_chi2, noise_rms, "
        "9 circuit parameters, SNR",
        "freq_grid": "freq.npy (MHz, common to all sweeps)",
        "P_max": P_MAX, "P_mix_absP": [[list(r), w] for r, w in P_MIX], "sign": "50/50",
        "jitter_own_circuit_x_within_segment_sd": JITTER,
        "noise": f"same tune segment; plain sample with prob. {P_PLAIN}, else segment mean + random-sign difference of two samples / sqrt(2)",
        "eta_fixed": float(V.ETA_FIX), "wq_fixed": float(V.WQ_FIX),
        "segments_per_split": seg_split, "test_groups": TEST_GROUPS, "val_groups": VAL_GROUPS,
        "shards": shards,
    }
    (out / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min; segments per split {seg_split}")


if __name__ == "__main__":
    main()
