"""Rerun events whose fit residual is above BAD_RMS with a wider search; keep the better fit."""
import time
from multiprocessing import Pool

import numpy as np
import yaml

import fit_selected as FS
import model as M

BAD_RMS = 80e-6


def wide(job):
    FS.N_CIRCUIT_STARTS = 200
    FS.N_JOINT_STARTS = 8
    FS.CANDIDATE_COST_FACTOR = 10.0
    return FS.fit_one(job)


if __name__ == "__main__":
    fits = yaml.safe_load((FS.OUT / "fits.yaml").read_text())
    base = yaml.safe_load(FS.BASE_FITS.read_text())["baselines"]
    bad = [r for r in fits if r["fit_rms_V"] > BAD_RMS]
    keys = ("file", "index", "time", "cc", "pol", "area", "baseline_id", "period_note")
    jobs = []
    for r in bad:
        ev = {k: r[k] for k in keys}
        b = base.get(r["baseline_id"])
        jobs.append((ev, [b["params"][n] for n in M.CIRCUIT_NAMES] if b else None))
    print(f"refitting {len(jobs)} events with rms > {BAD_RMS * 1e6:.0f} µV", flush=True)
    t0 = time.time()
    backup = {}
    for r in bad:  # keep the first-pass arrays in case the wide search is worse
        p = FS.OUT / f"event_{r['file'][:19]}_{r['index']}.npz"
        backup[(r["file"], r["index"])] = dict(np.load(p))
    with Pool(len(jobs)) as pool:
        new = pool.map(wide, jobs)
    by_key = {(r["file"], r["index"]): r for r in fits}
    for n in new:
        old = by_key[(n["file"], n["index"])]
        better = n["fit_rms_V"] < old["fit_rms_V"]
        print(f"  {n['time']} cc {n['cc']:7.3f}: rms {old['fit_rms_V'] * 1e6:.0f} -> {n['fit_rms_V'] * 1e6:.0f} µV, "
              f"mismatch {old['mismatch_rel'] * 100:+.1f} -> {n['mismatch_rel'] * 100:+.1f} %  {'(kept new)' if better else '(kept old)'}",
              flush=True)
        if better:
            by_key[(n["file"], n["index"])] = {**n, "refit_wide": True}
        else:
            np.savez(FS.OUT / f"event_{n['file'][:19]}_{n['index']}.npz", **backup[(n["file"], n["index"])])
    out = sorted(by_key.values(), key=lambda r: r["time"])
    (FS.OUT / "fits.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    print(f"done in {(time.time() - t0) / 60:.1f} min")
