"""Scan data-d once and collect every distinct baseline sweep used by the online analysis.

A baseline is identified by (base_file, base_stamp). Only non-empty run files plus the
current_* files without a non-empty run twin are read, so no event is counted twice.

Writes:
  cache/baselines.npz         baseline arrays + freq grids, one row per unique baseline
  cache/baselines_meta.yaml   per-baseline usage metadata
  cache/event_baseline_map.csv  file, event index, label, baseline id (one row per event)
"""
import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parents[1] / "data_creation/data-d"
CACHE = HERE / "cache"


def unique_data_files():
    files = sorted(os.listdir(DATA_DIR))
    size = {f: (DATA_DIR / f).stat().st_size for f in files}
    runs = {f.split("__")[0]: f for f in files if "__" in f and size[f] > 0}
    orphans = [f for f in files if f.startswith("current_") and size[f] > 0 and f[len("current_"):-4] not in runs]
    return sorted(runs.values()) + orphans


def main():
    CACHE.mkdir(exist_ok=True)
    baselines, meta, rows = {}, {}, []
    conflicts = 0

    for fname in unique_data_files():
        with open(DATA_DIR / fname) as fh:
            for idx, line in enumerate(fh):
                if not line.strip():
                    continue
                d = json.loads(line)
                key = (d.get("base_file"), d.get("base_stamp"))
                arr = np.asarray(d["baseline"], dtype=np.float64)
                digest = hashlib.sha1(arr.tobytes()).hexdigest()
                if key not in meta:
                    bid = len(meta)
                    baselines[key] = (arr, np.asarray(d["freq_list"], dtype=np.float64))
                    meta[key] = {
                        "id": bid, "base_file": key[0], "base_stamp": key[1], "base_time": d.get("base_time"),
                        "channel": d["channel"]["name"], "n_bins": len(arr), "digest": digest,
                        "n_events": 0, "labels": {}, "cc_values": set(), "first_event": d["start_time"],
                        "last_event": d["start_time"], "files": set(),
                    }
                m = meta[key]
                if digest != m["digest"]:
                    conflicts += 1
                m["n_events"] += 1
                lab = str(d.get("label"))
                m["labels"][lab] = m["labels"].get(lab, 0) + 1
                m["cc_values"].add(float(d["cc"]))
                m["last_event"] = max(m["last_event"], d["start_time"])
                m["first_event"] = min(m["first_event"], d["start_time"])
                m["files"].add(fname)
                rows.append((fname, idx, lab, m["id"]))
        print(f"{fname}: {len(meta)} baselines so far", flush=True)

    ordered = sorted(meta.values(), key=lambda m: m["id"])
    keys = sorted(meta, key=lambda k: meta[k]["id"])
    max_bins = max(m["n_bins"] for m in ordered)
    base = np.full((len(keys), max_bins), np.nan)
    freq = np.full((len(keys), max_bins), np.nan)
    for i, k in enumerate(keys):
        a, f = baselines[k]
        base[i, : len(a)] = a
        freq[i, : len(f)] = f
    np.savez(CACHE / "baselines.npz", baseline=base, freq=freq)

    for m in ordered:
        m["cc_values"] = sorted(m["cc_values"])
        m["files"] = sorted(m["files"])
        m.pop("digest")
    (CACHE / "baselines_meta.yaml").write_text(yaml.safe_dump(ordered, sort_keys=False))
    with open(CACHE / "event_baseline_map.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "event_index", "label", "baseline_id"])
        w.writerows(rows)
    print(f"\n{len(rows)} events, {len(keys)} unique baselines, "
          f"{conflicts} events whose (base_file, base_stamp) maps to a different array than first seen")


if __name__ == "__main__":
    main()
