"""Real-data validation: run a trained model on the 484 real lineshape-labelled sweeps (label_dataset stage 3).

The real labels carry the same tune split as the MC: labels from the MC test tunes (96, 118, 119) and val tunes
(105, 135) were never seen in any form in training -> they are the true real-data test. Labels from train tunes
share circuit and noise with the training MC (but the sweeps themselves were never used).

  python eval_real.py export                     (once, on the machine with label_dataset/results: writes
                                                  data_real/real_labels.parquet, ~2 MB; needs pyarrow, no torch)
  python eval_real.py eval --version VERSION     (any machine with torch and data_real/real_labels.parquet)

Output: Model_Performance/<version>/real_<split>.png, _metrics.yaml, _predictions.csv
The comparison uses P_err (stage-3 label error) in the pulls.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
REAL = HERE / "data_real" / "real_labels.parquet"


def export():
    sys.path.insert(0, str(ROOT / "mc_generation"))
    import stage5_generate as G  # split groups of the MC
    lab = pd.read_csv(ROOT / "label_dataset/results/stage3/labels_final.csv")
    rows, freq = [], None
    for r in lab.itertuples():
        z = np.load(ROOT / f"label_dataset/results/stage2/npz_full/event_{r.file[:19]}_{r.index}.npz")
        freq = z["f"] if freq is None else freq
        rows.append(z["raw"].astype(np.float32))
    split = np.full(len(lab), "train", dtype=object)
    for g in G.TEST_GROUPS:
        split[lab["segment"].isin(g)] = "test"
    for g in G.VAL_GROUPS:
        split[lab["segment"].isin(g)] = "val"
    df = pd.DataFrame(np.array(rows), columns=[str(i) for i in range(len(freq))])
    df["P"] = lab["P"].astype(np.float32)
    df["P_err"] = lab["err_total"].astype(np.float32)
    df["segment"] = lab["segment"].astype(np.int32)
    df["time"] = lab["time"]
    df["split"] = split
    REAL.parent.mkdir(exist_ok=True)
    df.to_parquet(REAL, index=False)
    np.save(REAL.parent / "freq.npy", freq)
    print(f"wrote {REAL}: {len(df)} labels, split {pd.Series(split).value_counts().to_dict()}")


def evaluate(version, device="auto"):
    import torch
    import data as D
    import evaluate as E
    from models import build_model
    from train import device_auto, predict

    cfg = yaml.safe_load((HERE / "Models" / version / "config.yaml").read_text())
    dev = device_auto(device)
    model = build_model(cfg["model"], cfg["in_channels"]).to(dev)
    model.load_state_dict(torch.load(HERE / "Models" / version / "best_model.pth", map_location=dev, weights_only=True))
    df = pd.read_parquet(REAL)
    freq = np.load(REAL.parent / "freq.npy")
    X = D.preprocess(df[[str(i) for i in range(D.N_BINS)]].to_numpy(np.float32), freq, cfg["input"])
    p, s = predict(model, X, dev, amp=dev.type == "cuda")
    df = df[["P", "P_err", "segment", "time", "split"]].assign(P_pred=p, sigma_pred=s)
    out = HERE / "Model_Performance" / version
    out.mkdir(parents=True, exist_ok=True)
    for split in ("test", "val", "train"):
        g = df[df["split"] == split]
        if len(g):
            E.report(g, out, f"real_{split}", title=f"{version}: real lineshape labels, {split} tunes")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["export", "eval"])
    ap.add_argument("--version")
    ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    if a.what == "export":
        export()
    else:
        evaluate(a.version, a.device)


if __name__ == "__main__":
    main()
