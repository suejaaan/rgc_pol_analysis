"""Train a polarization network on the RGC MC (mc_generation/stage5_generate.py). Adapted from ml/train_cnn_v2.py.

- splits from the dataset's `split` column (by tune segment): val for early stopping / model choice, test =
  held-out tunes, evaluated once at the end
- target P (vector polarization), output P and log sigma_P^2; loss L1 for the first WARMUP epochs, then
  Gaussian NLL + L1 (keeps the mean sharp while the uncertainty is learned)
- AdamW, cosine warm restarts (as v2), early stopping on val MAE; checkpoint + resume as upstream
- bf16 autocast on CUDA (RTX 5080), plain fp32 on MPS / CPU

Usage (from RGC_Pol_Analysis/ml):
  python train.py --data ../mc_generation/mc_data/mc1m --version cnn_both_1m
  python train.py --data ../mc_generation/mc_data/mc1m --version cnn_both_100k --max-train 100000   # learning curve
  python train.py --data ../mc_generation/mc_data/test10k --version smoke --epochs 2                  # quick test
Outputs: Models/<version>/ (best_model.pth, checkpoint.pth, config.yaml),
         Model_Performance/<version>/ (loss.csv, metrics.yaml, test_predictions.csv, plots)
"""
import argparse
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, TensorDataset

import data as D
import evaluate as E
from models import build_model

HERE = Path(__file__).resolve().parent


def device_auto(name="auto"):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def loss_fn(p, logvar, y, nll):
    l1 = F.l1_loss(p, y)
    if not nll:
        return l1
    return 0.5 * (logvar + (y - p) ** 2 * torch.exp(-logvar)).mean() + l1


def run_epoch(model, loader, device, opt=None, nll=True, amp=False, max_grad_norm=1.0):
    train = opt is not None
    model.train(train)
    tot, mae, n = 0.0, 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.set_grad_enabled(train), torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            p, lv = model(x)
        p, lv = p.float(), lv.float()
        loss = loss_fn(p, lv, y, nll)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
        tot += loss.item() * len(y)
        mae += (p - y).abs().sum().item()
        n += len(y)
    return tot / n, mae / n


@torch.no_grad()
def predict(model, X, device, batch=4096, amp=False):
    model.eval()
    P, S = [], []
    for i in range(0, len(X), batch):
        x = torch.from_numpy(X[i:i + batch]).to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
            p, lv = model(x)
        P.append(p.float().cpu().numpy())
        S.append(torch.exp(0.5 * lv.float()).cpu().numpy())
    return np.concatenate(P), np.concatenate(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--version", required=True)
    ap.add_argument("--model", default="cnn", choices=["cnn", "inception"])
    ap.add_argument("--input", default="both", choices=["both", "raw", "wingsub"])
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--warmup", type=int, default=3, help="epochs of pure L1 before the NLL term")
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    dev = device_auto(a.device)
    amp = dev.type == "cuda"
    model_dir, perf_dir = HERE / "Models" / a.version, HERE / "Model_Performance" / a.version
    model_dir.mkdir(parents=True, exist_ok=True)
    perf_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    freq = D.load_freq(a.data)
    Xtr, mtr = D.read_split(a.data, "train", a.max_train, a.seed)
    Xva, mva = D.read_split(a.data, "val")
    Xte, mte = D.read_split(a.data, "test")
    Xtr, Xva, Xte = (D.preprocess(X, freq, a.input) for X in (Xtr, Xva, Xte))
    ytr, yva, yte = (m["P"].to_numpy(np.float32) for m in (mtr, mva, mte))
    print(f"data: train {len(ytr):,}  val {len(yva):,}  test {len(yte):,}  input {a.input} {Xtr.shape[1:]}  "
          f"({time.time() - t0:.0f} s)  device {dev}", flush=True)
    print(f"  train segments {sorted(mtr['segment'].unique())}\n  val {sorted(mva['segment'].unique())}  "
          f"test {sorted(mte['segment'].unique())}", flush=True)

    pin = dev.type == "cuda"
    tl = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=a.batch_size,
                    shuffle=True, pin_memory=pin, num_workers=a.workers, drop_last=True)
    vl = DataLoader(TensorDataset(torch.from_numpy(Xva), torch.from_numpy(yva)), batch_size=4096, pin_memory=pin)

    model = build_model(a.model, D.n_channels(a.input)).to(dev)
    print(f"model {a.model}: {sum(p.numel() for p in model.parameters()):,} parameters", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    cfg = vars(a) | {"in_channels": D.n_channels(a.input), "n_train": int(len(ytr))}
    (model_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))

    ck = model_dir / "checkpoint.pth"
    start, best, bad, hist = 0, float("inf"), 0, []
    if ck.exists():
        c = torch.load(ck, map_location=dev, weights_only=False)
        model.load_state_dict(c["model"])
        opt.load_state_dict(c["opt"])
        sched.load_state_dict(c["sched"])
        start, best, bad, hist = c["epoch"], c["best"], c["bad"], c["hist"]
        print(f"resuming at epoch {start}", flush=True)

    for ep in range(start, a.epochs):
        te = time.time()
        nll = ep >= a.warmup
        tr_loss, tr_mae = run_epoch(model, tl, dev, opt, nll, amp)
        va_loss, va_mae = run_epoch(model, vl, dev, None, nll, amp)
        sched.step()
        hist.append({"epoch": ep + 1, "train_loss": tr_loss, "val_loss": va_loss, "train_mae": tr_mae,
                     "val_mae": va_mae, "lr": opt.param_groups[0]["lr"], "seconds": time.time() - te})
        improved = va_mae < best
        if improved:
            best, bad = va_mae, 0
            torch.save(model.state_dict(), model_dir / "best_model.pth")
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "epoch": ep + 1, "best": best, "bad": bad, "hist": hist}, ck)
        print(f"epoch {ep + 1:03d} | train loss {tr_loss:.5f} mae {tr_mae:.4f} | val loss {va_loss:.5f} "
              f"mae {va_mae:.4f}{' *' if improved else ''} | lr {opt.param_groups[0]['lr']:.1e} | "
              f"{time.time() - te:.0f} s", flush=True)
        if bad >= a.patience:
            print(f"early stop: val MAE has not improved for {a.patience} epochs", flush=True)
            break
    pd.DataFrame(hist).to_csv(perf_dir / "loss.csv", index=False)

    model.load_state_dict(torch.load(model_dir / "best_model.pth", map_location=dev, weights_only=True))
    for name, X, m in (("val", Xva, mva), ("test", Xte, mte)):
        p, s = predict(model, X, dev, amp=amp)
        E.report(m.assign(P_pred=p, sigma_pred=s), perf_dir, f"mc_{name}", title=f"{a.version}: MC {name} (held-out tunes)")
    E.plot_loss(perf_dir / "loss.csv", perf_dir / "loss.png")
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {model_dir}, {perf_dir}", flush=True)


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    main()
