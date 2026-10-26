# -*- coding: utf-8 -*-
"""Train the E-AQI prototype on KonIQ-10k (quality-transfer task, Section VI-B)
and evaluate against zero-shot baselines with the Section VI-C protocol.

Honest scope (matches the manuscript):
  * KonIQ-10k MOS is a GENERAL quality target, not a clipping-annoyance
    target; results are reported as quality-transfer performance.
  * Split: source-disjoint train/calib/test at the image level (KonIQ images
    are distinct sources), seeds = 3 (Section VI-C requirement).
  * Loss: MSE + pairwise rank penalty on eligible pairs (y_i > y_j + 0.05),
    Eq. 15 with lambda_r = 0.5 (aux heads disabled for this task).
  * Metrics: SRCC/KRCC on raw scores; PLCC/RMSE after 4PL fitted on the
    disjoint calib partition.

Usage:
  python train_koniq.py --data-root data/koniq10k --epochs 12 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as Fn
from torch.utils.data import Dataset, DataLoader

from eatqi.model import PrototypeEAQI
from eatqi.metrics import evaluate_with_calibration, srcc, krcc

# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------


class KonIQDataset(Dataset):
    """KonIQ-10k 512x384 images with MOS targets (higher is better).

    Expected layout (official release):
      <root>/512x384/*.jpg
      <root>/koniq10k_distributions_sets.csv   (file_name, MOS, MOS_zscore, ...)
    """

    def __init__(self, items, resize=(256, 192), train: bool = True):
        self.items = items
        self.resize = resize
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2  # imported lazily so the dataset stays picklable on Windows
        it = self.items[i]
        bgr = cv2.imread(it.path, cv2.IMREAD_COLOR)
        if bgr is None:
            raise FileNotFoundError(it.path)
        img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if self.resize:
            img = cv2.resize(img, self.resize, interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
        y = torch.tensor(float(it.score), dtype=torch.float32)
        return x, y


from dataclasses import dataclass


@dataclass
class Item:
    """One dataset record (module-level so Windows workers can pickle it)."""
    path: str
    score: float
    source_id: str
    score_raw: float = 0.0


def load_koniq_meta(root: str):
    import csv
    meta = None
    for cand in ("koniq10k_distributions_sets.csv",
                 os.path.join("metadata", "koniq10k_distributions_sets.csv")):
        p = os.path.join(root, cand)
        if os.path.exists(p):
            meta = p
            break
    if meta is None:
        raise FileNotFoundError(
            f"koniq10k_distributions_sets.csv not found under {root}")
    imgdir = None
    for cand in ("512x384", "images", "1024x768"):
        p = os.path.join(root, cand)
        if os.path.isdir(p):
            imgdir = p
            break
    if imgdir is None:
        raise FileNotFoundError(f"image directory not found under {root}")

    items = []
    with open(meta, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("image_name") or row.get("file_name")
            if name is None:
                continue
            path = os.path.join(imgdir, name)
            if not os.path.exists(path):
                continue
            # this release: MOS in [0,100] (higher better), 'set' column
            # marks training/validation/test; we ignore their split and
            # build our own source-disjoint split (Section VI-A)
            score = float(row["MOS"])
            items.append(Item(path, score, name))
    if not items:
        raise RuntimeError("no KonIQ rows matched an existing image file")
    return items


def source_disjoint_split(items, train_frac=0.7, calib_frac=0.1, seed=0):
    rng = np.random.default_rng(seed)
    names = sorted({it.source_id for it in items})
    perm = rng.permutation(len(names))
    n_tr = int(len(names) * train_frac)
    n_ca = int(len(names) * calib_frac)
    tr = {names[i] for i in perm[:n_tr]}
    ca = {names[i] for i in perm[n_tr:n_tr + n_ca]}
    parts = {"train": [], "calib": [], "test": []}
    for it in items:
        (parts["train"] if it.source_id in tr
         else parts["calib"] if it.source_id in ca else parts["test"]).append(it)
    return parts


# --------------------------------------------------------------------------
# loss (Eq. 15, aux terms disabled for this task)
# --------------------------------------------------------------------------

def loss_mse_rank(s, y, margin=0.1, eligibility=0.05, lam_r=0.5):
    mse = Fn.mse_loss(s, y)
    n = s.numel()
    if n < 2:
        return mse + torch.zeros((), device=s.device)
    diff = y.view(-1, 1) - y.view(1, -1)          # y_i - y_j
    mask = (diff > eligibility).float()
    k = mask.sum()
    if k.item() < 1:
        return mse
    si = s.view(-1, 1)
    rank = (torch.relu(margin - (si - si.t())) * mask).sum() / k
    return mse + lam_r * rank


# --------------------------------------------------------------------------
# eval helpers
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    preds, targets = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        s = model(x)["severity"]
        preds.append(s.float().cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(preds), np.concatenate(targets)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run_seed(seed: int, args, items, device) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    parts = source_disjoint_split(items, seed=seed)
    print(f"  split sizes: train={len(parts['train'])} "
          f"calib={len(parts['calib'])} test={len(parts['test'])}")

    # normalize targets to [0,1] using TRAIN statistics only; keep the raw
    # MOS scale for calibrated PLCC/RMSE reporting (Section VI-C)
    tr_scores = np.array([it.score for it in parts["train"]])
    lo, hi = float(tr_scores.min()), float(tr_scores.max())
    def norm(v):
        return (v - lo) / (hi - lo)
    for part in parts.values():
        for it in part:
            it.score = float(norm(it.score))

    dl_tr = DataLoader(KonIQDataset(parts["train"], args.resize, train=True),
                       batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True,
                       persistent_workers=args.workers > 0)
    dl_ca = DataLoader(KonIQDataset(parts["calib"], args.resize, train=False),
                       batch_size=args.batch, num_workers=2, pin_memory=True)
    dl_te = DataLoader(KonIQDataset(parts["test"], args.resize, train=False),
                       batch_size=args.batch, num_workers=2, pin_memory=True)

    model = PrototypeEAQI(aux_heads=False).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            s = model(x)["severity"]
            loss = loss_mse_rank(s, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach()); nb += 1
        sched.step()
        print(f"  seed {seed} epoch {ep + 1}/{args.epochs} "
              f"loss={tot / max(nb, 1):.5f} ({time.time() - t0:.0f}s)")

    p_cal, t_cal = predict(model, dl_ca, device)
    p_te, t_te = predict(model, dl_te, device)
    res = evaluate_with_calibration(p_cal, t_cal, p_te, t_te)
    if args.save_checkpoint:
        ckpt_path = os.path.join(args.out, f"eaqi_koniq_seed{seed}.pth")
        torch.save({"state_dict": model.state_dict(),
                    "seed": seed,
                    "resize": list(args.resize),
                    "target_scale": {"raw_min": lo, "raw_max": hi},
                    "metrics": {k: res[k] for k in ("srcc", "krcc", "plcc", "rmse")}},
                   ckpt_path)
        print("   checkpoint ->", os.path.normpath(ckpt_path))
    res["logistic_params"] = list(res["logistic_params"])
    res["n_params_M"] = round(n_params / 1e6, 4)
    res["train_seconds"] = round(time.time() - t0, 1)
    res["target_scale"] = {"raw_min": lo, "raw_max": hi,
                           "note": "SRCC/KRCC/PLCC/RMSE on normalized [0,1] targets"}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join("data", "koniq10k"))
    ap.add_argument("--out", default=os.path.join("..", "results"))
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resize", type=int, nargs=2, default=(256, 192))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--save-checkpoint", action="store_true",
                   help="save the trained model weights per seed")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    items = load_koniq_meta(args.data_root)
    print(f"KonIQ-10k: {len(items)} images; device={device}")

    per_seed = {}
    for seed in args.seeds:
        print(f"== seed {seed} ==")
        per_seed[seed] = run_seed(seed, args, items, device)
        print("  ", {k: round(v, 4) if isinstance(v, float) else v
                     for k, v in per_seed[seed].items() if k != "logistic_params"})

    keys = ["srcc", "krcc", "plcc", "rmse"]
    summary = {"n_seeds": len(args.seeds), "args": vars(args),
               "per_seed": {str(k): v for k, v in per_seed.items()}}
    for k in keys:
        vals = np.array([per_seed[s][k] for s in args.seeds])
        summary[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}

    os.makedirs(args.out, exist_ok=True)
    out = os.path.join(args.out, "koniq_eaqi_results.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("saved ->", os.path.normpath(out))
    print(json.dumps({k: summary[k] for k in keys}, indent=2))


if __name__ == "__main__":
    main()
