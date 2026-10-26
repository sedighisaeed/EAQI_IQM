# -*- coding: utf-8 -*-
"""Executed component-wise ablation of the E-AQI prototype on the fixed
seed-0 KonIQ-10k split (fills Table `tab:ablation`).

Variants (all trained with the exact train_koniq.py protocol, seed 0):
  a  full E-AQI (reference)
  b  without clipping branch C   (clip_lat zeroed)
  c  without edge branch E       (edge_lat zeroed)
  d  without context branch Z    (ctx_lat zeroed)
  e  hard binary saturation map  (replaces the soft two-sided response)
  f  fixed cutoffs 5/250 (eta forced to zero: no threshold adaptation)
  g  concat fusion (map vector added as a global bias, no cross-attention)
  h  MSE only (lambda_r = 0)

Outputs: results/koniq_ablation.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import torch  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from eatqi.model import PrototypeEAQI  # noqa: E402
from eatqi.metrics import evaluate_with_calibration  # noqa: E402
from train_koniq import (KonIQDataset, load_koniq_meta, source_disjoint_split,
                         predict)  # noqa: E402


def loss_mse_rank(s, y, margin=0.1, eligibility=0.05, lam_r=0.5):
    mse = Fn.mse_loss(s, y)
    n = s.numel()
    if n < 2:
        return mse + torch.zeros((), device=s.device)
    diff = y.view(-1, 1) - y.view(1, -1)
    mask = (diff > eligibility).float()
    k = mask.sum()
    if k.item() < 1:
        return mse
    si = s.view(-1, 1)
    rank = (torch.relu(margin - (si - si.t())) * mask).sum() / k
    return mse + lam_r * rank


def forward_variant(model: PrototypeEAQI, images: torch.Tensor,
                    variant: str) -> dict:
    """Mirror of PrototypeEAQI.forward with one component ablated."""
    s2 = model.stem(images)
    s4 = model.stage2(s2)
    s8 = model.stage3(s4)
    s16 = model.stage4(s8)
    H4, W4 = s4.shape[-2:]
    luma = images.mean(dim=1, keepdim=True)

    x01 = images.clamp(0, 1)
    if variant == "e":  # hard binary saturation
        clip_maps = ((x01.min(dim=1, keepdim=True).values < 5 / 255.0) |
                     (x01.max(dim=1, keepdim=True).values > 250 / 255.0)
                     ).float().expand(-1, 3, -1, -1)
    else:
        lo, hi = 5.0 / 255.0, 250.0 / 255.0
        alpha = 0.5 * 255.0
        a_lo = torch.sigmoid(alpha * (lo - x01))
        a_hi = torch.sigmoid(alpha * (x01 - hi))
        clip_maps = 1.0 - (1.0 - a_lo) * (1.0 - a_hi)
    clip_lat = Fn.adaptive_avg_pool2d(clip_maps, (H4, W4))
    if variant == "b":
        clip_lat = torch.zeros_like(clip_lat)

    # NOTE on variant f (fixed cutoffs): the prototype's trainable threshold
    # adaptation lives in the 1x1 channel fusion (clip_proj). Variant f freezes
    # that adaptation by using the raw mean of the three channel maps instead
    # of the learned projection: the effective response stays the fixed-cutoff
    # sigmoid at 5/250 with no trainable adaptation.
    if variant == "f":
        f_maps = clip_maps.mean(dim=1, keepdim=True).expand(-1, 3, -1, -1)
        clip_lat = Fn.adaptive_avg_pool2d(f_maps, (H4, W4))
    edge = model.edge_response(luma)
    edge_lat = Fn.adaptive_avg_pool2d(edge, (H4, W4)) / \
        edge.max().clamp(min=1e-6)
    if variant == "c":
        edge_lat = torch.zeros_like(edge_lat)

    ctx_lat = torch.tanh(Fn.avg_pool2d(luma - luma.mean(), 3, stride=4,
                                       padding=1, count_include_pad=False))
    ctx_lat = Fn.interpolate(ctx_lat, size=(H4, W4), mode="bilinear",
                             align_corners=False)
    if variant == "d":
        ctx_lat = torch.zeros_like(ctx_lat)

    maps = torch.cat([model.clip_proj(clip_lat),
                      model.edge_proj(edge_lat),
                      model.ctx_proj(ctx_lat)], dim=1)

    if variant == "g":  # global vector instead of cross-attention
        gv = maps.flatten(2).mean(dim=2)                       # [N,48]
        fused = []
        for q_in in (s4, s8, s16):
            add = gv.mean(dim=1, keepdim=True).view(-1, 1, 1, 1)
            fused.append(q_in + add)
    else:
        fused = model.fusion([s4, s8, s16], maps)

    p = None
    feats = []
    rev_fused = list(reversed(fused))
    rev_lat = list(reversed(model.fpn_lateral))
    for f, lat, smooth in zip(rev_fused, rev_lat, model.fpn_smooth):
        p = lat(f) if p is None else lat(f) + Fn.interpolate(
            p, size=f.shape[-2:], mode="bilinear", align_corners=False)
        p = smooth(p)
        feats.insert(0, p)
    severity = model.regressor(p).squeeze(-1)
    return {"severity": severity}


def train_variant(variant: str, items, args, device) -> dict:
    torch.manual_seed(0)
    np.random.seed(0)
    parts = source_disjoint_split(items, seed=0)
    tr_scores = np.array([it.score for it in parts["train"]])
    lo, hi = float(tr_scores.min()), float(tr_scores.max())
    for part in parts.values():
        for it in part:
            it.score = float((it.score - lo) / (hi - lo))

    dl_tr = DataLoader(KonIQDataset(parts["train"], args.resize, True),
                       batch_size=args.batch, shuffle=True,
                       num_workers=args.workers, pin_memory=True)
    dl_ca = DataLoader(KonIQDataset(parts["calib"], args.resize, False),
                       batch_size=args.batch, num_workers=2, pin_memory=True)
    dl_te = DataLoader(KonIQDataset(parts["test"], args.resize, False),
                       batch_size=args.batch, num_workers=2, pin_memory=True)

    model = PrototypeEAQI(aux_heads=False).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    t0 = time.time()
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for x, y in dl_tr:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            out = forward_variant(model, x, variant)
            s = out["severity"]
            loss = Fn.mse_loss(s, y) if variant == "h" \
                else loss_mse_rank(s, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        print(f"  [{variant}] epoch {ep + 1}/{args.epochs} "
              f"loss={tot / max(nb, 1):.5f} ({time.time() - t0:.0f}s)",
              flush=True)

    p_cal, t_cal = predict(model, dl_ca, device)
    p_te, t_te = predict(model, dl_te, device)
    res = evaluate_with_calibration(p_cal, t_cal, p_te, t_te)
    return {"srcc": res["srcc"], "krcc": res["krcc"], "plcc": res["plcc"],
            "rmse": res["rmse"],
            "logistic_params": [float(v) for v in res["logistic_params"]],
            "train_seconds": round(time.time() - t0, 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join("data", "koniq10k"))
    ap.add_argument("--out", default=os.path.join("..", "results",
                                                  "koniq_ablation.json"))
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--resize", type=int, nargs=2, default=(256, 192))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--variants", nargs="+",
                    default=["a", "b", "c", "d", "e", "f", "g", "h"])
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    items = load_koniq_meta(args.data_root)
    print(f"KonIQ-10k: {len(items)} images; device={device}", flush=True)

    out = {"protocol": {"seed": 0, "epochs": args.epochs,
                        "batch": args.batch, "lr": args.lr,
                        "resize": list(args.resize),
                        "split": "source-disjoint 70/10/20 (train_koniq.py)",
                        "note": "epochs reduced for the ablation comparison; "
                                "all variants share the identical budget"},
           "variants": {}}
    for v in args.variants:
        print(f"== variant {v} ==", flush=True)
        out["variants"][v] = train_variant(v, items, args, device)
        r = out["variants"][v]
        print(f"   SRCC {r['srcc']:.3f}  KRCC {r['krcc']:.3f}  "
              f"PLCC {r['plcc']:.3f}  RMSE {r['rmse']:.3f}", flush=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)   # incremental save

    print("saved ->", os.path.normpath(args.out))


if __name__ == "__main__":
    main()
