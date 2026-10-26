# -*- coding: utf-8 -*-
"""Zero-training baseline evaluation on the SAME KonIQ-10k test partition
used by train_koniq.py (Section VI-B/C protocol).

Baselines:
  * NIQE, BRISQUE, PIQE          (classical NR-IQA; pyiqa reference impl.)
  * PCoE (paper Section VI-B):   fraction of pixels that are saturated
                                 (any channel < 5 or > 250 in 8-bit) AND
                                 outside a 3-px-dilated Sobel-Otsu edge mask,
                                 mapped to annoyance polarity (higher = worse).
Score polarity is normalized to 'higher = better quality' before SRCC/KRCC by
documented sign flips only (no learned calibration): metrics computed on raw
scores with the documented polarity; PLCC/RMSE use the 4PL fitted on calib.

Usage:
  python eval_baselines_koniq.py --data-root data/koniq10k --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch

from train_koniq import load_koniq_meta, source_disjoint_split
from eatqi.metrics import evaluate_with_calibration


def pcoe_score(img_u8: np.ndarray) -> float:
    """Clipping-outside-edge fraction; HIGHER = WORSE (severity polarity)."""
    from scipy import ndimage
    sat = (img_u8 < 5) | (img_u8 > 250)
    sat_any = sat.any(axis=2)
    luma = img_u8 @ np.array([0.299, 0.587, 0.114])
    mag = np.hypot(ndimage.sobel(luma, 0), ndimage.sobel(luma, 1))
    thr = _otsu(mag)
    edges = mag > thr
    band = ndimage.binary_dilation(edges, iterations=3)
    outside = sat_any & ~band
    return float(outside.mean())


def _otsu(mag: np.ndarray) -> float:
    from skimage.filters import threshold_otsu
    return threshold_otsu(mag)


def iter_images(items, resize):
    import cv2
    for it in items:
        bgr = cv2.imread(it.path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(full, resize, interpolation=cv2.INTER_AREA)
        yield it, full, small


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=os.path.join("data", "koniq10k"))
    ap.add_argument("--out", default=os.path.join("..", "results"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["test", "all"], default="test")
    ap.add_argument("--resize", type=int, nargs=2, default=(256, 192))
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import pyiqa
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    items = load_koniq_meta(args.data_root)
    parts = source_disjoint_split(items, seed=args.seed)
    subset = parts["test"] if args.split == "test" else items
    calib_items = parts["calib"]   # official disjoint 4PL-calibration partition
    print(f"evaluating {len(subset)} images ({args.split}), "
          f"calib={len(calib_items)}, device={device}")

    metric_names = ["niqe", "brisque", "piqe", "topiq_nr"]
    metrics = {n: pyiqa.create_metric(n, device=device) for n in metric_names}

    # per-metric polarity: NIQE/BRISQUE/PIQE are lower-is-better -> flip to
    # quality polarity; TOPIQ-NR is HIGHER-is-better (KonIQ-trained quality)
    # -> keep as is.
    polarity_flip = {}
    for n, m in metrics.items():
        polarity_flip[n] = bool(getattr(m, "lower_better", True))

    rows = []
    for it, full, small in iter_images(subset, tuple(args.resize)):
        t = torch.from_numpy(small.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
        rec = {"name": it.source_id, "mos": it.score}
        for n, m in metrics.items():
            with torch.no_grad():
                v = float(m(t).item())
            # same per-metric polarity rule as the calib pass below
            rec[n] = -v if polarity_flip[n] else v
        rec["pcoe_clip"] = pcoe_score(full)     # severity polarity
        rows.append(rec)

    # 4PL fitted on the official CALIB partition, evaluated on TEST (VI-C).
    mos = np.array([r["mos"] for r in rows])          # test raw MOS [1,100]
    calib_scores = np.array([it.score for it in calib_items])
    lo, hi = float(calib_scores.min()), float(calib_scores.max())

    def score_row(full, small):
        t = torch.from_numpy(small.astype(np.float32) / 255.0) \
              .permute(2, 0, 1).unsqueeze(0).to(device)
        rec = {}
        for n, m in metrics.items():
            with torch.no_grad():
                v = float(m(t).item())
            rec[n] = -v if polarity_flip[n] else v     # quality polarity
        rec["pcoe_clip"] = pcoe_score(full)            # severity polarity
        return rec

    import cv2
    calib_rows = []
    for it in calib_items:
        bgr = cv2.imread(it.path, cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        full = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        small = cv2.resize(full, tuple(args.resize), interpolation=cv2.INTER_AREA)
        rec = score_row(full, small)
        rec["name"], rec["mos"] = it.source_id, it.score
        calib_rows.append(rec)

    mos_c01 = (np.array([r["mos"] for r in calib_rows]) - lo) / (hi - lo)
    mos01 = (mos - lo) / (hi - lo)

    results = {}
    for n in metric_names + ["pcoe_clip"]:
        pred_c = np.array([r[n] for r in calib_rows])
        pred_t = np.array([r[n] for r in rows])
        # severity-oriented target for the PCoE descriptor; quality target else
        target_c = mos_c01 if n != "pcoe_clip" else 1.0 - mos_c01
        target_t = mos01 if n != "pcoe_clip" else 1.0 - mos01
        res = evaluate_with_calibration(pred_c, target_c, pred_t, target_t)
        res["logistic_params"] = list(res["logistic_params"])
        res["polarity"] = ("higher=better quality" if n != "pcoe_clip"
                           else "higher=worse (severity; target=1-MOS01)")
        results[n] = res
        print("[final]", n,
              {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})

    out = {"protocol": {"n_images": len(rows), "seed": args.seed,
                        "split": args.split,
                        "resize_for_deep": list(args.resize),
                        "note": "same test partition as train_koniq.py seed"},
           "results": results,
           "rows": rows}
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"koniq_baselines_seed{args.seed}_{args.split}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved ->", os.path.normpath(path))


if __name__ == "__main__":
    main()
