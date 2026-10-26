# -*- coding: utf-8 -*-
"""Enhancement clipping-response study: how do IQMs respond to genuine
enhancement-induced clipping at matched brightness?

Design: for every base image we apply a gamma-corrected gain enhancement with
an increasing target-brightness parameter.  Stronger enhancement -> more
highlight clipping -> typically higher IQM quality scores, but at the cost of
clipped highlights.  We measure:

  * clipped-area fraction per enhancement level (ground-truth clipping dose);
  * IQM score per level (BRISQUE / NIQE / PIQE / TOPIQ-NR);

and report the SRCC between clipping dose and each IQM across images/levels,
i.e. the correlation of the 'clipping fraction' descriptor of the paper with
the IQM verdicts.  This quantifies whether IQMs *reward* aggressive clipping
(a validity problem for using them as enhancement judges) and motivates a
clipping-aware assessor.

Outputs: results/enhancement_clip_response.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def enhance_gain_gamma(img01: np.ndarray, gamma: float, gain: float) -> np.ndarray:
    """Gamma-then-gain enhancement; highlight clipping arises naturally."""
    x = np.power(np.clip(img01, 0.0, 1.0), gamma) * gain
    return np.clip(x, 0.0, 1.0)


def clipped_fraction(img01: np.ndarray, level: float = 1.0) -> float:
    """Fraction of pixels with at least one channel at the code-value cap."""
    m = (img01 >= level - 1.5 / 255.0).any(axis=-1)
    return float(m.mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "clipstim"))
    ap.add_argument("--out", default=os.path.join("..", "results"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--levels", type=int, default=8)
    args = ap.parse_args()

    import cv2
    import torch
    import pyiqa

    root = args.data
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = sorted(f for f in os.listdir(root) if f.lower().endswith(exts))
    if args.max_images:
        files = files[:args.max_images]
    if not files:
        raise SystemExit(f"no base images under {root}")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    metric_names = ["brisque", "niqe", "piqe", "topiq_nr"]
    metrics = {n: pyiqa.create_metric(n, device=device) for n in metric_names}

    # enhancement levels: gamma from 1.0 (identity) to 0.45 (strong lift)
    gammas = np.linspace(1.0, 0.45, args.levels)
    gain = 1.0

    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(root, f), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        img01 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float64) / 255.0
        for g in gammas:
            enh = enhance_gain_gamma(img01, float(g), gain)
            cf = clipped_fraction(enh)
            scores = {}
            for name, m in metrics.items():
                t = torch.from_numpy(enh.astype(np.float32)).permute(2, 0, 1).unsqueeze(0).to(device)
                with torch.no_grad():
                    scores[name] = float(m(t).item())
            rows.append({"image": f, "gamma": float(g), "clip_frac": cf,
                         "scores": scores})
        print(f"[{i + 1}/{len(files)}] {f}: gamma sweep done")

    # ---- analysis -----------------------------------------------------
    result_rows = []
    clip = np.array([r["clip_frac"] for r in rows])
    for name in metric_names:
        s = np.array([r["scores"][name] for r in rows])
        from scipy import stats as sps
        rho, p = sps.spearmanr(clip, s)
        result_rows.append({
            "metric": name,
            "srcc_clipfrac_vs_score": float(rho),
            "p_value": float(p),
            "n_obs": len(rows),
        })

    # per-image curves: does score improve monotonically with clipping?
    per_image_trend = []
    for f in sorted({r["image"] for r in rows}):
        rr = sorted((r for r in rows if r["image"] == f), key=lambda r: r["gamma"])
        entry = {"image": f, "clip_frac": [r["clip_frac"] for r in rr]}
        for name in metric_names:
            entry[f"score_{name}"] = [r["scores"][name] for r in rr]
        per_image_trend.append(entry)

    out = {
        "protocol": {
            "n_images": len(files),
            "gamma_levels": [float(g) for g in gammas],
            "gain": gain,
            "enhancement": "gamma lift then identity gain, clip at 1.0",
            "metrics": metric_names,
            "device": str(device),
            "runtime_s": round(time.time() - t0, 1),
        },
        "srcc_clipfrac_vs_score": result_rows,
        "rows": rows,
        "per_image_trend": per_image_trend,
    }
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "enhancement_clip_response.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print("saved ->", os.path.normpath(path))
    print(json.dumps(result_rows, indent=2))


if __name__ == "__main__":
    main()
