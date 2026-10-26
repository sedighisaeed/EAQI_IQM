# -*- coding: utf-8 -*-
"""Evaluate the trained E-AQI prototype and the PCoE descriptor on the two
executed computational clipping studies (reviewer request: the paper's own
measures were absent from Studies 1 and 2).

Study 1 (clipping location): regenerate the 168 area-matched E/T/X/N/C
stimuli exactly as clipping_location_study.py did, score every stimulus with
the seed-0 E-AQI checkpoint and the PCoE clipped-area-fraction descriptor,
and compute the same paired statistics as the four published baselines.

Study 2 (clipping dose): regenerate the 480-image gamma sweep, score every
image with E-AQI/PCoE, and compute SRCC(clip_frac, score) matching the
protocol of enhancement_clip_response.py.

Outputs: results/eaqi_studies12.json (verbatim, no manual edits).
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

from clipping_location_study import build_stimuli, load_base_images  # noqa: E402
from enhancement_clip_response import clipped_fraction, enhance_gain_gamma  # noqa: E402
from eatqi.model import PrototypeEAQI  # noqa: E402

import torch  # noqa: E402


# --------------------------------------------------------------------------
# E-AQI inference
# --------------------------------------------------------------------------

def load_eaqi(ckpt_path: str, device: str) -> tuple:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = PrototypeEAQI(aux_heads=False).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@torch.no_grad()
def eaqi_score(model, rgb: np.ndarray, device: str, resize=(256, 192)) -> float:
    import cv2
    img = cv2.resize(rgb, resize, interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1)
    x = x.unsqueeze(0).to(device)
    return float(model(x)["severity"].item())


def pcoe_score(rgb: np.ndarray, dilate: int = 3) -> float:
    """Non-edge clipped-area fraction (identical rule to eval_baselines_koniq)."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    img = rgb.astype(np.float64)
    luma = img @ np.array([0.299, 0.587, 0.114])
    gx = ndimage.sobel(luma, axis=1)
    gy = ndimage.sobel(luma, axis=0)
    mag = np.hypot(gx, gy)
    thr = threshold_otsu(mag)
    edges = ndimage.binary_dilation(mag > thr, iterations=dilate)
    sat = (img.min(axis=-1) < 5) | (img.max(axis=-1) > 250)
    return float((sat & ~edges).mean())


# --------------------------------------------------------------------------
# study 1
# --------------------------------------------------------------------------

def run_study1(model, device: str, max_images: int, gain: float) -> dict:
    data = os.path.join(HERE, "data", "clipstim")
    imgs = load_base_images(data, max_images)
    per_image = []
    t0 = time.time()
    for i, img in enumerate(imgs):
        built = build_stimuli(img, gain=gain)
        if built is None:
            continue
        scores = {}
        for cond, arr in built["stimuli"].items():
            scores[cond] = {
                "eaqi": eaqi_score(model, arr, device),
                "pcoe": pcoe_score(arr),
            }
        per_image.append({"index": i, "meta": built["meta"], "scores": scores})
        print(f"[study1 {i + 1}/{len(imgs)}] "
              f"eaqi T-E={scores['T']['eaqi'] - scores['E']['eaqi']:+.4f} "
              f"pcoe T-E={scores['T']['pcoe'] - scores['E']['pcoe']:+.4f}")

    from scipy import stats as sps
    agg = {}
    for meas in ("eaqi", "pcoe"):
        d = np.array([r["scores"]["T"][meas] - r["scores"]["E"][meas]
                      for r in per_image])
        w = sps.wilcoxon(d, alternative="two-sided", zero_method="wilcox")
        hl = np.median((d[:, None] + d[None, :]).ravel() / 2.0)
        rng = np.random.default_rng(0)
        boots = np.array([rng.choice(d, size=len(d), replace=True).mean()
                          for _ in range(2000)])
        agg[meas] = {
            "n_images": int(len(d)),
            "mean_diff": float(d.mean()),
            "sd_diff": float(d.std(ddof=1)),
            "wilcoxon_p": float(w.pvalue),
            "hodges_lehmann": float(hl),
            "boot_ci95": [float(np.percentile(boots, 2.5)),
                          float(np.percentile(boots, 97.5))],
            "pct_T_gt_E": float(100.0 * np.mean(d > 0)),
            # condition means (X/N/C model verdicts, reviewer request)
            "mean_E": float(np.mean([r["scores"]["E"][meas] for r in per_image])),
            "mean_T": float(np.mean([r["scores"]["T"][meas] for r in per_image])),
            "mean_X": float(np.mean([r["scores"]["X"][meas] for r in per_image])),
            "mean_N": float(np.mean([r["scores"]["N"][meas] for r in per_image])),
            "mean_C": float(np.mean([r["scores"]["C"][meas] for r in per_image])),
        }
    return {"per_image": per_image, "aggregate": agg,
            "runtime_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------
# study 2
# --------------------------------------------------------------------------

def run_study2(model, device: str, max_images: int, levels: int) -> dict:
    import cv2
    data = os.path.join(HERE, "data", "clipstim")
    exts = (".png", ".jpg", ".jpeg")
    files = sorted(f for f in os.listdir(data) if f.lower().endswith(exts))[:max_images]
    gammas = np.linspace(1.0, 0.45, levels)
    rows = []
    t0 = time.time()
    for i, f in enumerate(files):
        bgr = cv2.imread(os.path.join(data, f), cv2.IMREAD_COLOR)
        if bgr is None:
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        for g in gammas:
            enh01 = enhance_gain_gamma(rgb.astype(np.float64) / 255.0,
                                       float(g), 1.0)
            enh8 = enh01 * 255.0
            cf = clipped_fraction(enh01)
            rows.append({"image": f, "gamma": float(g), "clip_frac": cf,
                         "eaqi": eaqi_score(model, enh8, device),
                         "pcoe": pcoe_score(enh8)})
        print(f"[study2 {i + 1}/{len(files)}] {f}")
    from scipy import stats as sps
    clip = np.array([r["clip_frac"] for r in rows])
    res = {}
    for meas in ("eaqi", "pcoe"):
        s = np.array([r[meas] for r in rows])
        rho, p = sps.spearmanr(clip, s)
        res[meas] = {"srcc_clipfrac_vs_score": float(rho),
                     "p_value": float(p), "n_obs": len(rows)}
    return {"rows": rows, "srcc_clipfrac_vs_score": res,
            "runtime_s": round(time.time() - t0, 1)}


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join("..", "results",
                                                   "eaqi_koniq_seed0.pth"))
    ap.add_argument("--out", default=os.path.join("..", "results",
                                                  "eaqi_studies12.json"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--max-images", type=int, default=240)
    ap.add_argument("--dose-images", type=int, default=60)
    ap.add_argument("--gain", type=float, default=1.35)
    ap.add_argument("--levels", type=int, default=8)
    args = ap.parse_args()

    device = args.device
    model, ckpt = load_eaqi(args.ckpt, device)
    print(f"loaded E-AQI seed {ckpt.get('seed')} "
          f"(SRCC {ckpt['metrics']['srcc']:.3f}) device={device}")

    s1 = run_study1(model, device, args.max_images, args.gain)
    print("study 1 aggregate:",
          json.dumps({k: {kk: round(vv, 5) if isinstance(vv, float) else vv
                          for kk, vv in v.items()}
                      for k, v in s1["aggregate"].items()}, indent=2))
    s2 = run_study2(model, device, args.dose_images, args.levels)
    print("study 2 SRCC:", json.dumps(s2["srcc_clipfrac_vs_score"], indent=2))

    out = {
        "protocol": {
            "checkpoint": os.path.normpath(args.ckpt),
            "seed": ckpt.get("seed"),
            "eaqi_input_resize": [256, 192],
            "pcoe_rule": "saturated (any ch <5 or >250) outside 3x-dilated "
                         "Sobel-Otsu edge mask",
            "study1_gain": args.gain,
            "study2_gamma": [1.0, 0.45], "study2_levels": args.levels,
            "device": device,
        },
        "study1_location": s1,
        "study2_dose": s2,
    }
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print("saved ->", os.path.normpath(args.out))


if __name__ == "__main__":
    main()
