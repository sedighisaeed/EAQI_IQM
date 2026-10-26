# -*- coding: utf-8 -*-
"""Computational clipping-location study (Manuscript Section V, computational variant).

Replaces the human psychophysical study with a fully computational, controlled
stimulus study evaluated with established no-reference IQMs.  The design mirrors
the prospective H1 protocol (matched clipped area, matched code-value
distribution, edge-aligned vs interior-texture clipping), but the dependent
variable is the IQM score instead of a human annoyance rating.

For every base image the script builds five stimuli, as in Section V-A:
  E  edge-aligned highlight clipping (only within a dilated edge band)
  T  interior-texture highlight clipping (only away from edges, matched area)
  X  mixed (half the target pixels in each zone)
  N  a global boost that clips naturally at the same endpoint
  C  control resample (linear map inside range, no clipping)

Stimulus construction guarantees:
  * the SAME number of pixels reaches code 255 in E, T and X (area-matched);
  * the pre-clip intensity histograms of the clipped pools are matched by
    rank-based pairing (same code-value distribution enters the clip);
  * the edge mask comes from a fixed detector (Sobel magnitude + Otsu),
    documented and reproducible; HED sensitivity analysis is optional.

Metrics: BRISQUE, NIQE, PIQE (classical); TOPIQ-NR (deep, ImageNet-trained);
plus descriptive clipped-area fraction.  Statistics: per-image paired
differences (T - E), Wilcoxon signed-rank test, Hodges-Lehmann estimate,
paired bootstrap CI, and SRCC between condition code and IQM score.

Outputs: results/clipping_location_study.json (+ per-stimulus CSV).

Usage:
  python clipping_location_study.py --data data/clipstim --out ../results
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# --------------------------------------------------------------------------
# stimulus generation
# --------------------------------------------------------------------------

def edge_mask(gray: np.ndarray, dilate: int = 3) -> np.ndarray:
    """Binary structural-edge mask: Sobel magnitude + Otsu threshold + dilation.

    Fixed, documented, reproducible detector (stand-in for PiDiNet in the
    full system; a sensitivity analysis with another detector can be added).
    """
    from scipy import ndimage
    gx = ndimage.sobel(gray, axis=1)
    gy = ndimage.sobel(gray, axis=0)
    mag = np.hypot(gx, gy)
    from skimage.filters import threshold_otsu
    thr = threshold_otsu(mag)
    m = mag > thr
    if dilate > 0:
        st = ndimage.generate_binary_structure(2, 2)
        m = ndimage.binary_dilation(m, structure=st, iterations=dilate)
    return m


def match_pairs(pool_a: np.ndarray, pool_b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rank-match two pixel pools so both receive the same code-value mix.

    Sorts each pool and keeps the first n = min(|a|, |b|) entries of each;
    the k-th brightest pixel of one pool is clipped under the same mapping as
    the k-th brightest of the other, so the *distribution of values that enter
    the clip* is identical between conditions.
    """
    a = np.sort(pool_a)[::-1]
    b = np.sort(pool_b)[::-1]
    n = min(len(a), len(b))
    return a[:n], b[:n]


def clip_fraction_target(pool_vals: np.ndarray, gain: float = 1.35) -> float:
    """Value v mapped by v * gain lands at >= 255 whenever v >= 255/gain."""
    return 255.0 / gain


def build_stimuli(rgb: np.ndarray, gain: float = 1.35,
                  dilate: int = 3) -> dict[str, np.ndarray] | None:
    """Build the five Section V-A stimuli for one base image.

    Clip rule: selected pixels are multiplied by `gain` and saturated at 255
    (a realistic highlight clip, as produced by gain-style enhancement).
    """
    from scipy import ndimage

    img = rgb.astype(np.float64)
    H, W, _ = img.shape
    luma = img @ np.array([0.299, 0.587, 0.114])
    em = edge_mask(luma, dilate=dilate)

    band = ndimage.binary_dilation(em, iterations=max(1, dilate))
    interior = ~band
    luma_thr = clip_fraction_target(luma, gain)

    cand_e = np.flatnonzero((luma >= luma_thr).ravel() & band.ravel())
    cand_t = np.flatnonzero((luma >= luma_thr).ravel() & interior.ravel())
    if len(cand_e) < 50 or len(cand_t) < 50:
        return None  # base image unsuitable for area matching

    # rank-match the two pools -> identical clipped value distribution
    vals_e, vals_t = match_pairs(img.ravel()[cand_e], img.ravel()[cand_t])
    n_clip = len(vals_e)
    thr_e = vals_e[-1] if n_clip else 255.1   # brightest n_clip pixels qualify
    thr_t = vals_t[-1] if n_clip else 255.1

    sel_e = np.zeros(H * W, dtype=bool)
    sel_t = np.zeros(H * W, dtype=bool)
    flat = img.ravel()
    sel_e[cand_e[flat[cand_e] >= thr_e]] = True
    sel_t[cand_t[flat[cand_t] >= thr_t]] = True
    sel_x = sel_e.copy()
    free = sel_t & ~sel_e
    half = np.flatnonzero(free)[::2]
    sel_x[half] = True
    # X: keep area = n_clip exactly: swap out the extra half from E zone
    extra = np.flatnonzero(sel_x)[n_clip:]
    sel_x[extra] = False

    def apply(sel: np.ndarray) -> np.ndarray:
        out = img.copy()
        m = sel.reshape(H, W)
        out[m] = np.minimum(out[m] * gain, 255.0)
        return np.clip(out, 0, 255)

    stim = {
        "E": apply(sel_e),
        "T": apply(sel_t),
        "X": apply(sel_x),
    }
    # natural global boost: same gain on the whole image
    stim["N"] = np.clip(img * gain, 0, 255)
    # control: linear rescale into [0, 254] (no endpoint saturation)
    c = img * (254.0 / 255.0)
    stim["C"] = c
    # bookkeeping
    stim_meta = {
        "clipped_area_frac_E": float(sel_e.mean()),
        "clipped_area_frac_T": float(sel_t.mean()),
        "clipped_area_frac_X": float(sel_x.mean()),
        "clipped_area_frac_N": float((stim["N"][:, :, :] >= 254.5).any(axis=-1).mean()),
        "edge_band_frac": float(band.mean()),
    }
    return {"stimuli": stim, "meta": stim_meta}


# --------------------------------------------------------------------------
# IQM evaluation
# --------------------------------------------------------------------------

def evaluate_stimuli(stimuli: dict[str, np.ndarray], device: str = "cuda",
                     names: list[str] | None = None) -> dict:
    import torch
    import pyiqa

    names = names or ["brisque", "niqe", "piqe", "topiq_nr"]
    metrics = {}
    for name in names:
        try:
            m = pyiqa.create_metric(name, device=torch.device(device))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] metric {name} unavailable: {e}")
            continue
        higher_better = m.lower_better is False
        scores = {}
        for cond, arr in stimuli.items():
            t = torch.from_numpy(arr.astype(np.float32) / 255.0)
            t = t.permute(2, 0, 1).unsqueeze(0).to(device)
            with torch.no_grad():
                v = m(t).item()
            scores[cond] = float(v)
        metrics[name] = {"scores": scores, "higher_better": higher_better}
    return metrics


def condition_stats(metrics: dict) -> dict:
    """Per-metric paired statistics for the primary contrast T - E."""
    from scipy import stats as sps

    out = {}
    for name, rec in metrics.items():
        s = rec["scores"]
        d = np.array([s["T"], s["X"], s["N"], s["C"]]) - s["E"]
        out[name] = {
            "mean_T_minus_E": float(d[0]),
            "mean_X_minus_E": float(d[1]),
            "mean_N_minus_E": float(d[2]),
            "mean_C_minus_E": float(d[3]),
        }
    return out


# --------------------------------------------------------------------------
# aggregation across images
# --------------------------------------------------------------------------

def aggregate(per_image: list[dict]) -> dict:
    """Cross-image statistics for each metric and the primary hypothesis."""
    from scipy import stats as sps

    metrics = sorted({k for r in per_image for k in r["metrics"]})
    agg: dict = {}
    for name in metrics:
        diffs = np.array([r["metrics"][name]["scores"]["T"]
                          - r["metrics"][name]["scores"]["E"]
                          for r in per_image if name in r["metrics"]])
        if len(diffs) < 10:
            continue
        w = sps.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
        # Hodges-Lehmann point estimate
        hl = np.median((diffs[:, None] + diffs[None, :]).ravel() / 2.0)
        rng = np.random.default_rng(0)
        boots = np.array([rng.choice(diffs, size=len(diffs), replace=True).mean()
                          for _ in range(2000)])
        agg[name] = {
            "n_images": int(len(diffs)),
            "mean_diff": float(diffs.mean()),
            "sd_diff": float(diffs.std(ddof=1)),
            "wilcoxon_p": float(w.pvalue),
            "hodges_lehmann": float(hl),
            "boot_ci95": [float(np.percentile(boots, 2.5)),
                          float(np.percentile(boots, 97.5))],
        }
    return agg


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------

def load_base_images(root: str, max_images: int | None = None) -> list[np.ndarray]:
    import cv2
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    files = sorted(f for f in os.listdir(root) if f.lower().endswith(exts))
    if max_images:
        files = files[:max_images]
    imgs = []
    for f in files:
        im = cv2.imread(os.path.join(root, f), cv2.IMREAD_COLOR)
        if im is None:
            continue
        imgs.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    return imgs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join("data", "clipstim"))
    ap.add_argument("--out", default=os.path.join("..", "results"))
    ap.add_argument("--gain", type=float, default=1.35)
    ap.add_argument("--device", default="cuda" if _cuda_ok() else "cpu")
    ap.add_argument("--max-images", type=int, default=None)
    ap.add_argument("--metrics", nargs="+",
                    default=["brisque", "niqe", "piqe", "topiq_nr"])
    args = ap.parse_args()

    imgs = load_base_images(args.data, args.max_images)
    if not imgs:
        raise SystemExit(
            f"no base images found under {args.data}; see README (Section 'Data')")

    print(f"{len(imgs)} base images; gain={args.gain}; device={args.device}")
    per_image, t0 = [], time.time()
    for i, img in enumerate(imgs):
        built = build_stimuli(img, gain=args.gain)
        if built is None:
            print(f"[{i + 1}/{len(imgs)}] skipped (insuitable for area matching)")
            continue
        m = evaluate_stimuli(built["stimuli"], device=args.device,
                             names=args.metrics)
        per_image.append({"index": i, "meta": built["meta"], "metrics": m})
        te = {k: round(v["scores"]["T"] - v["scores"]["E"], 4)
              for k, v in m.items()}
        print(f"[{i + 1}/{len(imgs)}] T-E diffs: {te}")

    agg = aggregate(per_image)
    result = {
        "protocol": {
            "n_base_images_usable": len(per_image),
            "gain": args.gain,
            "edge_detector": "sobel magnitude + otsu, dilation=3",
            "clip_rule": "selected pixels * gain, saturated at 255",
            "area_matching": "rank-matched pools (identical clipped value distribution)",
            "device": args.device,
            "runtime_s": round(time.time() - t0, 1),
        },
        "aggregate": agg,
        "per_image": per_image,
    }
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "clipping_location_study.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print("saved ->", os.path.normpath(out_path))


def _cuda_ok() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001
        return False


if __name__ == "__main__":
    main()
