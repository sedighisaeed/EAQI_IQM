# -*- coding: utf-8 -*-
"""Generate the controlled clipping stimuli for the human perceptual study.

Mirrors the executed computational study 1 (code/clipping_location_study.py)
so that the human study tests exactly the same manipulation:

  E  edge-aligned highlight clipping  (gain 1.35, inside dilated Sobel-Otsu band)
  T  interior-texture highlight clipping (gain 1.35, away from edges)
  X  mixed (half the T pixels swapped in)
  N  natural global enhancement (whole image * gain)
  C  control (linear rescale into [0, 254], no clipping)

Guarantees (identical to study 1):
  * area-matched E/T/X clipping (same number of pixels reach 255);
  * rank-matched pre-clip code-value distributions between E and T pools;
  * excluded base images (fewer than 50 candidate pixels per zone).

Usage:
  python make_stimuli.py --n-images 100 --out stimuli
Outputs:
  stimuli/<base>_<cond>.jpg  (stimulus bitmaps, max side 1024)
  stimuli/manifest.json      (full protocol record incl. clipped-area stats)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.normpath(os.path.join(HERE, "..", "code"))
sys.path.insert(0, CODE_DIR)

from clipping_location_study import build_stimuli  # noqa: E402  (single source of truth)


def load_base_images(root: str, n: int, seed: int = 7) -> list[tuple[str, np.ndarray]]:
    import cv2
    files = sorted(f for f in os.listdir(root)
                   if f.lower().endswith((".png", ".jpg", ".jpeg")))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    out = []
    for f in files:
        im = cv2.imread(os.path.join(root, f), cv2.IMREAD_COLOR)
        if im is None:
            continue
        out.append((os.path.splitext(f)[0], cv2.cvtColor(im, cv2.COLOR_BGR2RGB)))
        if len(out) >= n:
            break
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(CODE_DIR, "data", "clipstim"))
    ap.add_argument("--out", default=os.path.join(HERE, "stimuli"))
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--gain", type=float, default=1.35)
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    import cv2

    os.makedirs(args.out, exist_ok=True)
    bases = load_base_images(args.data, args.n_images)
    print(f"{len(bases)} candidate base images")

    manifest = {"protocol": {
        "n_base_target": args.n_images,
        "gain": args.gain,
        "edge_detector": "sobel magnitude + otsu, dilation=3",
        "clip_rule": "selected pixels * gain, saturated at 255",
        "area_matching": "rank-matched pools (identical clipped value distribution)",
        "conditions": ["E", "T", "X", "N", "C"],
        "rating_scale": "0-100 annoyance/severity",
        "note": "H1 pair = T vs E (area- and histogram-matched).",
    }, "stimuli": []}

    kept = 0
    for name, rgb in bases:
        h, w = rgb.shape[:2]
        scale = args.max_side / max(h, w)
        if scale < 1.0:
            rgb = cv2.resize(rgb, (int(round(w * scale)), int(round(h * scale))),
                             interpolation=cv2.INTER_AREA)
        built = build_stimuli(rgb, gain=args.gain)
        if built is None:
            print(f"  {name}: skipped (unsuitable for area matching)")
            continue
        kept += 1
        for cond, arr in built["stimuli"].items():
            out_bgr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_RGB2BGR)
            fn = f"{name}_{cond}.jpg"
            cv2.imwrite(os.path.join(args.out, fn), out_bgr,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        rec = {"base": name, "files": {c: f"{name}_{c}.jpg"
                                       for c in built["stimuli"]},
               "meta": built["meta"]}
        manifest["stimuli"].append(rec)
        print(f"  {name}: ok "
              f"(E area {built['meta']['clipped_area_frac_E']:.3%}, "
              f"T area {built['meta']['clipped_area_frac_T']:.3%})")

    manifest["protocol"]["n_base_usable"] = kept
    with open(os.path.join(args.out, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"kept {kept} base images x 5 conditions = {kept * 5} stimuli")
    print("manifest ->", os.path.join(args.out, "manifest.json"))


if __name__ == "__main__":
    main()
