# -*- coding: utf-8 -*-
"""Synthetic mathematical checks of the corrected operator
(Manuscript Section IV, Table I — the single source of truth for those numbers).

Uses ONLY artificial intensity values.  It does NOT create human ratings,
image-level MOS, network predictions, or benchmark accuracy.  These checks
assess mathematical consistency only.

Checks:
  1. default two-sided mask at codes 0 and 255  -> both endpoints activate
  2. default mask at cutoffs 5 and 250          -> sigmoid midpoint (0.5)
  3. default mask at code 128                   -> mid-gray not flagged
  4. 8-bit vs 10-bit scaled mask difference     -> algebraic scale equivalence
  5. tightened vs default mask                  -> at least as large everywhere
  6. hard max-endpoint pile-up ratio            -> reference value
  7. superseded endpoint-average ratio          -> factor-of-two demonstration
  8. smooth ratio (delta, t)                    -> agreement in the limiting case
  +  finite-difference directional derivative of the smooth ratio (diagnostic)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from eatqi.clipping import saturation_map_1c          # noqa: E402
from eatqi.surrogate import (                          # noqa: E402
    endpoint_average_ratio, hard_pileup_ratio, logsumexp,
    smooth_histogram, smooth_pileup_ratio)

ALPHA_8BIT = 0.5


def make_diagnostic_sequence() -> np.ndarray:
    """Deterministic 1,000-value sequence per Section IV: twenty values at
    the bright endpoint 255, none at zero, and a deliberately nonuniform
    interior distribution whose modal bin (code 128) contains 104 members.

    Composition (fully deterministic; no random sampling):
      - 104 values at code 128   -> modal bin, 104 members
      -  20 values at code 255   -> bright-endpoint pile-up
      - 876 values spread over codes 1..254, cycling with an
        arithmetic step so every interior bin holds at most 4 members.
    """
    # codes 1..254 excluding 128, which is reserved for the modal bin
    interior_codes = np.array([c for c in range(1, 255) if c != 128])
    base = np.tile(interior_codes, 3)                    # 3 cycles = 759
    extra = interior_codes[:876 - len(base)]             # 117 more
    mid = np.concatenate([base, extra]).astype(np.float64)
    assert len(mid) == 876
    counts = np.bincount(mid.astype(int), minlength=256)
    assert counts.max() == 4 and counts[128] == 0
    return np.concatenate([np.full(104, 128.0), mid,
                           np.full(20, 255.0)])


def main(out_path: str | None = None) -> dict:
    results = {}

    # ---- 1-3: two-sided mask behavior on every 8-bit code -------------
    # Default fixed cutoffs (5, 250): Table I rows 1-3.  With alpha = 0.5
    # per code value, both endpoint responses equal sigmoid(2.5) and the
    # cutoff codes sit at the sigmoid midpoint 0.5.
    codes = np.arange(256, dtype=np.float64)
    m_default = saturation_map_1c(codes, cutoffs=(5.0, 250.0))
    results["default_mask_at_0"] = float(m_default[0])
    results["default_mask_at_255"] = float(m_default[255])
    results["default_mask_at_cutoff_5"] = float(m_default[5])
    results["default_mask_at_cutoff_250"] = float(m_default[250])
    results["default_mask_at_128"] = float(m_default[128])

    # ---- 4: bit-depth scale equivalence -------------------------------
    rng = np.random.default_rng(7)
    img8 = rng.integers(0, 256, size=(16, 16)).astype(np.float64)
    r = 1023.0 / 255.0
    img10 = img8 * r
    m8 = saturation_map_1c(img8, max_level=255.0)
    m10 = saturation_map_1c(img10, max_level=1023.0)
    results["bitdepth_scaled_mask_max_abs_diff"] = float(np.abs(m10 - m8).max())

    # ---- 5: tightened mask dominates default --------------------------
    codes = np.arange(256, dtype=np.float64)
    m_def = saturation_map_1c(codes, cutoffs=(5.0, 250.0))
    m_tight = saturation_map_1c(codes, cutoffs=(25.0, 230.0))
    results["tightened_minus_default_min"] = float((m_tight - m_def).min())
    results["tightened_mask_at_128"] = float(m_tight[128])

    # ---- 6-8: pile-up ratios on the diagnostic sequence ---------------
    seq = make_diagnostic_sequence()
    hard = hard_pileup_ratio(seq)
    old_avg = endpoint_average_ratio(seq)
    smooth = smooth_pileup_ratio(seq, delta=0.1, t=1e-5)
    results["hard_max_endpoint_ratio"] = hard
    results["superseded_endpoint_average_ratio"] = old_avg
    results["smooth_ratio_delta0.1_t1e-5"] = smooth
    results["factor_of_two_check"] = abs(2.0 * old_avg - hard)

    # ---- finite-difference directional derivative (diagnostic) --------
    # Six-value example: only the value near the bright endpoint (251)
    # carries nonzero gradient into the smooth-max numerator at this
    # bandwidth; perturbing it raises the ratio (documented diagnostic).
    vals = np.array([12.0, 37.0, 64.0, 128.0, 200.0, 251.0], dtype=np.float64)
    h = 1e-4
    i0 = 5  # perturb the value adjacent to the 255 endpoint
    vals_p, vals_m = vals.copy(), vals.copy()
    vals_p[i0] += h
    vals_m[i0] -= h
    f0 = smooth_pileup_ratio(vals, delta=2.0, t=1e-5)
    fp = smooth_pileup_ratio(vals_p, delta=2.0, t=1e-5)
    fm = smooth_pileup_ratio(vals_m, delta=2.0, t=1e-5)
    results["fd_directional_derivative_six_value_example"] = float((fp - fm) / (2 * h))
    results["fd_eval_point"] = "value index 5 (code 251), delta=2.0, t=1e-5, h=1e-4"
    results["smooth_ratio_at_eval_point"] = float(f0)

    # ---- LSE sanity: smooth max >= arithmetic max ---------------------
    H = smooth_histogram(seq, delta=0.1)
    lse = logsumexp(H, 1e-5)
    results["lse_ge_max"] = bool(lse >= H.max() - 1e-9)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(__file__), "..", "results",
                       "synthetic_checks.json")
    res = main(out)
    print(json.dumps(res, indent=2))
    print("saved ->", os.path.normpath(out))
