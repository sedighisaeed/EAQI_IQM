# -*- coding: utf-8 -*-
"""Two-sided adaptive saturation map (Manuscript Sections III-B and III-C).

Corrected operator (replaces the earlier centered-deviation formulation):

    D_c      = max_k h_c(k) + eps                       (Eq. 1)
    p_c^-    = h_c(0) / D_c ,  p_c^+ = h_c(L) / D_c     (Eq. 2)
    l_c      = 5  + 20 * 1[p_c^- > eta]                 (Eq. 3)
    u_c      = 250 - 20 * 1[p_c^+ > eta]
    a_c^-(i) = sigmoid(alpha * (l_c - I_c(i)))          (Eq. 4)
    a_c^+(i) = sigmoid(alpha * (I_c(i) - u_c))          (Eq. 5)
    M_c(i)   = 1 - (1 - a_c^-)(1 - a_c^+)               (Eq. 6)

Bit-depth rescaling (Section III-C): with r = L/255,
    alpha' = alpha / r,  l_c' = r l_c,  u_c' = r u_c
preserves every sigmoid argument algebraically (Eq. 7-8).  Fractional
cutoffs are permitted (e.g. 8-bit 250 -> 10-bit 250*1023/255 = 1002.941).
"""

from __future__ import annotations

import numpy as np

from . import ALPHA_8BIT, DEFAULT_HI, DEFAULT_LO, ETA, TIGHT_HI, TIGHT_LO

EPS = 1e-12


def endpoint_ratios(channel: np.ndarray, max_level: float = 255.0) -> tuple[float, float]:
    """Return (p_minus, p_plus): endpoint bin probabilities over the modal bin.

    p_c^- = h_c(0) / (max_k h_c(k) + eps);   p_c^+ = h_c(L) / (max_k h_c(k) + eps)
    """
    channel = np.asarray(channel)
    levels = int(round(max_level)) + 1
    hist, _ = np.histogram(channel, bins=levels, range=(0.0, max_level))
    denom = hist.max() + EPS
    return hist[0] / denom, hist[-1] / denom


def adaptive_cutoffs(channel: np.ndarray,
                     max_level: float = 255.0,
                     eta: float = ETA,
                     ) -> tuple[float, float]:
    """One-sided independent cutoff adaptation (Eq. 3).

    Returns (l_c, u_c) in raw code-value coordinates.  The dark cutoff
    tightens upward if (and only if) the dark endpoint piles up; the bright
    cutoff tightens downward if (and only if) the bright endpoint piles up.
    """
    base_lo = DEFAULT_LO * (max_level / 255.0)
    base_hi = DEFAULT_HI * (max_level / 255.0)
    tight_lo = TIGHT_LO * (max_level / 255.0)
    tight_hi = TIGHT_HI * (max_level / 255.0)

    p_minus, p_plus = endpoint_ratios(channel, max_level=max_level)
    l_c = tight_lo if p_minus > eta else base_lo
    u_c = tight_hi if p_plus > eta else base_hi
    return float(l_c), float(u_c)


def saturation_map_1c(channel: np.ndarray,
                      alpha: float | None = None,
                      max_level: float = 255.0,
                      eta: float = ETA,
                      cutoffs: tuple[float, float] | None = None,
                      ) -> np.ndarray:
    """Soft two-sided saturation map for one channel (Eqs. 4-6).

    `cutoffs=(l_c, u_c)` evaluates with FIXED cutoffs (used by the synthetic
    checks of Section IV, e.g. default (5, 250) or tightened (25, 230));
    when omitted, the one-sided adaptive rule of Eq. 3 selects them from the
    channel histogram with threshold `eta`.
    """
    if alpha is None:
        # alpha = 0.5 per 8-bit code value -> scale with bit depth
        alpha = ALPHA_8BIT / (max_level / 255.0)
    if cutoffs is None:
        l_c, u_c = adaptive_cutoffs(channel, max_level=max_level, eta=eta)
    else:
        l_c, u_c = cutoffs
    I = np.asarray(channel, dtype=np.float64)
    a_minus = _sigmoid(alpha * (l_c - I))
    a_plus = _sigmoid(alpha * (I - u_c))
    return 1.0 - (1.0 - a_minus) * (1.0 - a_plus)


def saturation_map(image: np.ndarray,
                   max_level: float = 255.0,
                   eta: float = ETA,
                   ) -> np.ndarray:
    """Per-channel saturation maps for an HxWxC (or HxW) image.

    Returns an array of the same spatial shape with a map per channel
    stacked on the last axis (HxWxC), or a single HxW map for grayscale.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 2:
        return saturation_map_1c(img, max_level=max_level, eta=eta)
    return np.stack(
        [saturation_map_1c(img[..., c], max_level=max_level, eta=eta)
         for c in range(img.shape[-1])],
        axis=-1,
    )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    # numerically stable logistic sigmoid
    out = np.empty_like(x, dtype=np.float64)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out
