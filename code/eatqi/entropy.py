# -*- coding: utf-8 -*-
"""Windowed local Shannon entropy (Manuscript Section III-D, Eqs. 9-10).

Corrected tensor description: a single linear depth-wise convolution on the
luminance channel CANNOT compute Shannon entropy.  The operator is:

  1. normalize luminance to Y in [0, 1];
  2. quantize into B = 16 bins -> per-pixel one-hot (bin-indicator) channels
     Q_q(i, j) = 1[Q(Y(i,j)) = q]                         (Eq. 9 inner term)
  3. average each indicator channel over a reflected-padding 9x9 window:
         p_q(i,j) = (1/81) * sum_{(v,w) in Omega_ij} 1[Q(Y(v,w)) = q]   (Eq. 9)
  4. reduce nonlinearly:
         M_ent(i,j) = -sum_q p_q log(p_q + eps) / log B                   (Eq. 10)

Step 3 is an average pooling (or unfold + mean) over the B indicator
channels — a linear operation on the EXPANDED bins, not on luminance.
"""

from __future__ import annotations

import numpy as np

from . import ENTROPY_BINS, ENTROPY_WINDOW


def bin_indicators(luma01: np.ndarray, n_bins: int = ENTROPY_BINS) -> np.ndarray:
    """Quantize normalized luminance into n_bins indicator channels.

    Returns HxWxB float array of 0/1 one-hot values.
    """
    luma01 = np.clip(np.asarray(luma01, dtype=np.float64), 0.0, 1.0)
    q = np.minimum((luma01 * n_bins).astype(np.int64), n_bins - 1)
    onehot = np.zeros(luma01.shape + (n_bins,), dtype=np.float64)
    np.put_along_axis(onehot, q[..., None], 1.0, axis=-1)
    return onehot


def _window_mean_reflect(channels: np.ndarray, window: int) -> np.ndarray:
    """Average each channel over a window x window window with reflection pad."""
    pad = window // 2
    padded = np.pad(channels,
                    ((pad, pad), (pad, pad), (0, 0)),
                    mode="reflect")
    H, W, B = channels.shape
    # sliding-window view: HxWxBx(window*window)
    from numpy.lib.stride_tricks import sliding_window_view
    sw = sliding_window_view(padded, (window, window), axis=(0, 1))
    return sw.mean(axis=(-1, -2))


def entropy_map(luma: np.ndarray,
                max_level: float = 255.0,
                n_bins: int = ENTROPY_BINS,
                window: int = ENTROPY_WINDOW,
                eps: float = 1e-12) -> np.ndarray:
    """Normalized local Shannon entropy map in [0, 1] (Eqs. 9-10)."""
    luma = np.asarray(luma, dtype=np.float64)
    y01 = np.clip(luma / max_level, 0.0, 1.0)
    indicators = bin_indicators(y01, n_bins=n_bins)
    probs = _window_mean_reflect(indicators, window)          # Eq. 9
    ent = -np.sum(probs * np.log(probs + eps), axis=-1) / np.log(n_bins)  # Eq. 10
    return ent
