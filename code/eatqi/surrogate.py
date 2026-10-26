# -*- coding: utf-8 -*-
"""Smooth histogram pile-up surrogate (Manuscript Section III-H, Eqs. 13-15).

Corrected formulation.  The earlier endpoint-AVERAGE surrogate converged to
half the max-based ratio when only one endpoint is populated.  The corrected
surrogate uses a log-sum-exp (smooth maximum) in the numerator:

    H_c(k) = (1/N) sum_i exp( -(I_c(i) - k)^2 / (2 delta^2) )       (Eq. 13)
    S_t(v) = t * log sum_j exp(v_j / t)                             (Eq. 14)
    P_c    = S_t(H_c(0), H_c(L))
             / (S_t(H_c(0),...,H_c(L)) + eps)                       (Eq. 15)

As delta -> 0 (integer codes) and t -> 0, P_c -> max(h(0), h(L)) / max_k h(k),
matching the discrete pile-up score of Section III-B.
"""

from __future__ import annotations

import numpy as np


def smooth_histogram(values: np.ndarray,
                     max_level: float = 255.0,
                     delta: float = 1.0,
                     n_levels: int | None = None) -> np.ndarray:
    """Gaussian-kernel histogram H_c(k) at every integer code level (Eq. 13)."""
    values = np.asarray(values, dtype=np.float64).ravel()
    if n_levels is None:
        n_levels = int(round(max_level)) + 1
    k = np.arange(n_levels, dtype=np.float64)
    # H[k] = mean_i exp(-(I_i - k)^2 / (2 delta^2));  exp may underflow for far
    # pairs, which is the intended limiting behavior.
    sq = (values[:, None] - k[None, :]) ** 2
    return np.exp(-sq / (2.0 * delta * delta)).mean(axis=0)


def logsumexp(v: np.ndarray, t: float) -> float:
    """Unnormalized log-sum-exp with temperature t > 0 (Eq. 14)."""
    v = np.asarray(v, dtype=np.float64)
    m = v.max()
    return float(m + t * np.log(np.exp((v - m) / t).sum()))


def smooth_pileup_ratio(values: np.ndarray,
                        max_level: float = 255.0,
                        delta: float = 1.0,
                        t: float = 1e-5,
                        eps: float = 1e-12) -> float:
    """Smooth pile-up ratio P_c (Eq. 15): smooth-max endpoints / smooth-max all."""
    H = smooth_histogram(values, max_level=max_level, delta=delta)
    num = logsumexp(np.array([H[0], H[-1]]), t)
    den = logsumexp(H, t)
    return num / (den + eps)


def hard_pileup_ratio(values: np.ndarray, max_level: float = 255.0) -> float:
    """Discrete reference score P_c = max(h(0), h(L)) / max_k h(k)."""
    values = np.asarray(values)
    levels = int(round(max_level)) + 1
    hist, _ = np.histogram(values, bins=levels, range=(0.0, max_level))
    return float(hist[[0, -1]].max() / (hist.max() + 1e-300))


def endpoint_average_ratio(values: np.ndarray, max_level: float = 255.0) -> float:
    """The superseded (incorrect) endpoint-average surrogate, kept only to
    demonstrate the factor-of-two error in diagnostics (Table I)."""
    values = np.asarray(values)
    levels = int(round(max_level)) + 1
    hist, _ = np.histogram(values, bins=levels, range=(0.0, max_level))
    modal = hist.max() + 1e-300
    return float((hist[0] + hist[-1]) / (2.0 * modal))


def soft_adaptive_cutoffs(values: np.ndarray,
                          max_level: float = 255.0,
                          delta: float = 1.0,
                          t: float = 1e-5,
                          eta: float = 0.015,
                          gamma: float = 200.0) -> tuple[float, float]:
    """One-sided soft adaptation: replace each indicator 1[p > eta] by
    sigmoid(gamma * (p - eta)) and interpolate the cutoffs (Section III-H).

    Returns (l_c, u_c) as continuous functions of the endpoint ratios:
        l_c = l_default + (l_tight - l_default) * s_minus
        u_c = u_default + (u_tight - u_default) * s_plus
    with s in (0, 1); gamma is selected on validation data.
    """
    scale = max_level / 255.0
    H = smooth_histogram(values, max_level=max_level, delta=delta)
    den = logsumexp(H, t) + 1e-12
    p_minus = logsumexp(np.array([H[0]]), t) / den
    p_plus = logsumexp(np.array([H[-1]]), t) / den
    s_minus = _sigmoid(gamma * (p_minus - eta))
    s_plus = _sigmoid(gamma * (p_plus - eta))
    l_c = 5.0 * scale + (25.0 - 5.0) * scale * s_minus
    u_c = 250.0 * scale - (250.0 - 230.0) * scale * s_plus
    return float(l_c), float(u_c)


def _sigmoid(x):
    x = np.atleast_1d(np.asarray(x, dtype=np.float64))
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return float(out[0]) if out.size == 1 else out
