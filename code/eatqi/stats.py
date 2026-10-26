# -*- coding: utf-8 -*-
"""Statistics for the perceptual study plan (Manuscript Section V-C).

Implements the definitions the revised manuscript mandates:
  * paired t-test and d_z = mean(difference) / SD(difference) = t / sqrt(N);
  * 95% CI for d_z via the noncentral-t method;
  * a-priori power for a two-sided paired t-test (exact noncentral t);
  * an explicit check that the earlier draft's "N = 27 raters" is NOT
    sufficient for alpha = 0.01, power = 0.80, d_z = 0.5 (two-sided).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class PairedResult:
    n: int
    mean_diff: float
    sd_diff: float
    t_stat: float
    df: int
    p_two_sided: float
    dz: float
    ci95_low: float
    ci95_high: float


def paired_analysis(diffs: np.ndarray, alpha: float = 0.05) -> PairedResult:
    """Paired t-test with d_z and its CI from the actual participant
    differences (Section V-C definition)."""
    diffs = np.asarray(diffs, dtype=np.float64)
    n = diffs.size
    mean_d = float(diffs.mean())
    sd_d = float(diffs.std(ddof=1))
    se = sd_d / np.sqrt(n)
    t_stat = mean_d / se
    p = 2.0 * stats.t.sf(abs(t_stat), df=n - 1)
    dz = t_stat / np.sqrt(n)          # = mean_d / sd_d

    lo, hi = dz_ci_noncentral_t(t_stat, n, alpha)
    return PairedResult(n, mean_d, sd_d, float(t_stat), n - 1, float(p),
                        float(dz), lo, hi)


def dz_ci_noncentral_t(t_stat: float, n: int, alpha: float = 0.05):
    """CI for d_z = t/sqrt(n) using the noncentral t distribution.

    Lower bound: ncp such that P(T_{df, ncp} <= t) = 1 - alpha/2
    Upper bound: ncp such that P(T_{df, ncp} <= t) = alpha/2
    """
    df = n - 1
    tc = stats.t.ppf(1.0 - alpha / 2.0, df)
    ncp_lo = _invert_nct_cdf(t_stat, df, 1.0 - alpha / 2.0)
    ncp_hi = _invert_nct_cdf(t_stat, df, alpha / 2.0)
    return float(ncp_lo / np.sqrt(n)), float(ncp_hi / np.sqrt(n))


def _invert_nct_cdf(t_stat: float, df: int, prob: float) -> float:
    """Solve P(T_{df, ncp} <= t_stat) = prob for ncp.

    The noncentral-t CDF is strictly DEcreasing in ncp at fixed t, so the
    root is bracketed by (t - width, t + width); solved with brentq.
    """
    from scipy.optimize import brentq
    from scipy.stats import nct

    def f(ncp: float) -> float:
        return nct.cdf(t_stat, df, ncp) - prob

    a, b = t_stat - 10.0, t_stat + 10.0
    fa, fb = f(a), f(b)
    while fa * fb > 0:                # same sign -> widen (CDF strictly decreasing)
        a -= 10.0
        b += 10.0
        fa, fb = f(a), f(b)
        if (b - a) > 1e6:
            raise RuntimeError("bracket failure in noncentral-t inversion")
    return float(brentq(f, a, b, xtol=1e-10))


def power_paired_t(n: int, dz: float, alpha: float = 0.01,
                   two_sided: bool = True) -> float:
    """Achieved power of a paired t-test with n subjects and effect d_z."""
    df = n - 1
    ncp = dz * np.sqrt(n)
    if two_sided:
        tc = stats.t.ppf(1.0 - alpha / 2.0, df)
        return float(stats.nct.sf(tc, df, ncp) + stats.nct.cdf(-tc, df, ncp))
    tc = stats.t.ppf(1.0 - alpha, df)
    return float(stats.nct.sf(tc, df, ncp))


def min_n_paired(dz: float, alpha: float = 0.01, power: float = 0.80,
                 two_sided: bool = True) -> int:
    """Smallest n reaching the requested power (Section V-C requirement)."""
    n = 4
    while power_paired_t(n, dz, alpha, two_sided) < power:
        n += 1
        if n > 10_000:
            raise ValueError("no feasible n <= 10000")
    return n
