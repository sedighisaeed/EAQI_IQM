# -*- coding: utf-8 -*-
"""Evaluation metrics and uncertainty (Manuscript Section VI-C).

  * SRCC / KRCC on polarity-corrected raw predictions;
  * PLCC / RMSE only after a monotone four-parameter logistic (4PL)
    calibration f(x) = b2 + (b1 - b2) / (1 + exp(-(x - b3) / b4))   (Eq. 16)
    fitted on a DISJOINT calibration partition;
  * paired, source-grouped bootstrap comparisons (2,000 resamples initial
    setting) applying the same resample to both methods;
  * seed-level aggregation: per-seed metrics, mean and sample SD
    (>= 3 documented seeds required by Section VI-C).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import optimize, stats


# --------------------------------------------------------------------------
# rank correlations on polarity-corrected raw predictions
# --------------------------------------------------------------------------

def srcc(pred: np.ndarray, target: np.ndarray) -> float:
    return float(stats.spearmanr(pred, target).statistic)


def krcc(pred: np.ndarray, target: np.ndarray) -> float:
    return float(stats.kendalltau(pred, target).statistic)


# --------------------------------------------------------------------------
# four-parameter logistic calibration (Eq. 16)
# --------------------------------------------------------------------------

def logistic4(x: np.ndarray, b1: float, b2: float, b3: float, b4: float):
    """Monotone 4PL: f(x) = b2 + (b1 - b2) / (1 + exp(-(x - b3)/b4)), b4 > 0."""
    return b2 + (b1 - b2) / (1.0 + np.exp(-(x - b3) / b4))


@dataclass
class Calibration4PL:
    params: tuple[float, float, float, float]

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return logistic4(np.asarray(x, dtype=np.float64), *self.params)


def fit_4pl(x: np.ndarray, y: np.ndarray) -> Calibration4PL:
    """Fit the monotone 4PL on a calibration partition (Section VI-C).

    Enforces b4 > 0 (monotone increasing in x); orient the inputs so that
    larger x means better quality, or orient prediction and target together
    toward severity, before calling this function.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    def residual(p):
        return logistic4(x, *p) - y

    y_range = max(y.max() - y.min(), 1e-9)
    p0 = (y.max(), y.min(), float(x.mean()), max(y_range / 6.0, 1e-3))
    sol, _ = optimize.leastsq(residual, p0)
    b1, b2, b3, b4 = (float(v) for v in sol)
    if b4 < 0:  # keep orientation monotone-increasing in x
        b4 = abs(b4)
    return Calibration4PL((b1, b2, b3, b4))


def plcc_rmse(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """PLCC and RMSE after logistic mapping (target in original MOS units)."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    r = float(np.corrcoef(pred, target)[0, 1])
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    return r, rmse


def evaluate_with_calibration(pred_calib: np.ndarray, target_calib: np.ndarray,
                              pred_test: np.ndarray,
                              target_test: np.ndarray) -> dict:
    """Full Section VI-C protocol: fit 4PL on the calibration partition,
    report SRCC/KRCC on raw polarity-corrected scores and PLCC/RMSE on
    calibrated test predictions (MOS units)."""
    cal = fit_4pl(pred_calib, target_calib)
    mapped = cal(pred_test)
    plcc, rmse = plcc_rmse(mapped, target_test)
    return {
        "srcc": srcc(pred_test, target_test),
        "krcc": krcc(pred_test, target_test),
        "plcc": plcc,
        "rmse": rmse,
        "logistic_params": cal.params,
    }


# --------------------------------------------------------------------------
# paired, source-grouped bootstrap comparison
# --------------------------------------------------------------------------

def paired_bootstrap(pred_a: np.ndarray, pred_b: np.ndarray, target: np.ndarray,
                     groups: np.ndarray | None = None,
                     n_resamples: int = 2000,
                     statistic=srcc,
                     rng_seed: int = 0) -> dict:
    """Paired bootstrap for comparing two models on the SAME images.

    Resampling units are source groups when `groups` is given (source-grouped
    bootstrap, Section VI-C); the identical resample is applied to both
    methods.  Returns the observed statistic difference and its CI.
    """
    pred_a = np.asarray(pred_a)
    pred_b = np.asarray(pred_b)
    target = np.asarray(target)
    n = len(target)
    rng = np.random.default_rng(rng_seed)

    if groups is None:
        unit_idx = [np.arange(n)]
        m = 1
    else:
        groups = np.asarray(groups)
        uniq = np.unique(groups)
        group_of = np.searchsorted(uniq, groups)
        unit_idx = [np.flatnonzero(group_of == g) for g in range(len(uniq))]
        m = len(uniq)

    delta_obs = statistic(pred_a, target) - statistic(pred_b, target)
    deltas = np.empty(n_resamples, dtype=np.float64)
    for r in range(n_resamples):
        picks = rng.integers(0, m, size=m)
        idx = np.concatenate([unit_idx[p] for p in picks])
        deltas[r] = (statistic(pred_a[idx], target[idx])
                     - statistic(pred_b[idx], target[idx]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_two = 2.0 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {
        "delta_obs": float(delta_obs),
        "ci95": (float(lo), float(hi)),
        "p_two_sided_bootstrap": float(min(1.0, p_two)),
        "resamples": n_resamples,
        "resample_unit": "source_group" if groups is not None else "image",
    }


# --------------------------------------------------------------------------
# seed-level aggregation (>= 3 seeds, per-seed metrics, mean and sample SD)
# --------------------------------------------------------------------------

def seed_report_helper():
    """Deliberately raises: used by tests to confirm the >= 3 seeds rule."""
    SeedReport().aggregate()


@dataclass
class SeedReport:
    per_seed: list = field(default_factory=list)

    def add(self, seed: int, metrics: dict) -> None:
        row = {"seed": int(seed)}
        row.update(metrics)
        self.per_seed.append(row)

    def aggregate(self) -> dict:
        if len(self.per_seed) < 3:
            raise ValueError("Section VI-C requires at least 3 documented seeds")
        keys = [k for k in self.per_seed[0] if k != "seed"]
        out = {"n_seeds": len(self.per_seed), "per_seed": self.per_seed}
        for k in keys:
            vals = np.array([row[k] for row in self.per_seed], dtype=np.float64)
            out[k] = {"mean": float(vals.mean()), "sd": float(vals.std(ddof=1))}
        return out
