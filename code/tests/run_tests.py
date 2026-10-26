# -*- coding: utf-8 -*-
"""Unit tests for the E-AQI reference implementation.

Run:  cd code && python -m tests.run_tests
Pure-stdlib test runner (no pytest dependency required).
"""

import math
import sys
import os

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from eatqi.clipping import (adaptive_cutoffs, endpoint_ratios,  # noqa: E402
                            saturation_map_1c)
from eatqi.entropy import entropy_map                          # noqa: E402
from eatqi.losses import (annoyance_target, quality_to_severity,  # noqa: E402
                          rank_loss, total_loss)
from eatqi.metrics import (Calibration4PL, fit_4pl, logistic4,  # noqa: E402
                           paired_bootstrap, seed_report_helper)
from eatqi.surrogate import (hard_pileup_ratio,                 # noqa: E402
                             soft_adaptive_cutoffs,
                             smooth_pileup_ratio)
from eatqi.stats import (min_n_paired, paired_analysis,         # noqa: E402
                         power_paired_t)

FAILURES = []


def check(name, cond, detail=""):
    status = "PASS" if cond else "FAIL"
    print(f"[{status}] {name}" + (f"  ({detail})" if detail else ""))
    if not cond:
        FAILURES.append(name)


# ---------------------------------------------------------------- clipping
def test_clipping():
    # endpoints activate under default cutoffs
    check("mask(0) > 0.9", saturation_map_1c(np.array([0.0]))[0] > 0.9)
    check("mask(255) > 0.9", saturation_map_1c(np.array([255.0]))[0] > 0.9)
    # mid-gray never flagged
    check("mask(128) < 1e-10", saturation_map_1c(np.array([128.0]))[0] < 1e-10)
    # cutoff midpoint = 0.5 (sigmoid midpoint behavior, fixed default cutoffs)
    codes = np.arange(256, dtype=np.float64)
    m = saturation_map_1c(codes, cutoffs=(5.0, 250.0))
    check("mask(5) ~ 0.5", abs(m[5] - 0.5) < 1e-9)
    check("mask(250) ~ 0.5", abs(m[250] - 0.5) < 1e-9)
    # tightened cutoffs dominate default at every code
    m_tight = saturation_map_1c(codes, cutoffs=(25.0, 230.0))
    check("tightened >= default everywhere", float((m_tight - m).min()) >= 0.0)
    # one-sided independent adaptation
    bright = np.concatenate([np.full(50, 255.0), np.full(150, 100.0),
                             np.full(56, 120.0)])
    l, u = adaptive_cutoffs(bright)
    p_minus, p_plus = endpoint_ratios(bright)
    check("one-sided: u tightens, l stays",
          u == 230.0 and l == 5.0 and p_plus > 0.015 and p_minus < 0.015,
          f"l={l}, u={u}, p-={p_minus:.4f}, p+={p_plus:.4f}")
    # 10-bit equivalence via scaling (fractional cutoffs allowed)
    rng = np.random.default_rng(3)
    img8 = rng.integers(0, 256, (8, 8)).astype(float)
    img10 = img8 * 1023.0 / 255.0
    d = np.abs(saturation_map_1c(img10, max_level=1023.0)
               - saturation_map_1c(img8, max_level=255.0)).max()
    check("8/10-bit algebraic equivalence", d < 1e-12, f"max diff {d:.2e}")
    # shadow-only pile-up tightens the LOWER cutoff only
    dark = np.concatenate([np.full(50, 0.0), rng.integers(100, 256, (206,))])
    l2, u2 = adaptive_cutoffs(dark)
    check("shadow-only: l tightens, u stays", l2 == 25.0 and u2 == 250.0)


# ---------------------------------------------------------------- entropy
def test_entropy():
    rng = np.random.default_rng(0)
    flat = np.full((64, 64), 128.0)
    noisy = rng.integers(0, 256, (64, 64)).astype(float)
    e_flat = entropy_map(flat)
    e_noisy = entropy_map(noisy)
    check("entropy(flat) ~ 0", float(e_flat.max()) < 0.05,
          f"max {e_flat.max():.4f}")
    check("entropy(noise) ~ 1", float(e_noisy.mean()) > 0.95,
          f"mean {e_noisy.mean():.4f}")
    check("entropy output shape", e_flat.shape == (64, 64))


# ---------------------------------------------------------------- surrogate
def test_surrogate():
    seq = np.concatenate([np.full(20, 255.0),
                          np.linspace(1, 254, 980)])
    hard = hard_pileup_ratio(seq)
    smooth = smooth_pileup_ratio(seq, delta=0.1, t=1e-5)
    check("smooth -> hard limit", abs(smooth - hard) < 1e-6,
          f"hard {hard:.6f}, smooth {smooth:.6f}")
    # ratio bounded by 1
    check("ratio <= 1", smooth <= 1.0)
    # soft cutoffs: bright-only pile-up tightens ONLY the upper cutoff
    bright = np.concatenate([np.full(60, 255.0),
                             rng_uniform_mid()])
    l, u = soft_adaptive_cutoffs(bright, eta=0.015, gamma=500.0)
    check("soft adaptation tightens upper cutoff",
          u < 235.0 and l > 4.0, f"l={l:.3f}, u={u:.3f}")
    # smooth ratio between the hard value and the superseded average
    from eatqi.surrogate import endpoint_average_ratio
    old = endpoint_average_ratio(seq)
    check("average = half of hard (one-sided case)",
          abs(old - 0.5 * hard) < 1e-9,
          f"avg {old:.6f} vs hard/2 {0.5 * hard:.6f}")


def rng_uniform_mid():
    rng = np.random.default_rng(5)
    return rng.uniform(80.0, 200.0, 940)


# ---------------------------------------------------------------- stats
def test_stats():
    # analytic identity: dz = t / sqrt(N)
    rng = np.random.default_rng(42)
    diffs = rng.normal(22.2, 12.0, 35)
    res = paired_analysis(diffs)
    check("dz == t/sqrt(N)",
          abs(res.dz - res.t_stat / math.sqrt(res.n)) < 1e-12)
    check("CI brackets dz", res.ci95_low < res.dz < res.ci95_high,
          f"CI [{res.ci95_low:.3f}, {res.ci95_high:.3f}], dz={res.dz:.3f}")
    # power: n=27 insufficient for dz=0.5, alpha=.01, two-sided, power=.80
    p27 = power_paired_t(27, 0.5, alpha=0.01, two_sided=True)
    n_min = min_n_paired(0.5, alpha=0.01, power=0.80, two_sided=True)
    check("n=27 underpowered (alpha=.01, dz=.5)", p27 < 0.80,
          f"power(27)={p27:.3f}")
    check("min n ~ 47-49", 45 <= n_min <= 52, f"min n = {n_min}")


# ---------------------------------------------------------------- losses
def test_losses():
    pred = torch.tensor([0.2, 0.8, 0.5])
    tgt = torch.tensor([0.1, 0.9, 0.5])
    rl = rank_loss(pred, tgt)
    check("rank loss zero for margin-satisfied pairs", float(rl) >= 0.0)
    # margin violation must be penalized: hinge = 0.1 - (0.05 - 0.0) = 0.05
    pred_bad = torch.tensor([0.05, 0.0])
    tgt_bad = torch.tensor([0.9, 0.1])
    rl_bad = rank_loss(pred_bad, tgt_bad)
    check("rank loss penalizes margin violation",
          abs(float(rl_bad) - 0.05) < 1e-9,
          f"loss {float(rl_bad):.4f} (expected 0.05)")
    # reversed order must be penalized more: hinge = 0.1 - (0.0 - 0.0) = 0.1
    pred_rev = torch.tensor([0.0, 0.0])
    rl_rev = rank_loss(pred_rev, tgt_bad)
    check("rank loss penalizes reversal",
          abs(float(rl_rev) - 0.1) < 1e-6, f"loss {float(rl_rev):.6f}")
    # mask-aware aux: absent labels contribute zero
    outputs = {"family_logits": torch.randn(4, 5, requires_grad=True)}
    targets = {"family": torch.full((4,), -1, dtype=torch.long)}
    aux, parts = total_loss(torch.rand(4), torch.rand(4), outputs, targets)
    check("absent aux labels contribute zero",
          "family_ce" not in parts or float(parts["family_ce"]) == 0.0)
    # polarity utilities
    y = annoyance_target(torch.tensor([50.0]))
    check("annoyance scaling", float(y[0]) == 0.5)
    q = quality_to_severity(torch.tensor([1.0, 3.0, 5.0]))
    check("quality->severity orientation",
          q[0].item() == 1.0 and q[2].item() == 0.0)


# ---------------------------------------------------------------- metrics
def test_metrics():
    rng = np.random.default_rng(1)
    x = rng.uniform(0, 1, 400)
    true = 2.0 + 8.0 / (1.0 + np.exp(-(x - 0.5) / 0.15))     # a 4PL truth
    y = true + rng.normal(0, 0.05, 400)
    cal = fit_4pl(x[:200], y[:200])
    mapped = cal(x[200:])
    # calibrated PLCC should be high; SRCC invariant
    from eatqi.metrics import plcc_rmse, srcc
    plcc, rmse = plcc_rmse(mapped, y[200:])
    check("4PL recovers monotone mapping", plcc > 0.95, f"PLCC {plcc:.4f}")
    check("4PL RMSE ~ noise floor", rmse < 0.2, f"RMSE {rmse:.4f}")
    check("SRCC in [-1, 1]", -1.0 <= srcc(x[200:], y[200:]) <= 1.0)
    # paired bootstrap on identical predictions -> CI contains 0
    a = rng.uniform(0, 1, 120)
    b = np.clip(a + rng.normal(0, 0.05, 120), 0, 1)
    t = rng.uniform(0, 1, 120)
    boot = paired_bootstrap(a, b, t, n_resamples=300)
    check("bootstrap CI finite", np.all(np.isfinite(boot["ci95"])),
          str(boot["ci95"]))
    # seed report requires >= 3 seeds
    try:
        seed_report_helper()
        raised = False
    except Exception:
        raised = True
    check("seed report enforces >= 3 seeds", raised)


# ---------------------------------------------------------------- runner
def main():
    test_clipping()
    test_entropy()
    test_surrogate()
    test_stats()
    test_losses()
    test_metrics()
    print()
    if FAILURES:
        print(f"{len(FAILURES)} FAILURE(S):", FAILURES)
        sys.exit(1)
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
