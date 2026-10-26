# -*- coding: utf-8 -*-
"""Trainable objective and score polarity (Manuscript Section III-G).

Objective (Eq. 12):  L_total = L_MSE + lambda_r * L_rank + lambda_a * L_aux

  * L_MSE: MSE between severity s and normalized annoyance target y = a/100.
  * L_rank: pairwise hinge  max(0, 0.1 - (s_i - s_j))  on eligible pairs with
    y_i > y_j + 0.05 (both thresholds on the [0, 1] target scale).
  * L_aux: cross-entropy for distortion family (5 classes) and severity class
    (3 classes), MSE for the continuous spatial mask, BCE-with-logits for the
    binary edge-interaction target.  Each term is averaged over VALID labeled
    samples only; an absent target contributes zero (mask-aware losses).

Polarity conventions (Section III-G):
  * annoyance ratings a in [0, 100] -> y = a / 100; report 100 * s.
  * quality benchmarks with MOS m in [m_min, m_max] (higher is better):
    y = 1 - (m - m_min) / (m_max - m_min); compute rank correlations against
    quality using -s (equivalently 1 - s).
  * higher-is-worse DMOS keeps severity polarity after documented
    normalization (e.g., LIVE DMOS); PIPAL uses Elo-derived quality scores.
  * A fixed-image LPIPS term has NO gradient w.r.t. assessor parameters and
    is deliberately EXCLUDED from this objective.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

DEFAULT_LAMBDA_R = 0.5
DEFAULT_LAMBDA_A = 0.3
RANK_MARGIN = 0.1
RANK_PAIR_THRESHOLD = 0.05


def mse_loss(pred: torch.Tensor, target: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
    """Mean-squared error averaged over valid (mask != 0) elements."""
    err = (pred - target) ** 2
    if mask is None:
        return err.mean()
    mask = mask.float()
    denom = mask.sum().clamp(min=1.0)
    return (err * mask).sum() / denom


def rank_loss(pred: torch.Tensor, target: torch.Tensor,
              margin: float = RANK_MARGIN,
              threshold: float = RANK_PAIR_THRESHOLD) -> torch.Tensor:
    """Pairwise hinge ranking loss on eligible pairs (Section III-G).

    Eligible pair: target_i > target_j + threshold.  Zero if no eligible
    pair exists in the batch.
    """
    pred = pred.flatten()
    target = target.flatten()
    hi, lo = torch.triu_indices(len(target), len(target), offset=1)
    if hi.numel() == 0:
        return pred.sum() * 0.0
    t_hi, t_lo = target[hi], target[lo]
    eligible = (t_hi - t_lo) > threshold
    if not torch.any(eligible):
        return pred.sum() * 0.0
    s_hi, s_lo = pred[hi[eligible]], pred[lo[eligible]]
    return F.relu(margin - (s_hi - s_lo)).mean()


def aux_losses(outputs: dict, targets: dict) -> tuple[torch.Tensor, dict]:
    """Mask-aware auxiliary supervision (Section III-F/G).

    outputs / targets keys:
      family_logits  [N,5] / family        [N]  (long, -1 = absent)
      severity_logits[N,3] / severity      [N]  (long, -1 = absent)
      spatial_map    [N,1,h,w] / spatial   [N,1,h,w] (float, nan = absent)
      edge_logits    [N] / edge_interaction [N] (long, -1 = absent)
    """
    parts: dict[str, torch.Tensor] = {}

    if "family_logits" in outputs:
        lab = targets["family"]
        valid = lab >= 0
        parts["family_ce"] = (F.cross_entropy(outputs["family_logits"][valid],
                                              lab[valid])
                              if valid.any() else outputs["family_logits"].sum() * 0.0)

    if "severity_logits" in outputs:
        lab = targets["severity"]
        valid = lab >= 0
        parts["severity_ce"] = (F.cross_entropy(outputs["severity_logits"][valid],
                                                lab[valid])
                                if valid.any() else outputs["severity_logits"].sum() * 0.0)

    if "spatial_map" in outputs:
        tgt = targets["spatial"]
        valid = torch.isfinite(tgt).all(dim=tuple(range(1, tgt.ndim)))   # [N]
        if valid.any():
            parts["spatial_mse"] = mse_loss(outputs["spatial_map"][valid],
                                            tgt[valid])
        else:
            parts["spatial_mse"] = outputs["spatial_map"].sum() * 0.0

    if "edge_logits" in outputs:
        lab = targets["edge_interaction"]
        valid = lab >= 0
        parts["edge_bce"] = (F.binary_cross_entropy_with_logits(
            outputs["edge_logits"][valid], lab[valid].float())
            if valid.any() else outputs["edge_logits"].sum() * 0.0)

    total = sum(parts.values()) if parts else torch.tensor(0.0)
    return total, parts


def total_loss(pred: torch.Tensor, target: torch.Tensor,
               outputs: dict, targets: dict,
               lambda_r: float = DEFAULT_LAMBDA_R,
               lambda_a: float = DEFAULT_LAMBDA_A) -> tuple[torch.Tensor, dict]:
    """L_total = L_MSE + lambda_r L_rank + lambda_a L_aux   (Eq. 12)."""
    l_mse = mse_loss(pred.flatten(), target.flatten())
    l_rank = rank_loss(pred.flatten(), target.flatten())
    l_aux, aux_parts = aux_losses(outputs, targets)
    loss = l_mse + lambda_r * l_rank + lambda_a * l_aux
    log = {"loss": float(loss.detach()),
           "mse": float(l_mse.detach()),
           "rank": float(l_rank.detach()),
           "aux": float(l_aux.detach()),
           **{k: float(v.detach()) for k, v in aux_parts.items()}}
    return loss, log


# --------------------------------------------------------------------------
# score polarity utilities (Section III-G)
# --------------------------------------------------------------------------

def annoyance_target(a_0_100: torch.Tensor) -> torch.Tensor:
    """a in [0, 100] -> y in [0, 1] severity target."""
    return a_0_100 / 100.0


def quality_to_severity(mos: torch.Tensor) -> torch.Tensor:
    """Higher-is-better MOS (any known scale) -> severity target y in [0, 1]."""
    m_min, m_max = mos.min(), mos.max()
    return 1.0 - (mos - m_min) / (m_max - m_min).clamp(min=1e-12)


def severity_to_quality(pred_severity: torch.Tensor) -> torch.Tensor:
    """Map severity predictions to quality orientation: use (-s) or 1 - s
    when computing rank correlations against a quality MOS (Section III-G)."""
    return 1.0 - pred_severity


def severity_scores_0_100(pred_severity: torch.Tensor) -> torch.Tensor:
    """Report annoyance as 100 * s (Section III-G)."""
    return 100.0 * pred_severity
