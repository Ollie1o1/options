"""Pure scoring metrics for the breakout backtest: calibration (Brier, ECE,
reliability), discrimination (AUC), distribution quality (pinball, coverage),
and a skill score versus the baseline."""
from __future__ import annotations
from typing import List, Optional
import numpy as np


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float:
    return float(np.mean((np.asarray(probs) - np.asarray(outcomes)) ** 2))


def reliability_bins(probs, outcomes, n_bins: int = 10) -> List[dict]:
    probs, outcomes = np.asarray(probs), np.asarray(outcomes)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        m = (probs >= lo) & (probs < hi if i < n_bins - 1 else probs <= hi)
        if m.any():
            out.append({"bin_lo": float(lo), "bin_hi": float(hi),
                        "mean_pred": float(np.mean(probs[m])),
                        "mean_obs": float(np.mean(outcomes[m])), "n": int(m.sum())})
    return out


def expected_calibration_error(probs, outcomes, n_bins: int = 10) -> float:
    probs = np.asarray(probs)
    bins = reliability_bins(probs, outcomes, n_bins)
    total = len(probs)
    return float(sum(b["n"] / total * abs(b["mean_pred"] - b["mean_obs"]) for b in bins)) if total else 0.0


def auc(scores, labels) -> Optional[float]:
    scores, labels = np.asarray(scores, dtype=float), np.asarray(labels)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ties
    _, inv, counts = np.unique(scores, return_inverse=True, return_counts=True)
    tie_mean = np.array([ranks[scores == s].mean() for s in scores])
    sum_pos = tie_mean[labels == 1].sum()
    n_pos, n_neg = len(pos), len(neg)
    return float((sum_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def pinball_loss(pred_q: np.ndarray, realized: np.ndarray, q: float) -> float:
    pred_q, realized = np.asarray(pred_q), np.asarray(realized)
    diff = realized - pred_q
    return float(np.mean(np.maximum(q * diff, (q - 1) * diff)))


def band_coverage(los, his, realized) -> float:
    los, his, realized = np.asarray(los), np.asarray(his), np.asarray(realized)
    return float(np.mean((realized >= los) & (realized <= his)))


def skill_score(brier_model: float, brier_baseline: float) -> float:
    return 1.0 - brier_model / brier_baseline if brier_baseline > 0 else 0.0
