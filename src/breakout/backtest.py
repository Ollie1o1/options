"""Walk-forward, point-in-time backtest for the breakout engine. At each grid
date it builds the forward-return distribution from only-past data, then scores
the +10% up-breakout probability (and the 80% band) against the realized
horizon-forward return. Calibration is fit on the first half of samples and
scored on the second — never on data the prediction saw."""
from __future__ import annotations
from typing import Dict
import numpy as np

from src.breakout.data import Series, HORIZONS
from src.breakout import features as F
from src.breakout.distribution import baseline_distribution, parametric_distribution, make_distribution
from src.breakout import metrics as M
from src.breakout.calibrate import fit_isotonic, apply_isotonic

UP_THRESHOLD = 0.10


def collect_samples(series_by_ticker: Dict[str, Series], horizon: int, model: str,
                    step: int = 21, start: int = 252, seed: int = 0) -> dict:
    up_probs, up_outcomes, los, his, realized = [], [], [], [], []
    for s in series_by_ticker.values():
        n = len(s.close)
        for t in range(start, n - horizon, step):
            d = make_distribution(s, t, horizon, model, seed)
            fwd = float(s.close[t + horizon] / s.close[t] - 1.0)
            up_probs.append(d.prob_ge(UP_THRESHOLD))
            up_outcomes.append(1.0 if fwd >= UP_THRESHOLD else 0.0)
            lo, hi = d.band(0.1, 0.9)
            los.append(lo); his.append(hi); realized.append(fwd)
    return {"up_probs": np.array(up_probs), "up_outcomes": np.array(up_outcomes),
            "los": np.array(los), "his": np.array(his), "realized": np.array(realized)}


def _score(samples: dict, baseline_probs: np.ndarray) -> dict:
    p, y = samples["up_probs"], samples["up_outcomes"]
    half = len(p) // 2
    # Always evaluate on a held-out second half; calibrate only when the
    # first half is large enough to fit a meaningful isotonic map.
    if half >= 5:
        cal = fit_isotonic(p[:half], y[:half])
        ps = np.array([apply_isotonic(cal, x) for x in p[half:]])
    else:
        ps = p[half:]
    ys = y[half:]
    bl = baseline_probs[half:]
    los = samples["los"][half:]
    his = samples["his"][half:]
    realized = samples["realized"][half:]
    return {
        "n": int(len(ys)),
        "brier": M.brier_score(ps, ys),
        "ece": M.expected_calibration_error(ps, ys),
        "auc": M.auc(ps, ys),
        "coverage": M.band_coverage(los, his, realized),
        "skill_vs_baseline": M.skill_score(M.brier_score(ps, ys), M.brier_score(bl, ys)),
    }


def run_backtest(series_by_ticker: Dict[str, Series], model: str = "parametric",
                 step: int = 21, seed: int = 0) -> dict:
    out = {}
    for label, h in HORIZONS.items():
        s = collect_samples(series_by_ticker, h, model, step=step, seed=seed)
        b = collect_samples(series_by_ticker, h, "baseline", step=step, seed=seed)
        out[label] = _score(s, b["up_probs"]) if len(s["up_probs"]) else {"n": 0}
    return out
