"""Cross-sectional scoring engine for the sector/asset outlook.

Momentum-style factors are *relative*, so each factor is z-scored across the
universe at a point in time, then blended with configurable weights into a
composite. The composite drives a BULLISH / NEUTRAL / BEARISH call and a 0-100
conviction, and the top contributing factors are surfaced as the "why".

Weights and thresholds live in DEFAULT_OUTLOOK_CONFIG (overridable from
config.json's `outlook` block), so the model is tunable and a regime overlay
can swap weight profiles without touching this logic.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional


DEFAULT_OUTLOOK_CONFIG: Dict[str, Any] = {
    "weights": {
        "mom_12_1": 0.35,
        "trend_score": 0.30,
        "relative_strength": 0.20,
        "reversal_1m": 0.15,
    },
    # Absolute (non-z-scored) factors would shift the whole universe with the
    # trend. The backtest showed they inflate the headline hit rate but HURT the
    # genuine relative skill (IC / market-relative hit), so they are off by
    # default. Left here as a tunable knob. (mkt_trend is still computed and fed
    # for the regime gate below.)
    "absolute_weights": {},
    "absolute_gain": 7.0,
    # Regime gate: don't fight the tape. A BEARISH (down) call only fires when
    # the broad market is itself in a downtrend (mkt_trend < gate); in an uptrend
    # a weak instrument is at most NEUTRAL ("underweight"), never an absolute
    # short. This is what makes the rare "down" calls trustworthy.
    "regime_gate": True,
    "regime_gate_level": 0.0,
    "bull_threshold": 0.4,   # composite above this → BULLISH
    "bear_threshold": -0.4,  # below this → BEARISH
    "conviction_gain": 0.9,  # tanh gain mapping composite → 0-100
    # human labels for drivers
    "factor_labels": {
        "mom_12_1": "12m momentum",
        "trend_score": "trend",
        "relative_strength": "rel-strength vs mkt",
        "reversal_1m": "1m reversal",
        "abs_trend": "uptrend" ,
        "mkt_trend": "market trend",
    },
}


def load_outlook_config(config_path: str = "config.json") -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULT_OUTLOOK_CONFIG))
    try:
        with open(config_path) as fh:
            user = (json.load(fh) or {}).get("outlook") or {}
    except Exception:
        user = {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def zscore_map(values: Dict[str, Optional[float]]) -> Dict[str, float]:
    """Z-score the non-None values across the universe. Constant → all zeros."""
    present = {k: float(v) for k, v in values.items() if v is not None}
    if not present:
        return {}
    n = len(present)
    mean = sum(present.values()) / n
    var = sum((v - mean) ** 2 for v in present.values()) / n
    std = math.sqrt(var)
    if std == 0:
        return {k: 0.0 for k in present}
    return {k: (v - mean) / std for k, v in present.items()}


def classify(composite: float, cfg: Dict[str, Any]) -> str:
    if composite >= cfg.get("bull_threshold", 0.4):
        return "BULLISH"
    if composite <= cfg.get("bear_threshold", -0.4):
        return "BEARISH"
    return "NEUTRAL"


def rank_universe(
    features_by_ticker: Dict[str, Dict[str, Optional[float]]], cfg: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """Rank the universe most-bullish → most-bearish with drivers and conviction."""
    weights = cfg.get("weights", {})
    labels = cfg.get("factor_labels", {})

    # cross-sectional z-score per factor
    z_by_factor: Dict[str, Dict[str, float]] = {}
    for factor in weights:
        z_by_factor[factor] = zscore_map(
            {tk: feats.get(factor) for tk, feats in features_by_ticker.items()}
        )

    rows: List[Dict[str, Any]] = []
    for tk in features_by_ticker:
        contribs = []
        composite = 0.0
        for factor, w in weights.items():
            z = z_by_factor.get(factor, {}).get(tk)
            if z is None:
                continue
            c = w * z
            composite += c
            contribs.append((factor, c, z))
        # absolute (non-z-scored) factors: shift the whole universe with the
        # real trend so bull markets read bullish, bear markets bearish.
        for factor, w in cfg.get("absolute_weights", {}).items():
            raw = features_by_ticker[tk].get(factor)
            if raw is None:
                continue
            c = w * math.tanh(cfg.get("absolute_gain", 7.0) * raw)
            composite += c
            contribs.append((factor, c, raw))

        # drivers: top-2 by absolute contribution, with direction
        contribs.sort(key=lambda x: abs(x[1]), reverse=True)
        drivers = []
        for factor, c, z in contribs[:2]:
            if abs(c) < 1e-9:
                continue
            sign = "+" if c > 0 else "−"
            drivers.append(f"{labels.get(factor, factor)} {sign}")
        direction = classify(composite, cfg)
        # regime gate: suppress absolute bearish calls when the market is up
        if cfg.get("regime_gate") and direction == "BEARISH":
            mkt = features_by_ticker[tk].get("mkt_trend")
            if mkt is not None and mkt > cfg.get("regime_gate_level", 0.0):
                direction = "NEUTRAL"
        conviction = round(50 + 50 * math.tanh(cfg.get("conviction_gain", 0.9) * composite))
        rows.append({
            "ticker": tk,
            "score": round(composite, 4),
            "direction": direction,
            "conviction": conviction,
            "drivers": ", ".join(drivers),
        })

    rows.sort(key=lambda r: r["score"], reverse=True)
    return rows


__all__ = [
    "DEFAULT_OUTLOOK_CONFIG", "load_outlook_config", "zscore_map",
    "classify", "rank_universe",
]
