"""Evidence-based lottery-ticket selector.

Buying far-OTM ("lottery") options is, on average, a documented loser: Boyer &
Vorkink (2014) show the most lottery-like options earn deeply negative average
returns because buyers overpay for skewness. So this selector does NOT chase
the cheapest / furthest-OTM contract. It scores candidates on factors the
literature rewards and disqualifies the classic traps:

  vol_cheapness  Goyal-Saretto (2009): buy when realized vol > implied / IV rank
                 is low (the option is cheap vs how much the stock actually moves)
  momentum       the underlying must move your way (trend-aligned)
  convexity      a reachable strike near a ~2-sigma sweet spot, NOT the max-OTM
                 moonshot (Boyer-Vorkink: max skewness = worst returns)
  liquidity      tight spread + real OI, so the ticket is actually exitable
  catalyst       an event in the option's life — but only rewarded when IV is
                 still cheap (otherwise you overpay and IV-crush eats the move)

Everything is config-driven (weights + guardrails) so the system can be tuned
and adapted without touching the logic. See DEFAULT_LOTTERY_CONFIG.
"""
from __future__ import annotations

import json
import math
from typing import Any, Dict, List, Optional


DEFAULT_LOTTERY_CONFIG: Dict[str, Any] = {
    "weights": {
        "vol_cheapness": 0.15,
        "momentum": 0.30,
        "convexity": 0.10,
        "liquidity": 0.10,
        "catalyst": 0.05,
        # explosiveness: fat-tail potential of the underlying. Naked far-OTM
        # lottery payoffs live in high-vol names, so this is weighted heavily —
        # the backtest showed the low-vol tilt was actively counterproductive.
        "explosiveness": 0.30,
    },
    # realized-vol level (annualised) that maps to a full explosiveness score
    "vol_level_cap": 1.2,
    # convexity sweet spot: Gaussian reward centred on target sigma OTM
    "convexity_sigma_target": 2.0,
    "convexity_sigma_band": 0.75,
    # how strongly aligned momentum moves the momentum score
    "momentum_gain": 5.0,
    # how strongly the realized-vs-implied gap moves the cheapness score
    "cheapness_gain": 1.5,
    # entry construction (used by the backtest / live picker, not scoring)
    "dte_target": 14,
    "otm_sigma_min": 1.5,
    "otm_sigma_max": 2.5,
    "guardrails": {
        "max_iv_rank": 0.85,
        "max_strike_sigma": 3.0,
        "max_spread_pct": 0.40,
        "min_open_interest": 50,
    },
}


def load_lottery_config(config_path: str = "config.json") -> Dict[str, Any]:
    """Merge a config.json 'lottery' block over the defaults (shallow per key)."""
    cfg = json.loads(json.dumps(DEFAULT_LOTTERY_CONFIG))  # deep copy
    try:
        with open(config_path) as fh:
            user = (json.load(fh) or {}).get("lottery") or {}
    except Exception:
        user = {}
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(cfg.get(k), dict):
            cfg[k].update(v)
        else:
            cfg[k] = v
    return cfg


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _cheapness_signal01(realized_vol: float, iv: float, gain: float) -> float:
    """Map (realized - implied)/implied into [0,1]; >0.5 means option is cheap."""
    if iv is None or iv <= 0 or realized_vol is None:
        return 0.5
    signal = (realized_vol - iv) / iv
    return _clamp01(0.5 + signal * gain)


def score_candidate(candidate: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Score one far-OTM candidate. Returns {score, components, disqualified, reason}."""
    g = cfg.get("guardrails", {})
    iv_rank = candidate.get("iv_rank")
    strike_sigma = candidate.get("strike_sigma")
    spread_pct = candidate.get("spread_pct")
    oi = candidate.get("open_interest")

    # ── hard disqualifiers (the Boyer-Vorkink traps) ───────────────────────────
    reasons = []
    if iv_rank is not None and iv_rank > g.get("max_iv_rank", 0.85):
        reasons.append(f"IV rank {iv_rank:.2f} > {g['max_iv_rank']}")
    if strike_sigma is not None and strike_sigma > g.get("max_strike_sigma", 3.0):
        reasons.append(f"strike {strike_sigma:.1f}σ > {g['max_strike_sigma']}σ")
    if spread_pct is not None and spread_pct > g.get("max_spread_pct", 0.40):
        reasons.append(f"spread {spread_pct:.0%} > {g['max_spread_pct']:.0%}")
    if oi is not None and oi < g.get("min_open_interest", 50):
        reasons.append(f"OI {oi} < {g['min_open_interest']}")
    disqualified = bool(reasons)

    # ── component scores, each in [0,1] ────────────────────────────────────────
    cheap01 = _cheapness_signal01(
        candidate.get("realized_vol"), candidate.get("iv"), cfg.get("cheapness_gain", 1.5)
    )
    ivr = iv_rank if iv_rank is not None else 0.5
    vol_cheapness = 0.5 * cheap01 + 0.5 * (1.0 - _clamp01(ivr))

    mom = candidate.get("momentum") or 0.0
    momentum = _clamp01(0.5 + mom * cfg.get("momentum_gain", 5.0))

    if strike_sigma is None:
        convexity = 0.0
    else:
        target = cfg.get("convexity_sigma_target", 2.0)
        band = cfg.get("convexity_sigma_band", 0.75)
        convexity = math.exp(-((strike_sigma - target) ** 2) / (2 * band * band))

    sp = spread_pct if spread_pct is not None else 0.2
    max_sp = g.get("max_spread_pct", 0.40)
    spread_score = 1.0 - _clamp01(sp / max_sp) if max_sp > 0 else 0.0
    oi_val = oi if oi is not None else 0
    oi_score = _clamp01(math.log10(oi_val + 1) / math.log10(5000))
    liquidity = 0.6 * spread_score + 0.4 * oi_score

    if candidate.get("has_catalyst"):
        catalyst = 0.2 + 0.8 * cheap01  # gated on cheap IV
    else:
        catalyst = 0.5  # no event = no information either way

    vol_level = candidate.get("vol_level")
    if vol_level is None:
        explosiveness = 0.0
    else:
        explosiveness = _clamp01(vol_level / cfg.get("vol_level_cap", 1.2))

    components = {
        "vol_cheapness": round(vol_cheapness, 4),
        "momentum": round(momentum, 4),
        "convexity": round(convexity, 4),
        "liquidity": round(liquidity, 4),
        "catalyst": round(catalyst, 4),
        "explosiveness": round(explosiveness, 4),
    }
    w = cfg.get("weights", {})
    score = sum(w.get(k, 0.0) * v for k, v in components.items())

    return {
        "score": round(score, 4),
        "components": components,
        "disqualified": disqualified,
        "reason": "; ".join(reasons) if reasons else "",
    }


def select_best(
    candidates: List[Dict[str, Any]], cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Return the highest-scoring eligible candidate (with score attached), or None."""
    scored = []
    for c in candidates:
        res = score_candidate(c, cfg)
        if res["disqualified"]:
            continue
        enriched = dict(c)
        enriched["_score"] = res["score"]
        enriched["_components"] = res["components"]
        scored.append(enriched)
    if not scored:
        return None
    return max(scored, key=lambda c: c["_score"])


__all__ = [
    "DEFAULT_LOTTERY_CONFIG",
    "load_lottery_config",
    "score_candidate",
    "select_best",
]
