"""Spread / iron-condor enrichment.

Reuses the per-leg component scores already computed by ``_score_fetched_data``
and attaches them to each spread/condor row alongside spread-specific features
(``credit_to_width_ratio``, ``return_on_risk``, ``delta_neutral_score``, …).
A new ``quality_score`` is then computed from ``credit_spread_weights`` /
``iron_condor_weights`` so multi-leg structures get scored on their own merits
instead of an averaged single-leg score.

Output columns mirror the single-leg ``picks`` DataFrame (``pop_score``,
``ev_score``, ``entry_iv``, ``entry_delta``, …) so the auto-log path can
persist component scores without a separate code branch — the optimizer can
then learn ``credit_spread_weights`` from real ledger data.
"""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Component scores carried through from the short leg.
_SHORT_LEG_SCORE_COLS = (
    "pop_score", "ev_score", "rr_score", "liquidity_score",
    "momentum_score", "iv_rank_score", "theta_score",
    "iv_advantage_score", "vrp_score", "iv_mispricing_score",
    "skew_align_score", "vega_risk_score", "term_structure_score",
    "catalyst_score", "em_realism_score", "gamma_theta_score",
    "gex_score", "gamma_magnitude_score", "gamma_pin_score",
    "iv_velocity_score", "max_pain_score", "oi_change_score",
    "option_rvol_score", "pcr_score", "sentiment_score_norm",
    "spread_score", "trader_pref_score",
)

# yfinance column name → paper_trades schema column name
_GREEK_COL_MAP = {
    "impliedVolatility": "entry_iv",
    "delta": "entry_delta",
    "gamma": "entry_gamma",
    "vega":  "entry_vega",
    "theta": "entry_theta",
}

DEFAULT_SPREAD_WEIGHTS: Dict[str, float] = {
    "pop":            0.25,
    "credit_to_width": 0.20,
    "iv_rank":        0.15,
    "return_on_risk": 0.10,
    "liquidity":      0.10,
    "theta":          0.08,
    "spread":         0.05,
    "momentum":       0.04,
    "catalyst":       0.03,
}

DEFAULT_IRON_WEIGHTS: Dict[str, float] = {
    "pop":             0.30,
    "credit_to_width": 0.20,
    "delta_neutral":   0.15,
    "iv_rank":         0.12,
    "liquidity":       0.10,
    "theta":           0.08,
    "spread":          0.05,
}


def _lookup_leg(
    df_scored: pd.DataFrame,
    symbol: str,
    expiration,
    strike: float,
    opt_type: str,
) -> Optional[pd.Series]:
    candidates = df_scored[
        (df_scored["symbol"] == symbol)
        & (df_scored["expiration"] == expiration)
        & (df_scored["type"] == opt_type)
        & (np.isclose(df_scored["strike"].astype(float), float(strike)))
    ]
    if candidates.empty:
        return None
    return candidates.iloc[0]


def _short_leg_option_type(spread_row) -> str:
    typ = str(spread_row.get("type", "")).lower()
    return "put" if "put" in typ else "call"


def _safe_score(val) -> Optional[float]:
    if val is None:
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if np.isnan(f):
        return None
    return f


def _weighted_score(row: dict, feature_to_col: Dict[str, str], weights: Dict[str, float]) -> Optional[float]:
    """Renormalize over only the features that have a non-null score."""
    weighted_sum = 0.0
    weight_total = 0.0
    for feature, weight in weights.items():
        col = feature_to_col.get(feature)
        if col is None:
            continue
        val = _safe_score(row.get(col))
        if val is None:
            continue
        weighted_sum += val * float(weight)
        weight_total += float(weight)
    if weight_total <= 0:
        return None
    return float(np.clip(weighted_sum / weight_total, 0.0, 1.0))


def enrich_credit_spreads(
    spreads_df: pd.DataFrame,
    df_scored: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Attach short-leg component scores + spread-specific features to each row,
    recompute ``quality_score`` from ``credit_spread_weights``."""
    if spreads_df.empty:
        return spreads_df.copy()

    weights = config.get("credit_spread_weights", DEFAULT_SPREAD_WEIGHTS)
    enriched_rows = []

    for _, spread in spreads_df.iterrows():
        row_out = spread.to_dict()
        opt_type = _short_leg_option_type(spread)
        short_leg = _lookup_leg(df_scored, spread["symbol"], spread["expiration"], spread["short_strike"], opt_type)
        long_leg = _lookup_leg(df_scored, spread["symbol"], spread["expiration"], spread["long_strike"], opt_type)

        if short_leg is None:
            enriched_rows.append(row_out)
            continue

        # Carry component scores from the short leg (the risk-bearing leg)
        for col in _SHORT_LEG_SCORE_COLS:
            if col in df_scored.columns:
                row_out[col] = short_leg.get(col)
        for src_col, dst_col in _GREEK_COL_MAP.items():
            if src_col in df_scored.columns:
                row_out[dst_col] = short_leg.get(src_col)

        # Spread-specific structural features
        spread_width = abs(float(spread["short_strike"]) - float(spread["long_strike"]))
        net_credit = float(spread.get("net_credit", 0) or 0)
        max_profit = float(spread.get("max_profit", 0) or 0)
        max_loss = float(spread.get("max_loss", 0) or 0)
        row_out["spread_width"] = spread_width

        c2w = net_credit / spread_width if spread_width > 0 else 0.0
        row_out["credit_to_width_ratio"] = c2w
        # 0-1 normalize: 0.20 (filter floor) → 0.0, 0.50 → 1.0
        row_out["credit_to_width_score"] = float(np.clip((c2w - 0.20) / 0.30, 0.0, 1.0))

        ror = max_profit / max_loss if max_loss > 0 else 0.0
        row_out["return_on_risk_ratio"] = ror
        row_out["return_on_risk_score"] = float(np.clip((ror - 0.20) / 0.80, 0.0, 1.0))

        if long_leg is not None:
            short_prem = float(short_leg.get("premium") or 0)
            long_prem = float(long_leg.get("premium") or 0)
            if short_prem > 0:
                row_out["protection_cost_ratio"] = long_prem / short_prem

        # Recompute quality_score from credit_spread_weights
        feature_to_col = {
            "pop":            "pop_score",
            "credit_to_width": "credit_to_width_score",
            "iv_rank":        "iv_rank_score",
            "return_on_risk": "return_on_risk_score",
            "liquidity":      "liquidity_score",
            "theta":          "theta_score",
            "spread":         "spread_score",
            "momentum":       "momentum_score",
            "catalyst":       "catalyst_score",
        }
        new_score = _weighted_score(row_out, feature_to_col, weights)
        if new_score is not None:
            row_out["quality_score"] = new_score

        # Convenience for downstream code
        row_out["entry_price"] = net_credit

        enriched_rows.append(row_out)

    out = pd.DataFrame(enriched_rows)
    if "quality_score" in out.columns:
        out = out.sort_values("quality_score", ascending=False).reset_index(drop=True)
    return out


def enrich_iron_condors(
    condors_df: pd.DataFrame,
    df_scored: pd.DataFrame,
    config: dict,
) -> pd.DataFrame:
    """Average per-leg scores across all 4 legs; add condor-specific features."""
    if condors_df.empty:
        return condors_df.copy()

    weights = config.get("iron_condor_weights", DEFAULT_IRON_WEIGHTS)
    enriched_rows = []

    for _, condor in condors_df.iterrows():
        row_out = condor.to_dict()
        sym = condor["symbol"]
        exp = condor["expiration"]

        sp = _lookup_leg(df_scored, sym, exp, condor["short_put_strike"], "put")
        lp = _lookup_leg(df_scored, sym, exp, condor["long_put_strike"], "put")
        sc = _lookup_leg(df_scored, sym, exp, condor["short_call_strike"], "call")
        lc = _lookup_leg(df_scored, sym, exp, condor["long_call_strike"], "call")
        legs = [leg for leg in (sp, lp, sc, lc) if leg is not None]
        if len(legs) < 4:
            enriched_rows.append(row_out)
            continue

        # Average each component across all 4 legs (the structure has 4-leg risk)
        for col in _SHORT_LEG_SCORE_COLS:
            if col not in df_scored.columns:
                continue
            vals = [_safe_score(leg.get(col)) for leg in legs]
            vals = [v for v in vals if v is not None]
            row_out[col] = float(np.mean(vals)) if vals else None

        # Greeks taken from short-put leg (canonical entry leg).
        for src_col, dst_col in _GREEK_COL_MAP.items():
            if src_col in df_scored.columns:
                row_out[dst_col] = sp.get(src_col)

        # Condor-specific features
        put_width = float(condor["short_put_strike"]) - float(condor["long_put_strike"])
        call_width = float(condor["long_call_strike"]) - float(condor["short_call_strike"])
        spread_width = max(put_width, call_width, 0.01)
        total_credit = float(condor.get("total_credit", 0) or 0)
        net_delta = float(condor.get("net_delta", 0) or 0)

        row_out["spread_width"] = spread_width
        c2w = total_credit / spread_width if spread_width > 0 else 0.0
        row_out["credit_to_width_ratio"] = c2w
        # iron condor credit/width is naturally lower than vertical: 0.10 floor → 0, 0.30 → 1
        row_out["credit_to_width_score"] = float(np.clip((c2w - 0.10) / 0.20, 0.0, 1.0))

        # Delta neutrality: net_delta near 0 is good. abs(0.00) → 1.0; abs(0.10+) → 0.0
        row_out["delta_neutral_score"] = float(np.clip(1.0 - abs(net_delta) / 0.10, 0.0, 1.0))

        # Combined PoP: 1 - (P(touch put) + P(touch call)) approximated by short deltas.
        sp_delta = abs(float(sp.get("delta") or 0))
        sc_delta = float(sc.get("delta") or 0)
        pop_combined = max(0.0, 1.0 - (sp_delta + sc_delta))
        row_out["pop_score"] = float(np.clip(pop_combined, 0.0, 1.0))

        feature_to_col = {
            "pop":             "pop_score",
            "credit_to_width": "credit_to_width_score",
            "delta_neutral":   "delta_neutral_score",
            "iv_rank":         "iv_rank_score",
            "liquidity":       "liquidity_score",
            "theta":           "theta_score",
            "spread":          "spread_score",
        }
        new_score = _weighted_score(row_out, feature_to_col, weights)
        if new_score is not None:
            row_out["quality_score"] = new_score

        row_out["entry_price"] = total_credit
        enriched_rows.append(row_out)

    out = pd.DataFrame(enriched_rows)
    if "quality_score" in out.columns:
        out = out.sort_values("quality_score", ascending=False).reset_index(drop=True)
    return out
