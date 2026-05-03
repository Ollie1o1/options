"""Crypto options scoring layer — 7 components.

Stripped down from the equity 23-component stack to avoid the overfit trap.
All scores in [0, 1]. Uniform weights initially; calibrate with IC analysis
once 30+ paper trades close per structure.

Components:
  iv_rank        — current ATM IV vs 60-day distribution
  vrp            — implied vol minus realized (positive ⇒ richly priced)
  term_structure — front IV vs back IV (contango/backwardation)
  skew           — 25-delta put IV vs 25-delta call IV
  funding_z      — perp funding z-score over 30-day window (extremes are signal)
  basis          — perp mark vs spot index (cash-and-carry signal)
  liquidity      — open interest + bid-ask tightness
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from . import data_fetching as _df

DEFAULT_WEIGHTS: Dict[str, float] = {
    "iv_rank":             1.0 / 9,
    "vrp":                 1.0 / 9,
    "term_structure":      1.0 / 9,
    "skew":                1.0 / 9,
    "funding_z":           1.0 / 9,
    "basis":               1.0 / 9,
    "funding_divergence":  1.0 / 9,
    "oi_surge":            1.0 / 9,
    "liquidity":           1.0 / 9,
}


def _clip01(x: float) -> float:
    if x is None or not math.isfinite(x):
        return 0.5
    return max(0.0, min(1.0, float(x)))


def _pick_expiry(chain: pd.DataFrame, target_dte: float) -> Optional[Any]:
    """Return the expiration date in `chain` closest to `target_dte` calendar days.

    Picking a target-DTE-aware expiry is essential — comparing IV across mixed
    expiries (e.g. 1-DTE gamma vol vs 9-month vega) is meaningless.
    """
    if chain.empty or "dte" not in chain.columns:
        return None
    by_exp = chain.groupby("expiration")["dte"].first().reset_index()
    if by_exp.empty:
        return None
    by_exp["dist"] = (by_exp["dte"] - float(target_dte)).abs()
    return by_exp.sort_values("dist").iloc[0]["expiration"]


def _atm_iv(chain: pd.DataFrame, expiration_filter=None,
             target_dte: float = 30.0) -> Optional[float]:
    """Mark IV of the contract closest to ATM at a SINGLE expiry.

    If no expiration_filter is provided, picks the expiry closest to
    `target_dte` so the IV reflects a meaningful tenor (default: 30 days).
    Averages call + put IV at the closest strike to remove put/call IV asymmetry.
    """
    if chain.empty:
        return None
    if expiration_filter is None:
        expiration_filter = _pick_expiry(chain, target_dte)
        if expiration_filter is None:
            return None
    df = chain[chain["expiration"] == expiration_filter]
    if df.empty:
        return None
    df = df.copy()
    df["moneyness"] = (df["strike"] - df["underlying_price"]).abs()
    # Take the closest strike, average call + put IV (ATM straddle IV).
    closest_strike = df.sort_values("moneyness").iloc[0]["strike"]
    atm_rows = df[df["strike"] == closest_strike]
    if atm_rows.empty:
        return None
    iv = atm_rows["mark_iv"].mean()
    return float(iv) if math.isfinite(iv) else None


def score_iv_rank(chain: pd.DataFrame, history: pd.DataFrame) -> float:
    """ATM IV percentile vs trailing 60-day realized-vol distribution.

    Higher percentile ⇒ richer premium ⇒ favors short-premium structures.
    Returns the percentile directly (0..1).
    """
    iv = _atm_iv(chain)
    if iv is None or history is None or history.empty:
        return 0.5
    log_r = np.log(history["Close"].astype(float)).diff().dropna()
    if len(log_r) < 60:
        return 0.5
    rolling = log_r.rolling(30).std() * math.sqrt(365)
    rolling = rolling.dropna().tail(60)
    if rolling.empty:
        return 0.5
    pct = float((rolling < iv).sum() / len(rolling))
    return _clip01(pct)


def score_vrp(chain: pd.DataFrame, history: pd.DataFrame) -> float:
    """Volatility risk premium: implied minus realized, normalized.

    Positive VRP ⇒ implied is richer than realized ⇒ supports short premium.
    Mapped to [0, 1] via tanh: VRP=0 → 0.5, VRP=+0.20 → ~0.95, VRP=-0.20 → ~0.05.
    """
    iv = _atm_iv(chain)
    if iv is None:
        return 0.5
    rv = _df.realized_vol(np.log(history["Close"].astype(float)).diff(), window=30)
    if not math.isfinite(rv):
        return 0.5
    vrp = iv - rv
    # tanh squash: scale so |VRP|=20% ≈ saturation
    score = 0.5 + 0.5 * math.tanh(vrp / 0.10)
    return _clip01(score)


def score_term_structure(chain: pd.DataFrame) -> float:
    """Compare 7-DTE ATM IV vs 30-DTE ATM IV.

    Backwardation (7-DTE > 30-DTE) is fear / event-pricing — favors avoiding
    short-front-month premium-selling. Contango (7-DTE < 30-DTE) is normal
    and supports short-front premium structures.

    Comparing min vs max expiry (e.g. 1-DTE vs 9-month) saturates the score
    almost always because of the gamma-vs-vega regime change — not signal.
    Picking 7-DTE and 30-DTE keeps both points in the same vol regime.

    Returns 0.5 when flat. <0.5 when backwardated, >0.5 when in contango.
    """
    if chain.empty:
        return 0.5
    front_iv = _atm_iv(chain, target_dte=7.0)
    back_iv  = _atm_iv(chain, target_dte=30.0)
    if front_iv is None or back_iv is None or front_iv <= 0:
        return 0.5
    slope = (back_iv - front_iv) / front_iv
    score = 0.5 + 0.5 * math.tanh(slope / 0.10)
    return _clip01(score)


def score_skew(chain: pd.DataFrame) -> float:
    """25-delta put IV vs 25-delta call IV at the 30-DTE expiry.

    Crypto skew is usually call-heavy (bullish skew) in bull, put-heavy in fear.
    For premium-selling, neutral skew (≈ 1.0 ratio) is best; extreme skew suggests
    one side is cheap-vol-dumping. Scored: closer to neutral ⇒ higher score.
    """
    if chain.empty:
        return 0.5
    target_exp = _pick_expiry(chain, 30.0)
    if target_exp is None:
        return 0.5
    front = chain[chain["expiration"] == target_exp]
    if front.empty:
        return 0.5
    underlying = float(front["underlying_price"].iloc[0])
    # Find ~25-delta strikes by moneyness proxy (no Greeks in summary feed).
    # 25Δ put ≈ 92% strike, 25Δ call ≈ 108% (rough rule of thumb at typical crypto IV)
    p25_target = underlying * 0.92
    c25_target = underlying * 1.08
    puts = front[front["type"] == "put"].copy()
    calls = front[front["type"] == "call"].copy()
    if puts.empty or calls.empty:
        return 0.5
    puts["dist"] = (puts["strike"] - p25_target).abs()
    calls["dist"] = (calls["strike"] - c25_target).abs()
    p_iv = float(puts.sort_values("dist").iloc[0]["mark_iv"])
    c_iv = float(calls.sort_values("dist").iloc[0]["mark_iv"])
    if c_iv <= 0:
        return 0.5
    ratio = p_iv / c_iv
    # Ratio 1.0 → 1.0 score (neutral). Ratio 1.30 (heavy put skew) → ~0.5. Ratio 0.70 → ~0.5.
    score = 1.0 - min(1.0, abs(math.log(ratio)) / 0.30)
    return _clip01(score)


def score_funding_z(funding_history: pd.DataFrame) -> float:
    """Z-score of latest funding vs 30-bar history.

    Extreme funding signals crowded positioning — a fade signal. We score
    higher when |z| is large (extremes), regardless of sign.
    """
    if funding_history is None or funding_history.empty:
        return 0.5
    rates = funding_history["funding_rate"].astype(float).dropna()
    if len(rates) < 10:
        return 0.5
    mean = float(rates.mean())
    std = float(rates.std())
    latest = float(rates.iloc[-1])
    if std <= 0:
        return 0.5
    z = (latest - mean) / std
    # Extreme z (|z| > 2) → high score (≈ 0.9). Neutral z=0 → 0.5.
    score = _clip01(0.5 + 0.5 * math.tanh(abs(z) / 2.0))
    return score


def score_funding_divergence(aggregated_funding: Optional[Dict[str, Any]]) -> float:
    """Score cross-exchange funding divergence.

    Wide divergence between Binance / Bybit / OKX / dYdX funding signals:
      • crowded positioning on one venue (the outlier)
      • fading-the-outlier mean reversion when divergence narrows
      • sometimes a real perp-perp arb if you can hold both sides

    Score is 0.5 (neutral) when all venues align, rising toward 1.0 as the
    cross-venue spread widens. Saturation at ~10% annualized spread.
    """
    if aggregated_funding is None:
        return 0.5
    div = aggregated_funding.get("divergence") or {}
    if not div or div.get("venue_count", 0) < 2:
        return 0.5
    spread_per_8h = float(div.get("max_8h", 0)) - float(div.get("min_8h", 0))
    # Convert to annualized for human-meaningful scaling
    spread_ann = spread_per_8h * 3 * 365
    # tanh squash: 5% annualized → ~0.7, 10% → ~0.88, 20% → ~0.96
    score = 0.5 + 0.5 * math.tanh(abs(spread_ann) / 0.05)
    return _clip01(score)


def score_oi_surge(oi_z: Optional[float]) -> float:
    """Score open-interest surge magnitude.

    OI z-score is computed against the 30-day OI distribution. Both directions
    of extreme OI (build-up or unwind) carry information:
      • |z| > 1.5 with high funding  → crowded longs, fade signal (sell calls)
      • |z| > 1.5 with low funding   → crowded shorts, fade signal (sell puts)
      • |z| < 0.5                    → no information, neutral
    Score rises with |z|, saturating around |z| = 2.5.
    """
    if oi_z is None or not math.isfinite(oi_z):
        return 0.5
    score = _clip01(0.5 + 0.5 * math.tanh(abs(float(oi_z)) / 1.5))
    return score


def score_basis(funding: Optional[Dict[str, float]]) -> float:
    """Spot-perp basis. Wide basis ⇒ basis trade opportunity.

    |basis_pct| > 0.20% (annualized via 8h funding × 3 × 365) is significant.
    Since basis_pct is point-in-time, we score on absolute magnitude.
    """
    if funding is None:
        return 0.5
    basis = funding.get("basis_pct")
    if basis is None or not math.isfinite(basis):
        return 0.5
    score = _clip01(0.5 + 0.5 * math.tanh(abs(basis) / 0.005))
    return score


def score_liquidity(row: pd.Series) -> float:
    """Per-contract liquidity: open interest scaled + tight bid/ask."""
    oi = float(row.get("open_interest") or 0.0)
    spread_pct = float(row.get("spread_pct") or 1.0)
    # OI score: 0 → 0, log-scaled to ≈ 1.0 at OI≈1000
    oi_score = math.log10(1.0 + max(0.0, oi)) / 3.0
    oi_score = _clip01(oi_score)
    # Spread score: tight spread (≤5%) → 1.0, ≥40% → 0
    spread_score = 1.0 - min(1.0, max(0.0, (spread_pct - 0.05) / 0.35))
    return _clip01(0.6 * oi_score + 0.4 * spread_score)


def score_chain(
    chain: pd.DataFrame,
    history: pd.DataFrame,
    funding: Optional[Dict[str, float]],
    funding_history: pd.DataFrame,
    weights: Optional[Dict[str, float]] = None,
    regime_multipliers: Optional[Dict[str, float]] = None,
    aggregated_funding: Optional[Dict[str, Any]] = None,
    oi_z: Optional[float] = None,
) -> pd.DataFrame:
    """Score every contract in the chain. Adds per-component columns and quality_score.

    `aggregated_funding` (optional) is the dict from
    `data_fetching.get_aggregated_funding()`. When supplied, enables the
    `funding_divergence` component. When None, that component falls back
    to neutral (0.5) so legacy / backtest callers still work.

    `oi_z` (optional) is the open-interest z-score from
    `data_fetching.oi_z_score(get_oi_history(...))`. Enables the `oi_surge`
    component when supplied; held neutral otherwise.
    """
    if chain.empty:
        return chain
    w = dict(weights or DEFAULT_WEIGHTS)
    if regime_multipliers:
        for k, mult in regime_multipliers.items():
            if k in w:
                w[k] = w[k] * float(mult)
    # Renormalize
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}

    # Chain-level (constant across rows for a single fetch)
    s_iv_rank            = score_iv_rank(chain, history)
    s_vrp                = score_vrp(chain, history)
    s_term_structure     = score_term_structure(chain)
    s_skew               = score_skew(chain)
    s_funding_z          = score_funding_z(funding_history)
    s_basis              = score_basis(funding)
    s_funding_divergence = score_funding_divergence(aggregated_funding)
    s_oi_surge           = score_oi_surge(oi_z)

    df = chain.copy()
    df["iv_rank_score"]            = s_iv_rank
    df["vrp_score"]                = s_vrp
    df["term_structure_score"]     = s_term_structure
    df["skew_score"]               = s_skew
    df["funding_z_score"]          = s_funding_z
    df["basis_score"]              = s_basis
    df["funding_divergence_score"] = s_funding_divergence
    df["oi_surge_score"]           = s_oi_surge
    df["liquidity_score"]          = df.apply(score_liquidity, axis=1)

    df["quality_score"] = (
        w["iv_rank"]            * df["iv_rank_score"]
        + w["vrp"]                * df["vrp_score"]
        + w["term_structure"]     * df["term_structure_score"]
        + w["skew"]               * df["skew_score"]
        + w["funding_z"]          * df["funding_z_score"]
        + w["basis"]              * df["basis_score"]
        + w["funding_divergence"] * df["funding_divergence_score"]
        + w["oi_surge"]           * df["oi_surge_score"]
        + w["liquidity"]          * df["liquidity_score"]
    )
    return df
