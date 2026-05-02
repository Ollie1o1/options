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
from typing import Dict, Optional

import numpy as np
import pandas as pd

from . import data_fetching as _df

DEFAULT_WEIGHTS: Dict[str, float] = {
    "iv_rank":        1.0 / 7,
    "vrp":            1.0 / 7,
    "term_structure": 1.0 / 7,
    "skew":           1.0 / 7,
    "funding_z":      1.0 / 7,
    "basis":          1.0 / 7,
    "liquidity":      1.0 / 7,
}


def _clip01(x: float) -> float:
    if x is None or not math.isfinite(x):
        return 0.5
    return max(0.0, min(1.0, float(x)))


def _atm_iv(chain: pd.DataFrame, expiration_filter=None) -> Optional[float]:
    """Mark IV of the contract closest to ATM, optionally restricted to one expiry."""
    if chain.empty:
        return None
    df = chain
    if expiration_filter is not None:
        df = df[df["expiration"] == expiration_filter]
    if df.empty:
        return None
    df = df.copy()
    df["moneyness"] = (df["strike"] - df["underlying_price"]).abs()
    atm = df.sort_values("moneyness").head(2)
    if atm.empty:
        return None
    iv = atm["mark_iv"].mean()
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
    """Front-expiry ATM IV vs back-expiry ATM IV.

    Backwardation (front > back) is fear / event-pricing — favors the front
    leg in a calendar OR avoiding short-front-month premium-selling.
    Contango (front < back) is normal — favors short-front premium.

    Returns 0.5 when flat. <0.5 when backwardated, >0.5 when in contango.
    """
    if chain.empty:
        return 0.5
    expiries = sorted(chain["expiration"].unique())
    if len(expiries) < 2:
        return 0.5
    front_iv = _atm_iv(chain, expiration_filter=expiries[0])
    back_iv = _atm_iv(chain, expiration_filter=expiries[-1])
    if front_iv is None or back_iv is None or front_iv <= 0:
        return 0.5
    slope = (back_iv - front_iv) / front_iv
    score = 0.5 + 0.5 * math.tanh(slope / 0.10)
    return _clip01(score)


def score_skew(chain: pd.DataFrame) -> float:
    """25-delta put IV vs 25-delta call IV (front expiry).

    Crypto skew is usually call-heavy (bullish skew) in bull, put-heavy in fear.
    For premium-selling, neutral skew (≈ 1.0 ratio) is best; extreme skew suggests
    one side is cheap-vol-dumping. Scored: closer to neutral ⇒ higher score.
    """
    if chain.empty:
        return 0.5
    expiries = sorted(chain["expiration"].unique())
    if not expiries:
        return 0.5
    front = chain[chain["expiration"] == expiries[0]]
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
) -> pd.DataFrame:
    """Score every contract in the chain. Adds per-component columns and quality_score."""
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
    s_iv_rank        = score_iv_rank(chain, history)
    s_vrp            = score_vrp(chain, history)
    s_term_structure = score_term_structure(chain)
    s_skew           = score_skew(chain)
    s_funding_z      = score_funding_z(funding_history)
    s_basis          = score_basis(funding)

    df = chain.copy()
    df["iv_rank_score"]        = s_iv_rank
    df["vrp_score"]            = s_vrp
    df["term_structure_score"] = s_term_structure
    df["skew_score"]           = s_skew
    df["funding_z_score"]      = s_funding_z
    df["basis_score"]          = s_basis
    df["liquidity_score"]      = df.apply(score_liquidity, axis=1)

    df["quality_score"] = (
        w["iv_rank"]        * df["iv_rank_score"]
        + w["vrp"]            * df["vrp_score"]
        + w["term_structure"] * df["term_structure_score"]
        + w["skew"]           * df["skew_score"]
        + w["funding_z"]      * df["funding_z_score"]
        + w["basis"]          * df["basis_score"]
        + w["liquidity"]      * df["liquidity_score"]
    )
    return df
