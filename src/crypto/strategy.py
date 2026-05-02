"""Strategy-aware scoring layer.

Per-contract scoring depends on which strategy you're considering. A 30-delta
short put is excellent for a Bull Put Spread but useless as a Long Call. The
same contract should produce different scores under different strategies.

Each strategy defines:
  - regime_fit:     how well it suits BULL / CHOP / BEAR
  - target_moneyness:  preferred strike-vs-spot ratio (e.g. 0.92 for short put)
  - target_dte_band:   preferred days-to-expiry window
  - direction:         "long_premium" | "short_premium" | "spread_credit" | "spread_debit"

Final per-(contract, strategy) score:
    chain_quality * regime_fit * moneyness_fit * dte_fit * (per-contract liquidity)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class StrategyDef:
    name: str
    direction: str               # long_premium | short_premium | spread_credit
    leg_type: str                # call | put | both (for spreads)
    target_moneyness: float      # strike/spot ratio (e.g. 0.92 = 8% OTM put)
    moneyness_tolerance: float   # how forgiving the fit is (in ratio units)
    target_dte_min: int
    target_dte_max: int
    regime_fit: Dict[str, float] = field(default_factory=dict)


# Strategy catalog — keep this small at v1. Add more once these calibrate.
STRATEGIES: List[StrategyDef] = [
    # ── Long premium (directional bets) ──────────────────────────────────
    StrategyDef(
        name="Long Call",
        direction="long_premium",
        leg_type="call",
        target_moneyness=1.00,           # ATM
        moneyness_tolerance=0.04,        # ±4%
        target_dte_min=14, target_dte_max=35,
        regime_fit={"bull": 1.00, "chop": 0.45, "bear": 0.20},
    ),
    StrategyDef(
        name="Long Put",
        direction="long_premium",
        leg_type="put",
        target_moneyness=1.00,
        moneyness_tolerance=0.04,
        target_dte_min=14, target_dte_max=35,
        regime_fit={"bull": 0.20, "chop": 0.45, "bear": 1.00},
    ),
    # ── Credit spreads (defined-risk premium selling) ────────────────────
    StrategyDef(
        name="Bull Put",                  # short put + long lower-strike put
        direction="spread_credit",
        leg_type="put",
        target_moneyness=0.93,           # 7% OTM short
        moneyness_tolerance=0.04,
        target_dte_min=21, target_dte_max=45,
        regime_fit={"bull": 1.00, "chop": 0.85, "bear": 0.30},
    ),
    StrategyDef(
        name="Bear Call",                 # short call + long higher-strike call
        direction="spread_credit",
        leg_type="call",
        target_moneyness=1.07,           # 7% OTM short
        moneyness_tolerance=0.04,
        target_dte_min=21, target_dte_max=45,
        regime_fit={"bull": 0.30, "chop": 0.85, "bear": 1.00},
    ),
    # ── Iron Condor (range-bound) ────────────────────────────────────────
    StrategyDef(
        name="Iron Condor",
        direction="spread_credit",
        leg_type="both",
        target_moneyness=1.0,            # symmetric around spot
        moneyness_tolerance=0.04,
        target_dte_min=30, target_dte_max=60,
        regime_fit={"bull": 0.40, "chop": 1.00, "bear": 0.40},
    ),
]


def _moneyness_fit(strike: float, spot: float, target: float, tol: float) -> float:
    """Smooth bell around the target ratio. Width = `tol` * 4 to score tail-off."""
    if spot <= 0:
        return 0.0
    ratio = strike / spot
    z = (ratio - target) / max(tol, 1e-6)
    # Gaussian-ish: 1.0 at target, ~0.6 at ±tol, ~0.1 at ±2.5*tol
    return float(math.exp(-0.5 * z * z))


def _dte_fit(dte: float, dte_min: int, dte_max: int) -> float:
    """Plateau at 1.0 inside [dte_min, dte_max], linear falloff outside."""
    if dte_min <= dte <= dte_max:
        return 1.0
    if dte < dte_min:
        # Falls off as DTE shrinks below dte_min; 0 by half of dte_min
        return max(0.0, dte / max(dte_min, 1))
    # dte > dte_max: falls off; 0 at 2× dte_max
    if dte_max <= 0:
        return 0.0
    over = dte - dte_max
    return max(0.0, 1.0 - over / float(dte_max))


def score_for_strategy(
    chain: pd.DataFrame,
    strategy: StrategyDef,
    regime_label: str,
    chain_quality: float,
) -> pd.DataFrame:
    """Add per-contract `strategy_score` for `strategy` to a copy of `chain`.

    For long-premium strategies, scores apply directly to single contracts.
    For credit-spread strategies, the score applies to the SHORT leg; pairing
    it with a long leg is done by spread_builder.
    """
    if chain.empty:
        return chain

    # Filter to relevant leg type
    if strategy.leg_type == "call":
        subset = chain[chain["type"] == "call"].copy()
    elif strategy.leg_type == "put":
        subset = chain[chain["type"] == "put"].copy()
    else:
        subset = chain.copy()
    if subset.empty:
        return subset

    regime_mult = float(strategy.regime_fit.get(regime_label.lower(), 0.5))
    subset["regime_fit"] = regime_mult
    subset["moneyness_fit"] = subset.apply(
        lambda r: _moneyness_fit(
            float(r["strike"]),
            float(r["underlying_price"]),
            strategy.target_moneyness,
            strategy.moneyness_tolerance,
        ),
        axis=1,
    )
    subset["dte_fit"] = subset["dte"].apply(
        lambda d: _dte_fit(float(d), strategy.target_dte_min, strategy.target_dte_max)
    )
    # Liquidity already on the row from the upstream scorer.
    liq = subset.get("liquidity_score", pd.Series([0.5] * len(subset), index=subset.index))
    subset["strategy_score"] = (
        max(0.10, float(chain_quality))   # floor so an unfavorable chain doesn't zero everything
        * (0.5 + 0.5 * regime_mult)       # soften regime — still rank a contract even if regime mismatched
        * subset["moneyness_fit"]
        * subset["dte_fit"]
        * (0.5 + 0.5 * liq)               # liquidity contribution
    )
    subset["strategy_name"] = strategy.name
    return subset.sort_values("strategy_score", ascending=False)


def build_credit_spread_candidates(
    chain: pd.DataFrame,
    strategy: StrategyDef,
    regime_label: str,
    chain_quality: float,
    width_targets: Tuple[float, ...] = (1000.0, 2000.0, 5000.0),
    top_n: int = 5,
) -> pd.DataFrame:
    """Enumerate credit-spread candidates for a Bull Put / Bear Call.

    Approach: rank short-leg candidates with `score_for_strategy`, then for each
    top-ranked short, find the closest matching long-leg by `width_targets` (in
    USD strike distance). Only same-expiry pairs.

    Returns a DataFrame of spread candidates with columns:
      strategy_name, expiration, dte, short_strike, long_strike, width,
      short_iv, long_iv, net_credit, max_profit, max_loss, breakeven, score.
    """
    if strategy.direction != "spread_credit" or strategy.leg_type == "both":
        return pd.DataFrame()
    if chain.empty:
        return pd.DataFrame()

    short_candidates = score_for_strategy(chain, strategy, regime_label, chain_quality)
    short_candidates = short_candidates.head(top_n * 3)  # buffer for pairing failures
    if short_candidates.empty:
        return pd.DataFrame()

    rows = []
    for _, short_row in short_candidates.iterrows():
        exp = short_row["expiration"]
        short_strike = float(short_row["strike"])
        # Find the same-expiry long leg whose strike is `width` away in the
        # OTM direction relative to underlying.
        same_exp = chain[(chain["expiration"] == exp) & (chain["type"] == strategy.leg_type)]
        if same_exp.empty:
            continue
        for width in width_targets:
            if strategy.leg_type == "put":
                # Bull Put: long leg further OTM = lower strike
                target_long = short_strike - width
            else:
                # Bear Call: long leg further OTM = higher strike
                target_long = short_strike + width
            same_exp_d = same_exp.copy()
            same_exp_d["dist"] = (same_exp_d["strike"] - target_long).abs()
            long_row = same_exp_d.sort_values("dist").iloc[0]
            actual_long = float(long_row["strike"])
            actual_width = abs(short_strike - actual_long)
            if actual_width <= 0:
                continue
            # Net credit: sell short bid, buy long ask. Use mid as proxy.
            short_credit_per_contract = float(short_row.get("mid_price") or 0)
            long_debit_per_contract = float(long_row.get("mid_price") or 0)
            net_credit = short_credit_per_contract - long_debit_per_contract
            if net_credit <= 0:
                continue  # skip unprofitable pairings
            max_profit = net_credit
            max_loss = actual_width - net_credit
            if max_loss <= 0:
                continue
            risk_reward = max_profit / max_loss if max_loss > 0 else 0.0
            # Score: penalize spreads with poor risk/reward, reward those near
            # the strategy's target width.
            rr_score = min(1.0, risk_reward / 0.40)  # 0.40 RR = full score
            # Use the SHORT leg's strategy_score as the foundation
            base_score = float(short_row["strategy_score"])
            spread_score = base_score * (0.6 + 0.4 * rr_score)
            be = (short_strike - net_credit) if strategy.leg_type == "put" else (short_strike + net_credit)
            rows.append({
                "strategy_name": strategy.name,
                "expiration": exp,
                "dte": int(short_row["dte"]),
                "short_strike": short_strike,
                "long_strike": actual_long,
                "width": actual_width,
                "short_iv": float(short_row["mark_iv"]),
                "long_iv": float(long_row["mark_iv"]),
                "net_credit": net_credit,
                "max_profit": max_profit,
                "max_loss": max_loss,
                "risk_reward": risk_reward,
                "breakeven": be,
                "underlying_price": float(short_row["underlying_price"]),
                "score": spread_score,
                "short_oi": int(short_row.get("open_interest") or 0),
                "long_oi": int(long_row.get("open_interest") or 0),
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)


def recommended_strategies_for_regime(regime_label: str) -> List[str]:
    """Return strategy names ordered by regime fit, descending."""
    label = regime_label.lower()
    scored = [(s.name, s.regime_fit.get(label, 0.5)) for s in STRATEGIES]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [name for name, fit in scored if fit >= 0.55]


def build_iron_condor_candidates(
    chain: pd.DataFrame,
    regime_label: str,
    chain_quality: float,
    width_targets: Tuple[float, ...] = (1000.0, 2000.0, 5000.0),
    top_n: int = 5,
) -> pd.DataFrame:
    """Pair a Bear Call (call-side credit spread) with a Bull Put (put-side credit
    spread) at the SAME expiry to form an iron condor.

    Width is symmetric across both wings (uses the same width target). Iron condor
    profits if underlying expires between the two short strikes; max loss is
    on either wing.
    """
    if chain.empty:
        return pd.DataFrame()

    bull_put = next((s for s in STRATEGIES if s.name == "Bull Put"), None)
    bear_call = next((s for s in STRATEGIES if s.name == "Bear Call"), None)
    if bull_put is None or bear_call is None:
        return pd.DataFrame()

    # Reuse the credit-spread builder to find ranked short legs per side.
    bp_picks = build_credit_spread_candidates(
        chain, bull_put, regime_label, chain_quality, width_targets=width_targets, top_n=top_n * 2,
    )
    bc_picks = build_credit_spread_candidates(
        chain, bear_call, regime_label, chain_quality, width_targets=width_targets, top_n=top_n * 2,
    )
    if bp_picks.empty or bc_picks.empty:
        return pd.DataFrame()

    rows = []
    # Pair on (expiration, width) — same expiry + same wing width = symmetric IC.
    for _, p in bp_picks.iterrows():
        exp = p["expiration"]
        width = p["width"]
        match = bc_picks[
            (bc_picks["expiration"] == exp)
            & (bc_picks["width"].between(width * 0.7, width * 1.3))
        ]
        if match.empty:
            continue
        c = match.iloc[0]
        net_credit = float(p["net_credit"]) + float(c["net_credit"])
        max_loss = max(float(p["max_loss"]), float(c["max_loss"]))
        max_profit = net_credit
        if max_loss <= 0:
            continue
        risk_reward = max_profit / max_loss
        rr_score = min(1.0, risk_reward / 0.40)
        score = 0.5 * (float(p["score"]) + float(c["score"])) * (0.6 + 0.4 * rr_score)
        rows.append({
            "strategy_name": "Iron Condor",
            "expiration": exp,
            "dte": int(p["dte"]),
            "short_put_strike":  float(p["short_strike"]),
            "long_put_strike":   float(p["long_strike"]),
            "short_call_strike": float(c["short_strike"]),
            "long_call_strike":  float(c["long_strike"]),
            "width": float(width),
            "net_credit": net_credit,
            "max_profit": max_profit,
            "max_loss": max_loss,
            "risk_reward": risk_reward,
            "underlying_price": float(p["underlying_price"]),
            "score": score,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    return df.sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
