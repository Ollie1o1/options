"""Crypto options backtester — walk-forward simulation.

We don't have a paid historical chain feed, so for now we *synthesize* chains
from spot history using BS pricing with a realistic IV assumption (rolling
realized vol scaled by the historical implied/realized ratio for crypto, ~1.10).

This gives us:
  • Backwardly-validated signal performance for the options-pricing components
    (iv_rank, vrp, term_structure, skew, moneyness_fit, dte_fit)
  • Per-strategy realized P&L over multiple regimes
  • Per-component IC analogous to the equity calibration loop
  • A baseline before we accumulate real historical chain snapshots

What we DON'T validate from a synthetic chain:
  • Funding / basis components (no history without paid feeds — held neutral)
  • Real bid/ask spreads (we synthesize a 5% spread proxy)
  • Open interest dynamics (held flat at synthetic OI)

When real snapshots accumulate via chain_snapshot.py, this backtester will
prefer them over synthetic chains for the dates they cover.
"""
from __future__ import annotations

import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.utils import bs_call, bs_put

from . import chain_snapshot as _snap
from . import data_fetching as _df
from . import regime as _regime
from . import scoring as _scoring
from . import strategy as _strategy

# ── Configuration ────────────────────────────────────────────────────────

# Strikes synthesized as a percentage band around spot.
_STRIKE_BAND = (0.70, 1.30)
# Strike granularity in % of spot (5% → strikes at 70%, 75%, 80% ... 130%).
_STRIKE_STEP = 0.025

# DTEs to synthesize per test date (matches typical Deribit weekly + monthly cadence).
_SYNTH_DTES = (7, 14, 21, 30, 45, 60, 90)

# Implied/realized vol ratio observed historically for BTC/ETH (~1.05–1.15).
# Multiply realized vol by this to get a baseline implied vol assumption.
_IV_RV_RATIO = 1.10

# Smile: ATM IV scaled up linearly with |log-moneyness|. Typical BTC smile
# adds ~5-15 vol points at 20% OTM.
_SMILE_SLOPE = 0.30

# Risk-free rate proxy.
_R = 0.045
# Dividend yield (no yield on crypto).
_Q = 0.0

# Exit rules — match the equity defaults.
_TP_LONG_PREMIUM   = 1.0    # +100%
_SL_LONG_PREMIUM   = -0.50  # -50%
_TP_CREDIT_SPREAD  = 0.50   # capture 50% of credit
_SL_CREDIT_SPREAD  = -1.00  # full credit lost (will be replaced by structural floor)
_TIME_EXIT_DTE     = 1      # close at expiry day - 1

# Synthetic OI / liquidity proxies.
_SYNTH_OI_ATM      = 1500
_SYNTH_OI_WING     = 250
_SYNTH_SPREAD_PCT  = 0.05


# ── Synthesis ────────────────────────────────────────────────────────────

def _smile_iv(base_iv: float, strike: float, spot: float) -> float:
    """Apply a simple symmetric smile around ATM."""
    if spot <= 0:
        return base_iv
    log_m = abs(math.log(max(strike, 1.0) / spot))
    return float(max(0.05, base_iv * (1.0 + _SMILE_SLOPE * log_m)))


def _synthetic_oi(strike: float, spot: float) -> int:
    """OI peaks ATM, decays at wings — rough proxy until we have real snapshots."""
    log_m = abs(math.log(max(strike, 1.0) / max(spot, 1.0)))
    weight = math.exp(-3.0 * log_m)
    return int(_SYNTH_OI_WING + (_SYNTH_OI_ATM - _SYNTH_OI_WING) * weight)


def synthesize_chain(
    spot: float,
    asof_date: _dt.date,
    dtes: Tuple[int, ...] = _SYNTH_DTES,
    base_iv: float = 0.60,
) -> pd.DataFrame:
    """Generate a synthetic options chain for `asof_date` at `spot` price.

    Strikes range from 70%–130% of spot in 2.5% steps. For each (DTE, strike)
    pair we generate both a call and a put priced via BS with a smile-adjusted IV.
    """
    rows = []
    strikes_pct = np.arange(_STRIKE_BAND[0], _STRIKE_BAND[1] + 1e-9, _STRIKE_STEP)
    for dte in dtes:
        T = max(dte / 365.0, 1e-6)
        exp_date = asof_date + _dt.timedelta(days=int(dte))
        for sp in strikes_pct:
            strike = round(spot * float(sp), 0)
            iv = _smile_iv(base_iv, strike, spot)
            for opt_type, pricer in (("call", bs_call), ("put", bs_put)):
                price = float(pricer(spot, strike, T, _R, iv, _Q))
                if price <= 0 or not math.isfinite(price):
                    continue
                spread = price * _SYNTH_SPREAD_PCT
                bid = max(0.0, price - spread / 2)
                ask = price + spread / 2
                rows.append({
                    "instrument_name": f"SYNTH-{exp_date:%d%b%y}-{int(strike)}-{opt_type[0].upper()}".upper(),
                    "underlying_price": float(spot),
                    "strike": float(strike),
                    "type": opt_type,
                    "expiration": exp_date,
                    "dte": float(dte),
                    "mark_iv": float(iv),
                    "bid_iv": float(iv),
                    "ask_iv": float(iv),
                    "bid_price": float(bid),
                    "ask_price": float(ask),
                    "mark_price": float(price),
                    "mid_price": float(price),
                    "volume": float(_synthetic_oi(strike, spot)) * 0.10,
                    "open_interest": float(_synthetic_oi(strike, spot)),
                    "spread_pct": float(spread / max(price, 1e-9)),
                })
    return pd.DataFrame(rows)


# ── Trade simulation ─────────────────────────────────────────────────────

def _bs_value(opt_type: str, S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0:
        intrinsic = max(0.0, (S - K) if opt_type == "call" else (K - S))
        return intrinsic
    pricer = bs_call if opt_type == "call" else bs_put
    val = float(pricer(S, K, T, _R, sigma, _Q))
    return max(0.0, val)


def _walk_forward_long(
    pick: pd.Series,
    spot_after: pd.Series,
    iv_after: pd.Series,
    entry_price: float,
    exit_dte: int,
) -> Tuple[float, str, int]:
    """Simulate a long-premium trade day-by-day until TP/SL/expiry."""
    K = float(pick["strike"])
    opt_type = str(pick["type"]).lower()
    target_dte = float(pick["dte"])
    held = 0
    for t, (date, S) in enumerate(spot_after.items(), start=1):
        if t > exit_dte:
            return -_SL_LONG_PREMIUM * 0 + 0.0, "Time Exit (max held)", held  # safety
        held = t
        sigma = float(iv_after.iloc[t - 1]) if t - 1 < len(iv_after) else float(pick["mark_iv"])
        T_remaining = max((target_dte - t) / 365.0, 1e-6)
        if T_remaining <= 1e-6:
            # expiry — intrinsic only
            value = max(0.0, (S - K) if opt_type == "call" else (K - S))
            pnl_pct = (value - entry_price) / max(entry_price, 1e-9)
            return float(pnl_pct), "Expiry", held
        value = _bs_value(opt_type, float(S), K, T_remaining, sigma)
        pnl_pct = (value - entry_price) / max(entry_price, 1e-9)
        if pnl_pct >= _TP_LONG_PREMIUM:
            return float(_TP_LONG_PREMIUM), "Take Profit", held
        if pnl_pct <= _SL_LONG_PREMIUM:
            return float(_SL_LONG_PREMIUM), "Stop Loss", held
    return float(pnl_pct), "Time Exit", held


def _walk_forward_credit_spread(
    short_strike: float,
    long_strike: float,
    leg_type: str,                # 'put' or 'call'
    target_dte: int,
    entry_credit: float,
    width: float,
    spot_after: pd.Series,
    iv_after: pd.Series,
    exit_dte: int,
) -> Tuple[float, str, int]:
    """Simulate a credit spread day-by-day."""
    held = 0
    for t, (date, S) in enumerate(spot_after.items(), start=1):
        if t > exit_dte:
            break
        held = t
        sigma = float(iv_after.iloc[t - 1]) if t - 1 < len(iv_after) else 0.60
        T_remaining = max((target_dte - t) / 365.0, 1e-6)
        # Cost to close = short value (we owe) - long value (we recover)
        short_val = _bs_value(leg_type, float(S), float(short_strike), T_remaining, sigma)
        long_val = _bs_value(leg_type, float(S), float(long_strike), T_remaining, sigma)
        cost_to_close = short_val - long_val
        # Profit = entry_credit - cost_to_close (at entry cost_to_close ≈ entry_credit, P&L ≈ 0)
        pnl_dollar = entry_credit - cost_to_close
        pnl_pct = pnl_dollar / max(entry_credit, 1e-9)
        # TP at 50% of credit captured
        if pnl_pct >= _TP_CREDIT_SPREAD:
            return float(_TP_CREDIT_SPREAD), "Take Profit", held
        # SL: structural floor = -(width/credit - 1)
        sl_pct = -((width / max(entry_credit, 1e-9)) - 1.0)
        if pnl_pct <= sl_pct:
            return float(sl_pct), "Stop Loss (max)", held
    # Reached time exit / expiry — pnl is whatever we computed last
    return float(pnl_pct), "Time Exit", held


# ── Walk-forward driver ─────────────────────────────────────────────────

@dataclass
class BacktestResult:
    currency: str
    start_date: str
    end_date: str
    trades: pd.DataFrame
    summary: Dict[str, Dict[str, float]] = field(default_factory=dict)
    component_ic: Dict[str, float] = field(default_factory=dict)


def _realized_vol_at(history: pd.DataFrame, asof_idx: int, window: int = 30) -> float:
    """30-day annualized realized vol up to `asof_idx`."""
    if asof_idx < window + 1:
        return 0.60  # fallback for early dates
    log_r = np.log(history["Close"].astype(float)).diff().iloc[: asof_idx + 1]
    sd = float(log_r.tail(window).std())
    if not math.isfinite(sd) or sd <= 0:
        return 0.60
    return sd * math.sqrt(365)


def run_backtest(
    currency: str = "BTC",
    days_back: int = 365,
    frequency_days: int = 7,
    progress_callback=None,
) -> Optional[BacktestResult]:
    """Walk forward through the last `days_back` of spot history.

    On each test date (every `frequency_days` apart), synthesize a chain,
    run the screener, log paper trades for each surfaced strategy bucket's
    top pick, and walk each forward to its exit.
    """
    # Pull enough spot to compute 30d rvol from day 1 of the test window.
    history = _df.get_spot_history(currency, days=days_back + 90)
    if history is None or history.empty or len(history) < 60:
        return None

    history = history.reset_index(drop=True)
    if "Date" in history.columns:
        history["Date"] = pd.to_datetime(history["Date"])
    elif history.index.name == "Date":
        history = history.reset_index()
        history["Date"] = pd.to_datetime(history["Date"])

    closes = history["Close"].astype(float).values
    dates = pd.to_datetime(history["Date"]).dt.date.values

    # First test date: enough lookback for 30d rvol + 200d MA classification
    first_idx = 200
    if first_idx >= len(history) - 7:
        return None

    # Build a list of test dates.
    test_indices = list(range(first_idx, len(history) - 7, frequency_days))

    trade_rows: List[dict] = []
    n_tests = len(test_indices)
    for n, idx in enumerate(test_indices, 1):
        if progress_callback:
            try:
                progress_callback(n, n_tests)
            except Exception:
                pass
        spot = float(closes[idx])
        asof = pd.Timestamp(dates[idx]).date()
        rvol = _realized_vol_at(history, idx)
        base_iv = rvol * _IV_RV_RATIO

        # Try a real snapshot first, else synthesize.
        chain = _snap.load_snapshot(asof.isoformat(), currency)
        chain_source = "snapshot"
        if chain is None or chain.empty:
            chain = synthesize_chain(spot, asof, base_iv=base_iv)
            chain_source = "synthetic"
        if chain.empty:
            continue

        # Provide history-up-to-asof so VRP / IV-rank / regime are point-in-time correct.
        history_up_to = history.iloc[: idx + 1].reset_index(drop=True)
        regime_obj = _regime.classify_btc(history_up_to)
        regime_label = regime_obj.label if regime_obj else "chop"

        # No funding/basis history → use neutral placeholders so they don't bias IC.
        scored = _scoring.score_chain(
            chain, history_up_to, funding=None,
            funding_history=pd.DataFrame(),  # neutral
            regime_multipliers=_regime.REGIME_WEIGHT_MULTIPLIERS.get(regime_label),
        )
        tradable = scored[
            (scored["dte"].between(5, 60))
            & (scored["mark_iv"] > 0)
            & (scored["mid_price"] > 0)
        ].copy()
        if tradable.empty:
            continue

        _chain_cols = (
            "iv_rank_score", "vrp_score", "term_structure_score",
            "skew_score", "funding_z_score", "basis_score",
            "funding_divergence_score", "oi_surge_score",
            "stablecoin_flow_score",
        )
        _present = [c for c in _chain_cols if c in scored.columns]
        chain_q = (sum(float(scored[c].iloc[0]) for c in _present) / max(1, len(_present)))

        # For each strategy with regime fit ≥ 0.55, simulate the top pick.
        for strat in _strategy.STRATEGIES:
            if float(strat.regime_fit.get(regime_label, 0)) < 0.55:
                continue
            if strat.direction == "spread_credit" and strat.leg_type != "both":
                spread_picks = _strategy.build_credit_spread_candidates(
                    tradable, strat, regime_label, chain_q, top_n=1,
                )
                if spread_picks.empty:
                    continue
                row = spread_picks.iloc[0]
                # Walk forward up to 28 days (or DTE-1, whichever is sooner)
                exit_window = min(28, int(row["dte"]) - 1)
                if exit_window <= 1:
                    continue
                # Slice forward IV: hold flat at base_iv for synthetic, since
                # we can't observe future IV. This biases stats toward delta
                # P&L; we accept that.
                spot_after = pd.Series(
                    closes[idx + 1: idx + 1 + exit_window],
                    index=dates[idx + 1: idx + 1 + exit_window],
                )
                iv_after = pd.Series([base_iv] * exit_window)
                pnl_pct, reason, held = _walk_forward_credit_spread(
                    short_strike=float(row["short_strike"]),
                    long_strike=float(row["long_strike"]),
                    leg_type=strat.leg_type,
                    target_dte=int(row["dte"]),
                    entry_credit=float(row["net_credit"]),
                    width=float(row["width"]),
                    spot_after=spot_after,
                    iv_after=iv_after,
                    exit_dte=exit_window,
                )
                trade_rows.append({
                    "asof": asof.isoformat(),
                    "strategy": strat.name,
                    "regime": regime_label,
                    "chain_source": chain_source,
                    "chain_quality": chain_q,
                    "iv_rank_score":        float(scored["iv_rank_score"].iloc[0]),
                    "vrp_score":            float(scored["vrp_score"].iloc[0]),
                    "term_structure_score": float(scored["term_structure_score"].iloc[0]),
                    "skew_score":           float(scored["skew_score"].iloc[0]),
                    "moneyness_fit":        float(row.get("moneyness_fit", 0)) if "moneyness_fit" in row.index else 0.0,
                    "dte_fit":              float(row.get("dte_fit", 0)) if "dte_fit" in row.index else 0.0,
                    "score": float(row["score"]),
                    "entry_credit": float(row["net_credit"]),
                    "max_loss": float(row["max_loss"]),
                    "pnl_pct": float(pnl_pct),
                    "pnl_usd": float(pnl_pct) * float(row["net_credit"]),
                    "exit_reason": reason,
                    "days_held": int(held),
                    "type": "spread",
                })
            else:
                long_picks = _strategy.score_for_strategy(
                    tradable, strat, regime_label, chain_q,
                ).head(1)
                if long_picks.empty:
                    continue
                row = long_picks.iloc[0]
                entry_price = float(row.get("ask_price") or row.get("mark_price") or 0)
                if entry_price <= 0:
                    continue
                exit_window = min(28, int(row["dte"]) - 1)
                if exit_window <= 1:
                    continue
                spot_after = pd.Series(
                    closes[idx + 1: idx + 1 + exit_window],
                    index=dates[idx + 1: idx + 1 + exit_window],
                )
                iv_after = pd.Series([base_iv] * exit_window)
                pnl_pct, reason, held = _walk_forward_long(
                    row, spot_after, iv_after, entry_price, exit_window,
                )
                trade_rows.append({
                    "asof": asof.isoformat(),
                    "strategy": strat.name,
                    "regime": regime_label,
                    "chain_source": chain_source,
                    "chain_quality": chain_q,
                    "iv_rank_score":        float(scored["iv_rank_score"].iloc[0]),
                    "vrp_score":            float(scored["vrp_score"].iloc[0]),
                    "term_structure_score": float(scored["term_structure_score"].iloc[0]),
                    "skew_score":           float(scored["skew_score"].iloc[0]),
                    "moneyness_fit":        float(row.get("moneyness_fit", 0)),
                    "dte_fit":              float(row.get("dte_fit", 0)),
                    "score": float(row["strategy_score"]),
                    "entry_price": float(entry_price),
                    "pnl_pct": float(pnl_pct),
                    "pnl_usd": float(pnl_pct) * float(entry_price),
                    "exit_reason": reason,
                    "days_held": int(held),
                    "type": "long_premium",
                })

    if not trade_rows:
        return None
    trades = pd.DataFrame(trade_rows)
    return BacktestResult(
        currency=currency.upper(),
        start_date=str(dates[first_idx]),
        end_date=str(dates[-1]),
        trades=trades,
        summary=_summarize(trades),
        component_ic=_component_ic(trades),
    )


# ── Aggregation ──────────────────────────────────────────────────────────

def _summarize(trades: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per-strategy breakdown: count, win rate, avg P&L, profit factor."""
    out: Dict[str, Dict[str, float]] = {}
    if trades.empty:
        return out
    for strat, df in trades.groupby("strategy"):
        gains = df[df["pnl_pct"] > 0]["pnl_pct"].sum()
        losses = -df[df["pnl_pct"] < 0]["pnl_pct"].sum()
        pf = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
        out[strat] = {
            "n": int(len(df)),
            "win_rate": float((df["pnl_pct"] > 0).mean()),
            "avg_pnl_pct": float(df["pnl_pct"].mean()),
            "median_pnl_pct": float(df["pnl_pct"].median()),
            "total_pnl_usd": float(df["pnl_usd"].sum()),
            "profit_factor": pf,
            "max_win_pct": float(df["pnl_pct"].max()),
            "max_loss_pct": float(df["pnl_pct"].min()),
        }
    # All-strategies row
    df = trades
    gains = df[df["pnl_pct"] > 0]["pnl_pct"].sum()
    losses = -df[df["pnl_pct"] < 0]["pnl_pct"].sum()
    pf_all = float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0
    out["__ALL__"] = {
        "n": int(len(df)),
        "win_rate": float((df["pnl_pct"] > 0).mean()),
        "avg_pnl_pct": float(df["pnl_pct"].mean()),
        "median_pnl_pct": float(df["pnl_pct"].median()),
        "total_pnl_usd": float(df["pnl_usd"].sum()),
        "profit_factor": pf_all,
        "max_win_pct": float(df["pnl_pct"].max()),
        "max_loss_pct": float(df["pnl_pct"].min()),
    }
    return out


def _component_ic(trades: pd.DataFrame) -> Dict[str, float]:
    """Spearman correlation between each component score and pnl_pct."""
    if trades.empty:
        return {}
    out: Dict[str, float] = {}
    components = (
        "iv_rank_score", "vrp_score", "term_structure_score", "skew_score",
        "moneyness_fit", "dte_fit", "score", "chain_quality",
    )
    pnl = trades["pnl_pct"]
    for comp in components:
        if comp not in trades.columns:
            continue
        s = trades[comp]
        if s.nunique() < 2:
            out[comp] = 0.0
            continue
        try:
            corr = s.corr(pnl, method="spearman")
            out[comp] = float(corr) if math.isfinite(corr) else 0.0
        except Exception:
            out[comp] = 0.0
    return out
