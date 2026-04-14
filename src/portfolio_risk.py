#!/usr/bin/env python3
"""
Portfolio-level risk aggregation: Greek exposure, GEX, and Monte Carlo VaR.

RiskAggregator reads open paper trades from SQLite, fetches current market data,
computes portfolio Greeks and GEX, and estimates 1-day Value-at-Risk via MJD.
"""

import math
import logging
import sqlite3
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import bs_call, bs_put, bs_delta, bs_gamma, bs_vega, safe_float
from .data_fetching import get_risk_free_rate

logger = logging.getLogger(__name__)

# Lazy-load yfinance to avoid startup hang
_yf = None

def _get_yf():
    """Lazily import yfinance on first use."""
    global _yf
    if _yf is None:
        import yfinance as _yf_mod
        _yf = _yf_mod
    return _yf

import time as _time
_IV_CACHE: dict = {}          # {key: (iv_value, timestamp)}
_IV_CACHE_TTL = 900           # 15 minutes


class RiskAggregator:
    """Aggregate portfolio Greeks, GEX, and VaR across open paper trades."""

    def __init__(
        self,
        db_path: str = "paper_trades.db",
        config: Optional[Dict] = None,
    ) -> None:
        self.db_path = db_path
        self.config = config or {}

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_open_trades(self) -> List[Dict]:
        """Load all OPEN trades from the paper trades database."""
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM trades WHERE status='OPEN'"
            ).fetchall()
            conn.close()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("Could not load open trades: %s", exc)
            return []

    def _fetch_spot(self, ticker: str) -> Optional[float]:
        """Fetch current spot price via fast_info, with history fallback."""
        try:
            tkr = _get_yf().Ticker(ticker)
            fi = getattr(tkr, "fast_info", None)
            if fi is not None:
                lp = safe_float(getattr(fi, "last_price", None))
                if lp and lp > 0:
                    return lp
            hist = tkr.history(period="5d")
            if not hist.empty:
                val = safe_float(hist["Close"].iloc[-1])
                if val and val > 0:
                    return val
        except Exception:
            pass
        return None

    _FALLBACK_IV = 0.30

    def _get_current_iv(
        self,
        ticker: str,
        expiration: str,
        strike: float,
        opt_type: str,
    ) -> Tuple[float, str]:
        """Fetch implied volatility from the option chain.

        Returns ``(iv, source)`` where ``source`` is one of:
          * ``"market"`` — freshly fetched from the option chain
          * ``"cache"``  — reused from the in-process IV cache
          * ``"fallback"`` — yfinance unavailable / empty chain / bad value;
             caller should treat downstream Greeks as stale.
        """
        cache_key = f"{ticker}:{str(expiration)[:10]}:{strike:.2f}:{opt_type}"
        cached = _IV_CACHE.get(cache_key)
        if cached is not None:
            iv_val, ts = cached
            if _time.monotonic() - ts < _IV_CACHE_TTL:
                return iv_val, "cache"
        try:
            tkr = _get_yf().Ticker(ticker)
            exps = tkr.options
            if not exps:
                return self._FALLBACK_IV, "fallback"
            # Pick the closest available expiration
            target = pd.Timestamp(expiration)
            exp_ts = [pd.Timestamp(e) for e in exps]
            closest = min(exp_ts, key=lambda e: abs((e - target).days))
            chain = tkr.option_chain(closest.strftime("%Y-%m-%d"))
            tbl = chain.calls if opt_type.lower() == "call" else chain.puts
            if tbl is None or tbl.empty:
                return self._FALLBACK_IV, "fallback"
            row = tbl.iloc[(tbl["strike"] - strike).abs().argsort()[:1]]
            iv = safe_float(row["impliedVolatility"].iloc[0])
            if iv and 0.01 < iv < 10.0:
                _IV_CACHE[cache_key] = (iv, _time.monotonic())
                return iv, "market"
        except Exception as exc:
            logger.debug("IV fetch failed for %s %s %s: %s", ticker, expiration, strike, exc)
        return self._FALLBACK_IV, "fallback"

    def _dte(self, expiration: str) -> float:
        """Days to expiry as a fraction of a year (floored at 0)."""
        try:
            exp_dt = pd.Timestamp(expiration)
            now = pd.Timestamp.now()
            # Strip timezone info from both to avoid TypeError on mixed-tz subtraction
            if exp_dt.tzinfo is not None:
                exp_dt = exp_dt.tz_localize(None)
            if now.tzinfo is not None:
                now = now.tz_localize(None)
            days = max((exp_dt.normalize() - now.normalize()).days, 0)
            return days / 365.0
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_open_positions_with_greeks(self, rfr: Optional[float] = None) -> pd.DataFrame:
        """
        Return a DataFrame of open positions enriched with live Greeks.

        Added columns: spot, T_years, sigma, delta, gamma, vega, gex
        GEX per contract = sign × gamma × S² × 0.01 × 100
            sign: +1 for calls (dealers short gamma), -1 for puts
        """
        if rfr is None:
            rfr = get_risk_free_rate()

        trades = self._load_open_trades()
        if not trades:
            return pd.DataFrame()

        rows = []
        for t in trades:
            ticker = t.get("ticker", "")
            expiration = t.get("expiration", "")
            strike = safe_float(t.get("strike")) or 0.0
            opt_type = (t.get("type") or "call").lower()

            if not ticker or not expiration or strike <= 0:
                continue

            spot = self._fetch_spot(ticker)
            if spot is None or spot <= 0:
                continue

            T = self._dte(expiration)
            if T <= 0:
                # Skip already-expired positions — their Greeks are meaningless
                continue

            sigma, iv_source = self._get_current_iv(ticker, expiration, strike, opt_type)
            if not sigma or sigma <= 0:
                sigma, iv_source = self._FALLBACK_IV, "fallback"

            try:
                delta = float(bs_delta(opt_type, spot, strike, T, rfr, sigma))
                gamma = float(bs_gamma(spot, strike, T, rfr, sigma))
                vega = float(bs_vega(spot, strike, T, rfr, sigma))
            except Exception:
                delta, gamma, vega = 0.0, 0.0, 0.0

            # GEX convention: dealers short calls → +GEX; short puts → −GEX
            gex_sign = 1.0 if opt_type == "call" else -1.0
            gex = gex_sign * gamma * (spot ** 2) * 0.01 * 100.0

            row = dict(t)
            row.update(
                spot=spot,
                T_years=T,
                sigma=sigma,
                iv_source=iv_source,
                delta=delta,
                gamma=gamma,
                vega=vega,
                gex=gex,
            )
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows)

    def get_portfolio_greeks(self, rfr: Optional[float] = None) -> Dict:
        """
        Return aggregate portfolio Greeks across all open positions.

        Keys: portfolio_delta, portfolio_gamma, portfolio_gex,
              portfolio_vega, n_positions, positions_df
        """
        df = self.get_open_positions_with_greeks(rfr=rfr)
        if df.empty:
            return {
                "portfolio_delta": 0.0,
                "portfolio_gamma": 0.0,
                "portfolio_gex": 0.0,
                "portfolio_vega": 0.0,
                "n_positions": 0,
                "n_iv_fallback": 0,
                "positions_df": df,
            }
        n_iv_fallback = (
            int((df["iv_source"] == "fallback").sum())
            if "iv_source" in df.columns
            else 0
        )
        return {
            "portfolio_delta": float(df["delta"].sum()),
            "portfolio_gamma": float(df["gamma"].sum()),
            "portfolio_gex": float(df["gex"].sum()),
            "portfolio_vega": float(df["vega"].sum()),
            "n_positions": len(df),
            "n_iv_fallback": n_iv_fallback,
            "positions_df": df,
        }

    def calculate_portfolio_var(
        self,
        confidence: float = 0.95,
        n_simulations: int = 10_000,
        horizon_days: int = 1,
        rfr: Optional[float] = None,
        random_seed: int = 42,
    ) -> Dict:
        """
        Estimate portfolio VaR / CVaR via MJD Monte Carlo.

        For each open position we simulate horizon_days price paths with the
        same MJD parameters used in simulation.py, re-price the option at each
        scenario's terminal price, and sum P&L across all positions to build a
        portfolio P&L distribution.

        Returns: var_95, cvar_95, mean_pnl, n_positions, pnl_distribution
        """
        empty = {
            "var_95": 0.0,
            "cvar_95": 0.0,
            "mean_pnl": 0.0,
            "n_positions": 0,
            "n_iv_fallback": 0,
            "pnl_distribution": np.array([]),
        }

        trades = self._load_open_trades()
        if not trades:
            return empty

        if rfr is None:
            rfr = get_risk_free_rate()

        df = self.get_open_positions_with_greeks(rfr=rfr)
        if df.empty:
            return empty

        # MJD parameters (configurable via config.json)
        jump_intensity = float(self.config.get("var_jump_intensity", 2.0))
        jump_mean = float(self.config.get("var_jump_mean", -0.02))
        jump_vol = float(self.config.get("var_jump_vol", 0.04))

        rng = np.random.default_rng(random_seed)
        T_horizon = horizon_days / 365.0
        n_steps = max(horizon_days, 1)
        dt = T_horizon / n_steps

        portfolio_pnl = np.zeros(n_simulations)

        for _, pos in df.iterrows():
            spot = float(pos["spot"])
            strike = float(pos.get("strike", 0))
            T_curr = float(pos["T_years"])
            sigma = float(pos["sigma"])
            opt_type = str(pos.get("type", "call")).lower()
            entry_price = safe_float(pos.get("entry_price"))

            # Skip positions with missing or zero entry price (P&L would be wrong)
            if not entry_price or entry_price <= 0:
                continue
            if spot <= 0 or strike <= 0 or sigma <= 0:
                continue

            # MJD drift correction
            jump_corr = jump_intensity * (math.exp(jump_mean + 0.5 * jump_vol ** 2) - 1.0)
            drift = (rfr - jump_corr - 0.5 * sigma ** 2) * dt
            diffusion = sigma * math.sqrt(dt)

            # Simulate price paths
            z = rng.standard_normal((n_simulations, n_steps))
            jump_occurs = rng.random((n_simulations, n_steps)) < (jump_intensity * dt)
            jump_sizes = rng.normal(jump_mean, jump_vol, (n_simulations, n_steps))
            log_steps = drift + diffusion * z + np.where(jump_occurs, jump_sizes, 0.0)
            terminal_log = np.sum(log_steps, axis=1)
            S_T = spot * np.exp(terminal_log)

            # Re-price option at terminal price (1 contract = 100 shares)
            T_remaining = max(T_curr - T_horizon, 0.0)
            try:
                if T_remaining <= 0:
                    # Expired: intrinsic value only
                    if opt_type == "call":
                        exit_prices = np.maximum(S_T - strike, 0.0)
                    else:
                        exit_prices = np.maximum(strike - S_T, 0.0)
                else:
                    if opt_type == "call":
                        exit_prices = bs_call(S_T, strike, T_remaining, rfr, sigma)
                    else:
                        exit_prices = bs_put(S_T, strike, T_remaining, rfr, sigma)
                exit_prices = np.asarray(exit_prices, dtype=float)
                strategy_name = str(pos.get("strategy_name", ""))
                is_short = any(k in strategy_name.lower() for k in ("short", "credit", "covered", "cash-secured", "cash secured", "naked", "iron condor", "sell"))
                if is_short:
                    pnl = (entry_price - exit_prices) * 100.0
                else:
                    pnl = (exit_prices - entry_price) * 100.0
            except Exception:
                continue

            portfolio_pnl += pnl

        if not np.any(portfolio_pnl != 0):
            return empty

        # VaR / CVaR at requested confidence level
        percentile = (1.0 - confidence) * 100.0
        var = float(-np.percentile(portfolio_pnl, percentile))
        cvar_mask = portfolio_pnl < -var
        cvar = float(-portfolio_pnl[cvar_mask].mean()) if cvar_mask.any() else var

        n_iv_fallback = (
            int((df["iv_source"] == "fallback").sum())
            if "iv_source" in df.columns
            else 0
        )
        return {
            "var_95": var,
            "cvar_95": cvar,
            "mean_pnl": float(portfolio_pnl.mean()),
            "n_positions": len(df),
            "n_iv_fallback": n_iv_fallback,
            "pnl_distribution": portfolio_pnl,
        }

    def is_risk_off_required(self, config: Optional[Dict] = None) -> Tuple[bool, str]:
        """
        Return (True, reason) when absolute portfolio GEX exceeds the configured limit.

        Default limit: 50,000 (configurable via portfolio_gex_limit in config.json).
        """
        cfg = config or self.config
        limit = float(cfg.get("portfolio_gex_limit", 50_000.0))
        try:
            greeks = self.get_portfolio_greeks()
            gex = greeks.get("portfolio_gex", 0.0)
            if abs(gex) > limit:
                direction = "long" if gex > 0 else "short"
                return (
                    True,
                    f"Portfolio GEX {gex:+,.0f} exceeds limit ±{limit:,.0f} "
                    f"({direction} gamma exposure)",
                )
        except Exception as exc:
            logger.debug("GEX gate check failed: %s", exc)
        return False, ""
