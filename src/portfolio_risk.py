#!/usr/bin/env python3
"""
Portfolio-level risk aggregation: Greek exposure, GEX, and Monte Carlo VaR.

RiskAggregator reads open paper trades from SQLite, fetches current market data,
computes portfolio Greeks and GEX, and estimates 1-day Value-at-Risk via MJD.
"""

import math
import logging
import sqlite3
from contextlib import closing
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .utils import bs_call, bs_put, bs_delta, bs_gamma, bs_vega, safe_float, is_short_position
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

# Spot prices are shared across every open position: a 60-position book with 29
# unique tickers used to fire 60 sequential yfinance calls, and the whole book
# is priced twice per session (startup update_positions + the in-scan GEX gate).
# A process-wide short-TTL cache collapses that to one fetch per ticker per
# minute — the dominant post-input stall on a single-ticker scan.
_SPOT_CACHE: dict = {}        # {ticker: (spot, timestamp)}
_SPOT_CACHE_TTL = 60          # seconds; spot moves, so keep this short

# The fully priced positions frame (spot + per-leg option-chain IV for every
# open trade) is the single most expensive thing on the interactive scan path,
# and `run_scan` asks for it TWICE per single-ticker scan — the RISK-OFF GEX
# gate and the executive-summary VaR line. Memoizing the whole frame collapses
# those two ~7s passes into one, keyed on the open-book signature so a closed
# position invalidates it.
_POSITIONS_CACHE: dict = {}   # {(db_path, rfr_key, book_signature): (df, timestamp)}
_POSITIONS_CACHE_TTL = 60     # seconds


def reset_spot_cache() -> None:
    """Drop every cached spot AND the priced-positions memo. Tests + forced refresh."""
    _SPOT_CACHE.clear()
    _POSITIONS_CACHE.clear()


def risk_off_filters_picks(config: Optional[Dict] = None) -> bool:
    """Whether RISK-OFF mode should FILTER the scan's picks, or just warn.

    In paper/research mode the picks are data for validating the screener, not
    live orders, so erasing them on portfolio exposure is counter-productive —
    real risk enforcement belongs in the execution/preflight layer. Set
    ``portfolio_gex_filter_picks: false`` in config to make RISK-OFF advisory
    (warn but keep all picks). Defaults to True to preserve enforcement.
    """
    try:
        return bool((config or {}).get("portfolio_gex_filter_picks", True))
    except Exception:
        return True


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
            with closing(sqlite3.connect(self.db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT * FROM trades WHERE status='OPEN'"
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            logger.debug("Could not load open trades: %s", exc)
            return []

    def _fetch_spot_uncached(self, ticker: str) -> Optional[float]:
        """The raw network hop: fast_info, with a 5d-history fallback."""
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

    def _fetch_spot(self, ticker: str) -> Optional[float]:
        """Spot price, deduplicated and cached per ticker for `_SPOT_CACHE_TTL`.

        A failed fetch returns None and is NOT cached, so a transient outage does
        not pin a whole ticker to "no spot" for the TTL.
        """
        cached = _SPOT_CACHE.get(ticker)
        if cached is not None:
            spot, ts = cached
            if _time.monotonic() - ts < _SPOT_CACHE_TTL:
                return spot
        spot = self._fetch_spot_uncached(ticker)
        if spot is not None and spot > 0:
            _SPOT_CACHE[ticker] = (spot, _time.monotonic())
        return spot

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

        Added columns: spot, T_years, sigma, delta, gamma, vega, gex,
        quantity (normalized), is_short.

        Greeks are the BOOK's exposure: direction-signed (short positions
        flip the sign) and scaled by quantity. GEX per position is the book's
        own gamma dollars per 1% move = signed gamma × S² × 0.01 × 100.
        """
        if rfr is None:
            rfr = get_risk_free_rate()

        trades = self._load_open_trades()
        if not trades:
            return pd.DataFrame()

        # Serve a recent identical computation (same book, same rfr) from the memo
        # so the two per-scan callers don't each re-price the whole portfolio.
        signature = tuple(sorted(
            (str(t.get("ticker")), str(t.get("expiration")),
             float(safe_float(t.get("strike")) or 0.0), (t.get("type") or "call").lower(),
             float(safe_float(t.get("quantity")) or 1.0),
             is_short_position(str(t.get("strategy_name") or "")))
            for t in trades))
        cache_key = (self.db_path, round(float(rfr), 4), signature)
        cached = _POSITIONS_CACHE.get(cache_key)
        if cached is not None:
            df_cached, ts = cached
            if _time.monotonic() - ts < _POSITIONS_CACHE_TTL:
                return df_cached.copy()

        # One spot per ticker per call: resolves duplicates even when a ticker
        # fails (None is not written to the cross-call TTL cache, so without this
        # a dead ticker held in N positions would be retried N times).
        call_spots: Dict[str, Optional[float]] = {}

        rows = []
        for t in trades:
            ticker = t.get("ticker", "")
            expiration = t.get("expiration", "")
            strike = safe_float(t.get("strike")) or 0.0
            opt_type = (t.get("type") or "call").lower()

            if not ticker or not expiration or strike <= 0:
                continue

            # Expiry is a free local check — do it BEFORE the network fetch so an
            # already-expired position costs no yfinance round-trip.
            T = self._dte(expiration)
            if T <= 0:
                # Skip already-expired positions — their Greeks are meaningless
                continue

            if ticker not in call_spots:
                call_spots[ticker] = self._fetch_spot(ticker)
            spot = call_spots[ticker]
            if spot is None or spot <= 0:
                continue

            sigma, iv_source = self._get_current_iv(ticker, expiration, strike, opt_type)
            if not sigma or sigma <= 0:
                sigma, iv_source = self._FALLBACK_IV, "fallback"

            # The book's exposure, not the contract's: short strategies flip
            # the sign, and multi-lot / fractional rows scale by quantity.
            qty = safe_float(t.get("quantity")) or 1.0
            if qty <= 0:
                qty = 1.0
            is_short = is_short_position(str(t.get("strategy_name") or ""))
            dir_sign = -1.0 if is_short else 1.0

            try:
                delta = dir_sign * qty * float(bs_delta(opt_type, spot, strike, T, rfr, sigma))
                gamma = dir_sign * qty * float(bs_gamma(spot, strike, T, rfr, sigma))
                vega = dir_sign * qty * float(bs_vega(spot, strike, T, rfr, sigma))
            except Exception:
                delta, gamma, vega = 0.0, 0.0, 0.0

            # Book gamma dollars per 1% move (gamma is already signed × qty)
            gex = gamma * (spot ** 2) * 0.01 * 100.0

            row = dict(t)
            row.update(
                spot=spot,
                T_years=T,
                sigma=sigma,
                iv_source=iv_source,
                quantity=qty,
                is_short=is_short,
                delta=delta,
                gamma=gamma,
                vega=vega,
                gex=gex,
            )
            rows.append(row)

        if not rows:
            # Every position failed to price (e.g. a transient data outage). Do
            # NOT memoize emptiness — the next call should get a fresh attempt.
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        _POSITIONS_CACHE[cache_key] = (df, _time.monotonic())
        return df.copy()

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
        # Cross-position correlation (one-factor model). Independent draws per
        # position would grant a correlated long-call book full diversification
        # credit — VaR understated exactly when it matters. Default 0.7.
        rho = float(self.config.get("var_correlation", 0.7))
        rho = min(max(rho, 0.0), 1.0)
        w_idio = math.sqrt(max(0.0, 1.0 - rho * rho))

        rng = np.random.default_rng(random_seed)
        T_horizon = horizon_days / 365.0
        n_steps = max(horizon_days, 1)
        dt = T_horizon / n_steps

        # Shared market shocks: jumps are market events (one -2% gap hits the
        # whole book), and each position mixes the market factor with its own
        # idiosyncratic draw below.
        z_mkt = rng.standard_normal((n_simulations, n_steps))
        jump_occurs = rng.random((n_simulations, n_steps)) < (jump_intensity * dt)
        jump_sizes = rng.normal(jump_mean, jump_vol, (n_simulations, n_steps))

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

            # One-factor correlated paths: shared market shock + own noise.
            # The idio draw happens even at rho=1 so RNG consumption (and thus
            # every position's market shock) is independent of rho.
            z_idio = rng.standard_normal((n_simulations, n_steps))
            z = rho * z_mkt + w_idio * z_idio
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
                # Direction and quantity come from the priced frame (single
                # source: utils.is_short_position, normalized qty).
                qty = float(safe_float(pos.get("quantity")) or 1.0)
                if bool(pos.get("is_short", False)):
                    pnl = (entry_price - exit_prices) * 100.0 * qty
                else:
                    pnl = (exit_prices - entry_price) * 100.0 * qty
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
