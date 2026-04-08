#!/usr/bin/env python3
"""
Volatility Analytics — vol cone, IV surface, regime classifier, EWMA forecast.
All functions degrade gracefully when data is unavailable.
"""

import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from datetime import datetime, date

try:
    import numpy as np
    HAS_NP = True
except ImportError:
    HAS_NP = False

try:
    import pandas as pd
    HAS_PD = True
except ImportError:
    HAS_PD = False

try:
    import yfinance as yf
    HAS_YF = True
except ImportError:
    HAS_YF = False

try:
    from .utils import bs_delta, safe_float
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False

try:
    from .data_fetching import get_risk_free_rate as _get_rfr
    HAS_RFR = True
except ImportError:
    HAS_RFR = False


def realized_vol(prices, window: int = 30) -> float:
    """Compute annualised realized vol from a price Series using log returns."""
    try:
        log_rets = np.log(prices / prices.shift(1)).dropna()
        rv = float(log_rets.iloc[-window:].std() * math.sqrt(252))
        return rv
    except Exception:
        return float("nan")


def compute_vol_cone(
    ticker: str,
    windows: list = None,
    period: str = "2y",
) -> Optional[dict]:
    """
    Fetch price history via yfinance and compute rolling realized vol for each window.

    Returns dict: window -> {"min", "p25", "median", "p75", "max", "current", "pctile"}
    where pctile = percentile rank of current reading in the full historical distribution.
    Returns None if data unavailable.
    """
    if not HAS_NP or not HAS_PD or not HAS_YF:
        return None
    if windows is None:
        windows = [10, 21, 30, 63, 126, 252]
    try:
        tkr = yf.Ticker(ticker)
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist = tkr.history(period=period)
        if hist.empty or len(hist) < max(windows) + 5:
            return None
        close = hist["Close"].dropna()
        if len(close) < max(windows) + 5:
            return None
        log_rets = np.log(close / close.shift(1)).dropna()
        result = {}
        for w in windows:
            rolling_rv = log_rets.rolling(w).std() * math.sqrt(252)
            rolling_rv = rolling_rv.dropna()
            if len(rolling_rv) < 5:
                continue
            current = float(rolling_rv.iloc[-1])
            arr = rolling_rv.values
            pctile_rank = float(np.mean(arr <= current))
            result[w] = {
                "min": float(np.percentile(arr, 0)),
                "p25": float(np.percentile(arr, 25)),
                "median": float(np.percentile(arr, 50)),
                "p75": float(np.percentile(arr, 75)),
                "max": float(np.percentile(arr, 100)),
                "current": current,
                "pctile": pctile_rank,
            }
        return result if result else None
    except Exception:
        return None


def _fetch_single_expiry(tkr, exp_str: str, spot: float, rfr: float, today) -> Optional[dict]:
    """Fetch and process a single expiry for the IV surface. Returns a row dict or None."""
    import warnings
    try:
        exp_date = datetime.strptime(exp_str[:10], "%Y-%m-%d").date()
        dte = (exp_date - today).days
        if dte < 1:
            return None
        T = dte / 365.0

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            chain = tkr.option_chain(exp_str)
        calls_df = chain.calls
        puts_df = chain.puts
        if calls_df.empty or puts_df.empty:
            return None

        # ATM IV: option with strike closest to spot
        calls_df = calls_df.dropna(subset=["impliedVolatility", "strike"])
        puts_df = puts_df.dropna(subset=["impliedVolatility", "strike"])
        if calls_df.empty or puts_df.empty:
            return None

        atm_call_idx = (calls_df["strike"] - spot).abs().idxmin()
        atm_iv = float(calls_df.loc[atm_call_idx, "impliedVolatility"])

        # 25-delta call and put
        call_25d_iv = None
        put_25d_iv = None

        if HAS_UTILS and len(calls_df) >= 3:
            best_diff = float("inf")
            for _, row in calls_df.iterrows():
                K = float(row["strike"])
                sigma = float(row["impliedVolatility"])
                if sigma <= 0 or K <= 0:
                    continue
                try:
                    d = abs(float(bs_delta("call", spot, K, T, rfr, sigma)))
                    diff = abs(d - 0.25)
                    if diff < best_diff:
                        best_diff = diff
                        call_25d_iv = sigma
                except Exception:
                    continue

        if HAS_UTILS and len(puts_df) >= 3:
            best_diff = float("inf")
            for _, row in puts_df.iterrows():
                K = float(row["strike"])
                sigma = float(row["impliedVolatility"])
                if sigma <= 0 or K <= 0:
                    continue
                try:
                    d = abs(float(bs_delta("put", spot, K, T, rfr, sigma)))
                    diff = abs(d - 0.25)
                    if diff < best_diff:
                        best_diff = diff
                        put_25d_iv = sigma
                except Exception:
                    continue

        skew_25d = (put_25d_iv - call_25d_iv) if (put_25d_iv is not None and call_25d_iv is not None) else None

        return {
            "expiration": exp_str[:10],
            "dte": dte,
            "strike_atm": float(calls_df.loc[atm_call_idx, "strike"]),
            "atm_iv": atm_iv,
            "call_25d_iv": call_25d_iv,
            "put_25d_iv": put_25d_iv,
            "skew_25d": skew_25d,
        }
    except Exception:
        return None


def compute_iv_surface(ticker: str) -> Optional["pd.DataFrame"]:
    """
    Fetch current options chain via yfinance for all available expirations.
    Returns DataFrame with columns: expiration, dte, strike, call_iv, put_iv, skew_25d, atm_iv.
    For each expiration: find 25-delta call and put (abs delta closest to 0.25).
    Returns None if data unavailable.
    """
    if not HAS_YF or not HAS_PD or not HAS_NP or not HAS_UTILS:
        return None
    try:
        tkr = yf.Ticker(ticker)
        expirations = tkr.options
        if not expirations:
            return None

        # Spot price
        try:
            spot = float(tkr.fast_info.last_price or 0)
        except Exception:
            spot = 0.0
        if spot <= 0:
            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    hist = tkr.history(period="2d")
                spot = float(hist["Close"].iloc[-1]) if not hist.empty else 0.0
            except Exception:
                spot = 0.0
        if spot <= 0:
            return None

        today = date.today()
        rfr = _get_rfr() if HAS_RFR else 0.045

        # Fetch expirations concurrently (limit to first 10)
        target_exps = [e for e in expirations[:10]]
        rows = []
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(_fetch_single_expiry, tkr, exp_str, spot, rfr, today): exp_str
                for exp_str in target_exps
            }
            for future in as_completed(futures, timeout=15):
                try:
                    row = future.result(timeout=0)
                    if row is not None:
                        rows.append(row)
                except Exception:
                    pass

        if not rows:
            return None
        # Sort by DTE so term structure is in order
        df = pd.DataFrame(rows)
        return df.sort_values("dte").reset_index(drop=True)
    except Exception:
        return None


def classify_vol_regime(ticker: str, current_iv: Optional[float] = None) -> dict:
    """
    Return a regime classification dict with keys:
        regime, iv_pctile_30d, hv_iv_ratio, term_structure, skew_direction, recommendation.
    """
    result = {
        "regime": "UNKNOWN",
        "iv_pctile_30d": None,
        "hv_iv_ratio": None,
        "term_structure": "UNKNOWN",
        "skew_direction": "UNKNOWN",
        "recommendation": "Insufficient data for regime classification",
    }
    try:
        cone = compute_vol_cone(ticker, windows=[30])
        if cone and 30 in cone:
            d30 = cone[30]
            iv_pctile = d30["pctile"]
            result["iv_pctile_30d"] = iv_pctile
            hv_30 = d30["current"]  # current 30d realized vol

            # Try to get current IV from options chain if not provided
            if current_iv is None and HAS_YF:
                try:
                    tkr = yf.Ticker(ticker)
                    exps = tkr.options
                    if exps:
                        import warnings
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            ch = tkr.option_chain(exps[0])
                        calls = ch.calls.dropna(subset=["impliedVolatility"])
                        if not calls.empty:
                            try:
                                sp = float(tkr.fast_info.last_price or 0)
                            except Exception:
                                sp = 0.0
                            if sp > 0:
                                atm_idx = (calls["strike"] - sp).abs().idxmin()
                                current_iv = float(calls.loc[atm_idx, "impliedVolatility"])
                except Exception:
                    pass

            if current_iv and current_iv > 0 and hv_30 > 0:
                result["hv_iv_ratio"] = hv_30 / current_iv
            elif current_iv and current_iv > 0:
                result["hv_iv_ratio"] = None

            # Regime from IV percentile
            if iv_pctile >= 0.90:
                result["regime"] = "EXTREME"
            elif iv_pctile >= 0.70:
                result["regime"] = "HIGH_IV"
            elif iv_pctile >= 0.30:
                result["regime"] = "NORMAL"
            else:
                result["regime"] = "LOW_IV"

        # Term structure from IV surface
        surface = compute_iv_surface(ticker)
        if surface is not None and len(surface) >= 2:
            first_iv = surface.iloc[0]["atm_iv"]
            last_iv = surface.iloc[-1]["atm_iv"]
            if last_iv > first_iv + 0.005:
                result["term_structure"] = "CONTANGO"
            elif last_iv < first_iv - 0.005:
                result["term_structure"] = "BACKWARDATION"
            else:
                result["term_structure"] = "FLAT"

            # Skew from first available expiration
            skew_25d = surface.iloc[0].get("skew_25d")
            if skew_25d is not None and not (isinstance(skew_25d, float) and math.isnan(skew_25d)):
                sv = float(skew_25d)
                if sv > 0.02:
                    result["skew_direction"] = "PUT_SKEW"
                elif sv < -0.02:
                    result["skew_direction"] = "CALL_SKEW"
                else:
                    result["skew_direction"] = "FLAT"

        # Build recommendation
        regime = result["regime"]
        ts = result["term_structure"]
        hv_iv = result["hv_iv_ratio"]
        skew = result["skew_direction"]

        if regime in ("HIGH_IV", "EXTREME"):
            if hv_iv is not None and hv_iv < 0.85:
                rec = f"SELL premium \u2014 IV elevated vs realized (HV/IV={hv_iv:.2f}x), {ts.lower()} term structure"
            else:
                rec = f"IV elevated \u2014 consider defined-risk premium selling, {ts.lower()} structure"
        elif regime == "LOW_IV":
            rec = f"BUY premium or vol \u2014 IV historically cheap, {ts.lower()} structure"
        else:
            if skew == "PUT_SKEW":
                rec = "NEUTRAL \u2014 consider put spreads or straddles; put skew elevated"
            else:
                rec = f"NEUTRAL \u2014 standard screening approach, {ts.lower()} structure"

        result["recommendation"] = rec

    except Exception:
        pass
    return result


__all__ = [
    "realized_vol",
    "compute_vol_cone",
    "compute_iv_surface",
    "classify_vol_regime",
]
