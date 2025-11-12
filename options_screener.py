#!/usr/bin/env python3
"""
Options Screener (Top 5 low / 5 medium / 5 high by premium)

Features:
- Fetches options chains via yfinance (Yahoo Finance data; check terms).
- Scores contracts by liquidity (volume/OI), spread tightness, delta quality, and IV balance.
- Categorizes by premium into low/medium/high and picks top 5 in each.
- User-friendly prompts, input validation, and formatted console output.

Note:
- Not financial advice. For personal/informational use only.
- Data availability and timeliness depend on the data provider.
"""

import sys
import math
import os
import csv
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed

# Dependency checks
missing = []
try:
    import pandas as pd
except Exception:
    missing.append("pandas")
try:
    import yfinance as yf
except Exception:
    missing.append("yfinance")
if missing:
    print(f"Missing dependencies: {', '.join(missing)}")
    print("Install with: pip install " + " ".join(missing))
    sys.exit(1)


def norm_cdf(x: float) -> float:
    # Standard normal CDF using erf to avoid external deps
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_delta(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """
    Black-Scholes delta. Returns:
      call: N(d1)
      put:  N(d1) - 1
    """
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        if option_type.lower() == "call":
            return norm_cdf(d1)
        else:
            return norm_cdf(d1) - 1.0
    except Exception:
        return None


def bs_price(option_type: str, S: float, K: float, T: float, r: float, sigma: float) -> Optional[float]:
    """Black-Scholes theoretical option price (using N(d1), N(d2))."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        if option_type.lower() == "call":
            return S * math.exp(r * T) * norm_cdf(d1) - K * norm_cdf(d2)
        else:
            return K * norm_cdf(-d2) - S * math.exp(r * T) * norm_cdf(-d1)
    except Exception:
        return None


def _d1d2(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[Optional[float], Optional[float]]:
    """Compute d1, d2 for Black-Scholes; returns (d1, d2) or (None, None)."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None, None
        d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        return d1, d2
    except Exception:
        return None, None


def calculate_probability_of_profit(option_type: str, S: float, K: float, T: float, sigma: float, premium: float) -> Optional[float]:
    """Calculate probability of profit at expiration."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0 or premium <= 0:
            return None
        
        # Break-even point
        if option_type.lower() == "call":
            breakeven = K + premium
        else:  # put
            breakeven = K - premium
        
        # Probability that stock will be beyond break-even
        d = (math.log(S / breakeven) - (0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        
        if option_type.lower() == "call":
            return norm_cdf(d)
        else:
            return 1.0 - norm_cdf(d)
    except Exception:
        return None


def calculate_expected_move(S: float, sigma: float, T: float) -> Optional[float]:
    """Calculate expected move (1 standard deviation) until expiration."""
    try:
        if S <= 0 or sigma <= 0 or T <= 0:
            return None
        return S * sigma * math.sqrt(T)
    except Exception:
        return None


def calculate_probability_of_touch(option_type: str, S: float, K: float, T: float, sigma: float) -> Optional[float]:
    """Calculate probability that option will touch the strike price before expiration."""
    try:
        if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
            return None
        
        # Probability of touching is approximately 2 * delta for ATM options
        # More precise: P(touch) ≈ 2 * N(d2)
        d2 = (math.log(S / K) - (0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
        
        if option_type.lower() == "call" and K > S:
            return 2 * norm_cdf(d2)
        elif option_type.lower() == "put" and K < S:
            return 2 * (1.0 - norm_cdf(d2))
        else:
            # ITM already touched
            return 1.0
    except Exception:
        return None


def calculate_risk_reward(option_type: str, premium: float, S: float, K: float) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Calculate max loss, break-even, and potential reward ratio."""
    try:
        if premium <= 0 or S <= 0 or K <= 0:
            return None, None, None
        
        max_loss = premium * 100  # Per contract
        
        if option_type.lower() == "call":
            breakeven = K + premium
            # Potential reward: assume move to 2x expected (simplified)
            potential_reward = max(0, (S * 1.5 - K) * 100 - max_loss)
        else:  # put
            breakeven = K - premium
            potential_reward = max(0, (K - S * 0.5) * 100 - max_loss)
        
        risk_reward_ratio = potential_reward / max_loss if max_loss > 0 else 0
        
        return max_loss, breakeven, risk_reward_ratio
    except Exception:
        return None, None, None


def get_historical_volatility(ticker: yf.Ticker, period: int = 30) -> Optional[float]:
    """Fetch historical volatility (annualized) from recent price data."""
    try:
        hist = ticker.history(period=f"{period+10}d", interval="1d")
        if hist.empty or len(hist) < period:
            return None
        
        # Calculate log returns
        returns = (hist['Close'] / hist['Close'].shift(1)).apply(math.log).dropna()
        
        if len(returns) < 2:
            return None
        
        # Annualized standard deviation
        daily_vol = returns.std()
        annual_vol = daily_vol * math.sqrt(252)  # Trading days
        
        return annual_vol
    except Exception:
        return None


def safe_float(x, default=None):
    try:
        if x is None:
            return default
        v = float(x)
        if math.isfinite(v):
            return v
        return default
    except Exception:
        return default


def get_underlying_price(ticker: yf.Ticker) -> Optional[float]:
    # Try fast_info, then info, then last close
    try:
        fi = getattr(ticker, "fast_info", None)
        if fi:
            lp = safe_float(getattr(fi, "last_price", None))
            if lp:
                return lp
            # Sometimes as dict
            lp = safe_float(getattr(fi, "last_price", None) or fi.get("last_price") if isinstance(fi, dict) else None)
            if lp:
                return lp
    except Exception:
        pass
    try:
        info = ticker.info or {}
        lp = safe_float(info.get("regularMarketPrice"))
        if lp:
            return lp
    except Exception:
        pass
    try:
        hist = ticker.history(period="5d", interval="1d")
        if not hist.empty:
            return safe_float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return None


def get_risk_free_rate() -> float:
    """
    Fetch the current risk-free rate from yfinance using 13-week Treasury bill (^IRX).
    Returns annualized rate as decimal (e.g., 0.045 for 4.5%).
    Falls back to 4.5% if unable to fetch.
    """
    default_rate = 0.045
    try:
        # ^IRX is the 13-week Treasury bill yield (quoted as annual %)
        tbill = yf.Ticker("^IRX")
        # Try fast_info first
        try:
            fi = getattr(tbill, "fast_info", None)
            if fi:
                rate = safe_float(getattr(fi, "last_price", None))
                if rate and rate > 0:
                    return rate / 100.0  # Convert from percentage to decimal
        except Exception:
            pass
        
        # Try info dict
        try:
            info = tbill.info or {}
            rate = safe_float(info.get("regularMarketPrice"))
            if rate and rate > 0:
                return rate / 100.0
        except Exception:
            pass
        
        # Try recent history
        hist = tbill.history(period="5d", interval="1d")
        if not hist.empty:
            rate = safe_float(hist["Close"].iloc[-1])
            if rate and rate > 0:
                return rate / 100.0
    except Exception:
        pass
    
    # Fallback to default
    return default_rate


def fetch_options_yfinance(symbol: str, max_expiries: int) -> Tuple[pd.DataFrame, Optional[float]]:
    """Fetch options data and historical volatility for a symbol."""
    tkr = yf.Ticker(symbol)
    underlying = get_underlying_price(tkr)
    if underlying is None:
        raise ValueError("Could not determine underlying price for ticker.")
    
    # Fetch historical volatility
    hv = get_historical_volatility(tkr, period=30)
    
    try:
        expirations = tkr.options
    except Exception as e:
        raise RuntimeError(f"Failed to fetch options expirations: {e}")
    if not expirations:
        raise RuntimeError("No options expirations available.")

    expirations = expirations[:max_expiries]
    frames = []
    for exp in expirations:
        try:
            oc = tkr.option_chain(exp)
        except Exception as e:
            # Skip this expiration if it fails
            continue
        for opt_type, df in [("call", oc.calls), ("put", oc.puts)]:
            if df is None or df.empty:
                continue
            sub = df.copy()
            sub["type"] = opt_type
            sub["expiration"] = exp
            sub["symbol"] = symbol.upper()

            # Normalize column names we rely on
            # yfinance has: 'strike','lastPrice','bid','ask','volume','openInterest','impliedVolatility','lastTradeDate','inTheMoney','contractSymbol'
            for col in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility"]:
                if col not in sub.columns:
                    sub[col] = pd.NA

            frames.append(sub)

    if not frames:
        raise RuntimeError("No options data frames fetched from yfinance.")
    df = pd.concat(frames, ignore_index=True)
    df["underlying"] = underlying
    df["hv_30d"] = hv  # Add historical volatility column
    return df, hv


def enrich_and_score(df: pd.DataFrame, min_dte: int, max_dte: int, risk_free_rate: float) -> pd.DataFrame:
    # Prepare and filter
    now = datetime.now(timezone.utc)

    # expiration to dt
    df["exp_dt"] = pd.to_datetime(df["expiration"], errors="coerce", utc=True)
    df = df[df["exp_dt"].notna()].copy()
    df["T_years"] = (df["exp_dt"] - now).dt.total_seconds() / (365.0 * 24 * 3600)
    # filter by DTE bounds
    df = df[(df["T_years"] > min_dte / 365.0) & (df["T_years"] < max_dte / 365.0)].copy()

    # Numerics
    for c in ["strike", "lastPrice", "bid", "ask", "volume", "openInterest", "impliedVolatility", "underlying"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Premium as mid if possible, else last
    df["bid"] = df["bid"].fillna(0.0)
    df["ask"] = df["ask"].fillna(0.0)
    df["mid"] = (df["bid"] + df["ask"]) / 2.0
    df["premium"] = df["mid"].where(df["mid"] > 0.0, df["lastPrice"])

    # Drop where we have no usable premium
    df = df[(df["premium"].notna()) & (df["premium"] > 0)].copy()

    # Spread pct (relative to mid)
    df["spread_pct"] = (df["ask"] - df["bid"]) / df["mid"]
    df.loc[~df["spread_pct"].replace([pd.NA, pd.NaT], pd.NA).apply(lambda x: pd.notna(x) and math.isfinite(x)), "spread_pct"] = float("inf")

    # Liquidity filters: remove totally dead contracts
    df["volume"] = df["volume"].fillna(0).astype(float)
    df["openInterest"] = df["openInterest"].fillna(0).astype(float)
    df = df[(df["volume"] > 0) | (df["openInterest"] > 0)].copy()

    if df.empty:
        return df

    # Fill missing IV with chain median per expiration + type to avoid skew
    df["impliedVolatility"] = df["impliedVolatility"].astype(float)
    df["iv_group_median"] = df.groupby(["exp_dt", "type"])["impliedVolatility"].transform(lambda s: s.median(skipna=True))
    df["impliedVolatility"] = df["impliedVolatility"].fillna(df["iv_group_median"])
    overall_iv_median = df["impliedVolatility"].median(skipna=True)
    df["impliedVolatility"] = df["impliedVolatility"].fillna(overall_iv_median)

    # Compute delta
    def _row_delta(row):
        d = bs_delta(
            option_type=row["type"],
            S=safe_float(row["underlying"], 0.0) or 0.0,
            K=safe_float(row["strike"], 0.0) or 0.0,
            T=safe_float(row["T_years"], 0.0) or 0.0,
            r=risk_free_rate,
            sigma=max(1e-9, safe_float(row["impliedVolatility"], 0.0) or 0.0),
        )
        return float("nan") if d is None else d

    df["delta"] = df.apply(_row_delta, axis=1)
    df["abs_delta"] = df["delta"].abs()
    
    # === NEW ADVANCED METRICS ===
    
    # Probability of Profit
    def _calc_pop(row):
        return calculate_probability_of_profit(
            row["type"],
            safe_float(row["underlying"]),
            safe_float(row["strike"]),
            safe_float(row["T_years"]),
            safe_float(row["impliedVolatility"]),
            safe_float(row["premium"])
        )
    df["prob_profit"] = df.apply(_calc_pop, axis=1)
    
    # Expected Move
    def _calc_exp_move(row):
        return calculate_expected_move(
            safe_float(row["underlying"]),
            safe_float(row["impliedVolatility"]),
            safe_float(row["T_years"])
        )
    df["expected_move"] = df.apply(_calc_exp_move, axis=1)
    
    # Probability of Touch
    def _calc_pot(row):
        return calculate_probability_of_touch(
            row["type"],
            safe_float(row["underlying"]),
            safe_float(row["strike"]),
            safe_float(row["T_years"]),
            safe_float(row["impliedVolatility"])
        )
    df["prob_touch"] = df.apply(_calc_pot, axis=1)
    
    # Risk/Reward Analysis
    def _calc_rr(row):
        max_loss, breakeven, rr_ratio = calculate_risk_reward(
            row["type"],
            safe_float(row["premium"]),
            safe_float(row["underlying"]),
            safe_float(row["strike"])
        )
        return pd.Series({
            'max_loss': max_loss,
            'breakeven': breakeven,
            'rr_ratio': rr_ratio
        })
    
    rr_data = df.apply(_calc_rr, axis=1)
    df["max_loss"] = rr_data["max_loss"]
    df["breakeven"] = rr_data["breakeven"]
    df["rr_ratio"] = rr_data["rr_ratio"]
    
    # IV vs HV comparison (IV advantage)
    if "hv_30d" in df.columns and df["hv_30d"].notna().any():
        df["iv_vs_hv"] = df["impliedVolatility"] - df["hv_30d"]
        df["iv_hv_ratio"] = df["impliedVolatility"] / df["hv_30d"].replace(0, float('nan'))
    else:
        df["iv_vs_hv"] = 0.0
        df["iv_hv_ratio"] = 1.0
    
    # IV Skew (calls vs puts at same strike/expiry)
    df["iv_skew"] = 0.0  # Default
    for (exp, strike), group in df.groupby(["expiration", "strike"]):
        if len(group) == 2:  # Has both call and put
            call_iv = group[group["type"] == "call"]["impliedVolatility"].values
            put_iv = group[group["type"] == "put"]["impliedVolatility"].values
            if len(call_iv) > 0 and len(put_iv) > 0:
                skew = put_iv[0] - call_iv[0]
                df.loc[group.index, "iv_skew"] = skew
    
    # Liquidity Quality Flags
    df["liquidity_flag"] = "GOOD"
    df.loc[(df["volume"] < 10) & (df["openInterest"] < 100), "liquidity_flag"] = "POOR"
    df.loc[(df["volume"] >= 10) & (df["volume"] < 50) & (df["openInterest"] >= 100) & (df["openInterest"] < 500), "liquidity_flag"] = "FAIR"
    
    # Wide Spread Flag
    df["spread_flag"] = "OK"
    df.loc[df["spread_pct"] > 0.10, "spread_flag"] = "WIDE"
    df.loc[df["spread_pct"] > 0.20, "spread_flag"] = "VERY_WIDE"
    
    # === END NEW METRICS ===

    # Expected value (EV) and probability ITM using Black-Scholes
    def _calc_ev(row):
        S = safe_float(row["underlying"], 0.0) or 0.0
        K = safe_float(row["strike"], 0.0) or 0.0
        T = safe_float(row["T_years"], 0.0) or 0.0
        sigma = max(1e-9, safe_float(row["impliedVolatility"], 0.0) or 0.0)
        opt_type = row["type"].lower()
        d1, d2 = _d1d2(S, K, T, risk_free_rate, sigma)
        if d1 is None:
            return pd.Series({"p_itm": None, "theo_value": None, "ev_per_contract": None})
        # Probability of expiring ITM under risk-neutral measure
        p_itm = norm_cdf(d2) if opt_type == "call" else norm_cdf(-d2)
        # Expected payoff at expiration (risk-neutral)
        if opt_type == "call":
            expected_payoff = S * math.exp(risk_free_rate * T) * norm_cdf(d1) - K * norm_cdf(d2)
        else:
            expected_payoff = K * norm_cdf(-d2) - S * math.exp(risk_free_rate * T) * norm_cdf(-d1)
        theo_value = expected_payoff  # theoretical RN value
        premium = safe_float(row["premium"], 0.0) or 0.0
        # Approximate round-trip half-spread cost (enter+exit ~ 1 spread)
        spread_pct = row.get("spread_pct", 0.0)
        spread_cost = 100.0 * premium * float(spread_pct if pd.notna(spread_pct) else 0.0)
        ev = 100.0 * (expected_payoff - premium) - spread_cost
        return pd.Series({"p_itm": p_itm, "theo_value": theo_value, "ev_per_contract": ev})

    ev_data = df.apply(_calc_ev, axis=1)
    df["p_itm"] = ev_data["p_itm"]
    df["theo_value"] = ev_data["theo_value"]
    df["ev_per_contract"] = ev_data["ev_per_contract"]

    # Normalize features using ranks to reduce outlier impact
    def rank_norm(s: pd.Series) -> pd.Series:
        n = len(s)
        if n <= 1:
            return pd.Series([0.5] * n, index=s.index)
        r = s.rank(method="average", na_option="keep")
        return (r - 1.0) / (n - 1.0)

    vol_n = rank_norm(df["volume"].fillna(0))
    oi_n = rank_norm(df["openInterest"].fillna(0))

    # Spread score: 1 for very tight spreads, 0 for very wide
    # Cap spread at 25% of mid; beyond that is treated equally poor
    sp = df["spread_pct"].replace([pd.NA, pd.NaT], float("inf"))
    sp = sp.clip(lower=0, upper=0.25)
    spread_score = 1.0 - (sp / 0.25)

    # Delta quality: target around 0.4 absolute delta
    delta_target = 0.40
    delta_quality = 1.0 - (df["abs_delta"] - delta_target).abs() / max(delta_target, 1e-6)
    delta_quality = delta_quality.clip(lower=0.0, upper=1.0)

    # IV quality: prefer moderate IV vs chain (avoid extremes)
    iv_n = rank_norm(df["impliedVolatility"].fillna(df["impliedVolatility"].median()))
    iv_quality = 1.0 - (2.0 * (iv_n - 0.5).abs())  # 1 at mid, 0 at edges

    # Liquidity (volume+OI)
    liquidity = 0.5 * (vol_n + oi_n)
    
    # IV Advantage Score (prefer IV > HV but not extreme)
    iv_advantage = df["iv_vs_hv"].fillna(0).clip(lower=-0.2, upper=0.2)
    iv_advantage_score = (iv_advantage + 0.2) / 0.4  # Normalize to 0-1
    
    # Probability of Profit Score
    pop_score = df["prob_profit"].fillna(0.5).clip(lower=0, upper=1)
    
    # Risk/Reward Score (higher is better, cap at 3:1)
    rr_score = df["rr_ratio"].fillna(0).clip(lower=0, upper=3) / 3.0

    # EV score (rank-normalized expected value per contract)
    ev_score = rank_norm(df["ev_per_contract"].fillna(df["ev_per_contract"].median()))

    # Enhanced Composite Score (prioritize profit accuracy via EV and PoP)
    df["quality_score"] = (
        0.30 * ev_score +
        0.20 * liquidity +
        0.15 * iv_advantage_score +
        0.15 * pop_score +
        0.10 * spread_score +
        0.10 * delta_quality
    )
    df["ev_score"] = ev_score

    # Keep helpful computed columns
    df["spread_pct"] = df["spread_pct"].replace([float("inf"), -float("inf")], pd.NA)
    df["liquidity_score"] = liquidity
    df["delta_quality"] = delta_quality
    df["iv_quality"] = iv_quality
    df["spread_score"] = spread_score

    # Basic sanity ordering hints
    df = df.sort_values(["quality_score", "volume", "openInterest"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def categorize_by_premium(df: pd.DataFrame, budget: Optional[float] = None) -> pd.DataFrame:
    """Categorize by premium using quantiles (single-stock) or budget-based (multi-stock)."""
    if df.empty:
        return df
    
    # Calculate contract cost
    df["contract_cost"] = df["premium"] * 100
    
    if budget is not None:
        # Budget mode: categorize based on % of budget
        # LOW: 0-33% of budget, MEDIUM: 33-66%, HIGH: 66-100%
        def cat_budget(cost):
            pct = cost / budget
            if pct <= 0.33:
                return "LOW"
            elif pct <= 0.66:
                return "MEDIUM"
            else:
                return "HIGH"
        df["price_bucket"] = df["contract_cost"].apply(cat_budget)
    else:
        # Single-stock mode: use quantiles
        premiums = df["premium"].astype(float)
        q1 = premiums.quantile(1/3)
        q2 = premiums.quantile(2/3)

        def cat(p):
            if p <= q1:
                return "LOW"
            elif p <= q2:
                return "MEDIUM"
            else:
                return "HIGH"

        df["price_bucket"] = premiums.apply(cat)
    
    return df


def pick_top_per_bucket(df: pd.DataFrame, per_bucket: int = 5, diversify_tickers: bool = False) -> pd.DataFrame:
    """Pick top options per bucket, optionally diversifying across tickers."""
    picks = []
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        sub = df[df["price_bucket"] == bucket].copy()
        if sub.empty:
            continue
        
        if diversify_tickers and "symbol" in sub.columns:
            # Try to get diverse tickers in budget mode
            # Sort by quality first (now includes EV)
            sub = sub.sort_values(
                by=["quality_score", "spread_pct", "volume", "openInterest", "T_years"],
                ascending=[False, True, False, False, True],
            )
            
            # Pick best from each ticker, then fill remaining slots
            selected = []
            tickers_used = set()
            
            # First pass: best option from each unique ticker
            for _, row in sub.iterrows():
                if row["symbol"] not in tickers_used:
                    selected.append(row)
                    tickers_used.add(row["symbol"])
                    if len(selected) >= per_bucket:
                        break
            
            # Second pass: fill remaining slots with next best regardless of ticker
            if len(selected) < per_bucket:
                for _, row in sub.iterrows():
                    if len(selected) >= per_bucket:
                        break
                    # Check if this exact row is already selected
                    is_duplicate = any(
                        (s["symbol"] == row["symbol"] and 
                         s["strike"] == row["strike"] and 
                         s["expiration"] == row["expiration"] and
                         s["type"] == row["type"]) 
                        for s in selected
                    )
                    if not is_duplicate:
                        selected.append(row)
            
            picks.append(pd.DataFrame(selected))
        else:
            # Standard sorting for single-stock mode
            sub = sub.sort_values(
                by=["quality_score", "spread_pct", "volume", "openInterest", "T_years"],
                ascending=[False, True, False, False, True],
            )
            picks.append(sub.head(per_bucket))
    
    if not picks:
        return pd.DataFrame()
    out = pd.concat(picks, ignore_index=True)
    return out


def format_pct(x: Optional[float]) -> str:
    try:
        if x is None or (isinstance(x, float) and not math.isfinite(x)) or pd.isna(x):
            return "-"
        return f"{100.0 * float(x):.1f}%"
    except Exception:
        return "-"


def format_money(x: Optional[float]) -> str:
    try:
        return f"${float(x):.2f}"
    except Exception:
        return "-"


def determine_moneyness(row: pd.Series) -> str:
    """Determine if option is ITM or OTM based on strike vs underlying."""
    try:
        strike = float(row["strike"])
        underlying = float(row["underlying"])
        opt_type = row["type"].lower()
        
        if opt_type == "call":
            return "ITM" if strike < underlying else "OTM"
        else:  # put
            return "ITM" if strike > underlying else "OTM"
    except Exception:
        return "---"


def rationale_row(row: pd.Series, chain_iv_median: float) -> str:
    parts: List[str] = []
    # Liquidity
    parts.append(f"liquidity vol {int(row['volume'])}, OI {int(row['openInterest'])}")
    # Spread
    sp = row.get("spread_pct", pd.NA)
    if pd.notna(sp) and math.isfinite(sp):
        parts.append(f"spread {format_pct(sp)}")
    # Delta
    d = row.get("delta", pd.NA)
    if pd.notna(d) and math.isfinite(d):
        parts.append(f"delta {d:+.2f}")
    # IV vs chain
    iv = row.get("impliedVolatility", pd.NA)
    if pd.notna(iv) and math.isfinite(iv):
        rel = "≈" if abs(float(iv) - chain_iv_median) <= 0.02 else ("above" if iv > chain_iv_median else "below")
        parts.append(f"IV {format_pct(iv)} ({rel} chain median {format_pct(chain_iv_median)})")
    # Overall
    parts.append(f"quality {row['quality_score']:.2f}")
    return "; ".join(parts)


def print_report(df_picks: pd.DataFrame, underlying_price: float, rfr: float, num_expiries: int, min_dte: int, max_dte: int, mode: str = "Single-stock", budget: Optional[float] = None):
    """Enhanced report with context, formatting, top pick, and summary."""
    if df_picks.empty:
        print("No picks available after filtering.")
        return
    
    chain_iv_median = df_picks["impliedVolatility"].median(skipna=True)
    
    # Header with context
    print("\n" + "="*80)
    if mode == "Budget scan":
        print(f"  OPTIONS SCREENER REPORT - MULTI-TICKER (Budget: ${budget:.2f})")
    elif mode == "Discovery scan":
        unique_tickers = df_picks["symbol"].nunique()
        print(f"  OPTIONS SCREENER REPORT - DISCOVERY MODE ({unique_tickers} Tickers)")
    else:
        print(f"  OPTIONS SCREENER REPORT - {df_picks.iloc[0]['symbol']}")
    print("="*80)
    
    if mode == "Single-stock":
        print(f"  Stock Price: ${underlying_price:.2f}")
    elif mode == "Budget scan":
        print(f"  Budget Constraint: ${budget:.2f} per contract (premium × 100)")
        print(f"  Categories: LOW ($0-${budget*0.33:.2f}) | MEDIUM (${budget*0.33:.2f}-${budget*0.66:.2f}) | HIGH (${budget*0.66:.2f}-${budget:.2f})")
    elif mode == "Discovery scan":
        print(f"  Scan Type: Top opportunities across all price ranges (no budget limit)")
        print(f"  Categories: LOW (bottom 33%) | MEDIUM (middle 33%) | HIGH (top 33%) by premium")
    print(f"  Risk-Free Rate: {rfr*100:.2f}% (13-week Treasury)")
    print(f"  Expirations Scanned: {num_expiries}")
    print(f"  DTE Range: {min_dte} - {max_dte} days")
    print(f"  Chain Median IV: {format_pct(chain_iv_median)}")
    print(f"  Mode: {mode}")
    print("="*80)

    def header(txt: str):
        print("\n" + "─" * 80)
        print(f"  {txt}")
        print("─" * 80)

    # Print each bucket with summary stats
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        sub = df_picks[df_picks["price_bucket"] == bucket]
        if sub.empty:
            continue
        
        # Bucket header
        header(f"{bucket} PREMIUM (Top {len(sub)} Picks)")
        
        # Category summary stats
        avg_iv = sub["impliedVolatility"].mean()
        avg_spread = sub["spread_pct"].mean(skipna=True)
        median_delta = sub["abs_delta"].median()
        print(f"  Summary: Avg IV {format_pct(avg_iv)} | Avg Spread {format_pct(avg_spread)} | Median |Δ| {median_delta:.2f}\n")
        
        # Column headers (add Ticker for multi-stock modes)
        if mode in ["Budget scan", "Discovery scan"]:
            print(f"  {'Tkr':<5} {'Type':<5} {'Strike':<8} {'Exp':<12} {'Prem':<8} {'IV':<7} {'OI':<8} {'Vol':<7} {'Δ':<7} {'Tag':<4}")
            print("  " + "-"*78)
        else:
            print(f"  {'Type':<5} {'Strike':<8} {'Exp':<12} {'Prem':<8} {'IV':<7} {'OI':<8} {'Vol':<8} {'Δ':<7} {'Tag':<4}")
            print("  " + "-"*76)
        
        for _, r in sub.iterrows():
            exp = pd.to_datetime(r["expiration"]).date()
            moneyness = determine_moneyness(r)
            dte = int(r["T_years"] * 365)
            
            # Main line with aligned columns
            if mode in ["Budget scan", "Discovery scan"]:
                print(
                    f"  {r['symbol']:<5} "
                    f"{r['type'].upper():<5} "
                    f"{r['strike']:>7.2f} "
                    f"{exp} "
                    f"{format_money(r['premium']):<8} "
                    f"{format_pct(r['impliedVolatility']):<7} "
                    f"{int(r['openInterest']):>6} "
                    f"{int(r['volume']):>6} "
                    f"{r['delta']:>+6.2f} "
                    f"{moneyness:<4}"
                )
            else:
                print(
                    f"  {r['type'].upper():<5} "
                    f"{r['strike']:>7.2f} "
                    f"{exp} "
                    f"{format_money(r['premium']):<8} "
                    f"{format_pct(r['impliedVolatility']):<7} "
                    f"{int(r['openInterest']):>7} "
                    f"{int(r['volume']):>7} "
                    f"{r['delta']:>+6.2f} "
                    f"{moneyness:<4}"
                )
            # Rationale
            cost_per_contract = r['premium'] * 100
            if mode in ["Budget scan", "Discovery scan"]:
                ticker_info = f"${r['underlying']:.2f}"
                print(f"    → {rationale_row(r, chain_iv_median)} | DTE: {dte}d | Stock: {ticker_info} | Cost: ${cost_per_contract:.2f}\n")
            else:
                print(f"    → {rationale_row(r, chain_iv_median)} | DTE: {dte}d\n")


def export_to_csv(df_picks: pd.DataFrame, mode: str, budget: Optional[float] = None) -> str:
    """Export picks to CSV with timestamp."""
    try:
        # Create exports directory if it doesn't exist
        os.makedirs("exports", exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"exports/options_picks_{mode.replace(' ', '_')}_{timestamp}.csv"
        
        # Select relevant columns for export
        export_cols = [
            "symbol", "type", "strike", "expiration", "premium", "underlying",
            "delta", "impliedVolatility", "hv_30d", "iv_vs_hv",
            "volume", "openInterest", "spread_pct",
            "prob_profit", "expected_move", "prob_touch", "p_itm",
            "max_loss", "breakeven", "rr_ratio",
            "theo_value", "ev_per_contract", "ev_score",
            "quality_score", "liquidity_flag", "spread_flag", "price_bucket"
        ]
        
        # Filter to existing columns
        export_cols = [c for c in export_cols if c in df_picks.columns]
        
        df_picks[export_cols].to_csv(filename, index=False)
        return filename
    except Exception as e:
        print(f"Warning: Could not export CSV: {e}")
        return None


def log_trade_entry(df_picks: pd.DataFrame, mode: str) -> None:
    """Log trade entries for future P/L tracking."""
    try:
        # Create trades_log directory if it doesn't exist
        os.makedirs("trades_log", exist_ok=True)
        
        log_file = "trades_log/entries.csv"
        file_exists = os.path.exists(log_file)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        with open(log_file, 'a', newline='') as f:
            fieldnames = [
                'timestamp', 'mode', 'symbol', 'type', 'strike', 'expiration',
                'entry_premium', 'entry_underlying', 'delta', 'iv', 'hv',
                'prob_profit', 'p_itm', 'rr_ratio', 'theo_value', 'ev_per_contract',
                'quality_score', 'status'
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            
            if not file_exists:
                writer.writeheader()
            
            for _, row in df_picks.iterrows():
                writer.writerow({
                    'timestamp': timestamp,
                    'mode': mode,
                    'symbol': row.get('symbol', ''),
                    'type': row.get('type', ''),
                    'strike': row.get('strike', ''),
                    'expiration': row.get('expiration', ''),
                    'entry_premium': row.get('premium', ''),
                    'entry_underlying': row.get('underlying', ''),
                    'delta': row.get('delta', ''),
                    'iv': row.get('impliedVolatility', ''),
                    'hv': row.get('hv_30d', ''),
                    'prob_profit': row.get('prob_profit', ''),
                    'p_itm': row.get('p_itm', ''),
                    'rr_ratio': row.get('rr_ratio', ''),
                    'theo_value': row.get('theo_value', ''),
                    'ev_per_contract': row.get('ev_per_contract', ''),
                    'quality_score': row.get('quality_score', ''),
                    'status': 'OPEN'
                })
        
        print(f"\n  💾 Trade entries logged to {log_file}")
    except Exception as e:
        print(f"Warning: Could not log trades: {e}")


def setup_logging() -> logging.Logger:
    """Configure a simple console logger and JSONL file logger.
    LOG_LEVEL env var controls verbosity (default INFO).
    """
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), format="%(message)s")
    logger = logging.getLogger("options_screener")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("logs", exist_ok=True)
    logger.json_path = os.path.join("logs", f"run_{ts}.jsonl")  # type: ignore[attr-defined]
    return logger


def log_picks_json(logger: logging.Logger, picks_df: pd.DataFrame, context: Dict):
    """Append picks to a JSONL log for later evaluation/backtesting."""
    try:
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "context": context,
            "picks": picks_df.to_dict(orient="records"),
        }
        with open(logger.json_path, "a") as f:  # type: ignore[attr-defined]
            f.write(json.dumps(payload) + "\n")
    except Exception:
        pass


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    sfx = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{sfx}: ").strip()
    return default if (not val and default is not None) else val


def main():
    print("Options Screener (yfinance)")
    print("Note: For personal/informational use only. Review data provider terms.")
    print("\nModes:")
    print("  1. Enter a ticker (e.g., AAPL) for single-stock analysis")
    print("  2. Enter 'ALL' for budget-based multi-stock scan")
    print("  3. Enter 'DISCOVER' to scan top 100 most-traded tickers (no budget limit)\n")
    
    symbol_input = prompt_input("Enter stock ticker, 'ALL', or 'DISCOVER'", "").upper()
    
    # Determine mode
    is_budget_mode = (symbol_input == "ALL")
    is_discovery_mode = (symbol_input == "DISCOVER" or symbol_input == "")
    
    if is_discovery_mode:
        mode = "Discovery scan"
    elif is_budget_mode:
        mode = "Budget scan"
    else:
        mode = "Single-stock"
    
    budget = None
    tickers = []
    
    if is_discovery_mode:
        # Discovery mode: scan top 100 most-traded options tickers
        print("\n=== DISCOVERY MODE ===")
        print("Scanning top 100 most-traded options tickers for best opportunities...")
        
        # Top 100 most liquid options tickers (ordered by typical volume)
        tickers = [
            # Major Indices & ETFs
            "SPY", "QQQ", "IWM", "DIA", "VOO", "VTI", "EEM", "GLD", "SLV", "TLT",
            "XLF", "XLE", "XLK", "XLV", "XLI", "XLP", "XLY", "XLU", "XLB", "XLRE",
            # Mega Cap Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "NFLX", "AMD", "INTC",
            "CRM", "ORCL", "ADBE", "CSCO", "AVGO", "QCOM", "TXN", "AMAT", "MU", "LRCX",
            # Financial
            "JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "V",
            "MA", "PYPL", "SQ", "COIN",
            # Healthcare & Pharma
            "JNJ", "UNH", "PFE", "ABBV", "MRK", "TMO", "LLY", "ABT", "DHR", "BMY",
            "AMGN", "GILD", "CVS", "MRNA", "BNTX",
            # Consumer & Retail
            "WMT", "HD", "DIS", "NKE", "MCD", "SBUX", "TGT", "COST", "LOW", "TJX",
            # Energy
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO",
            # Industrial & Manufacturing
            "BA", "CAT", "GE", "MMM", "HON", "UPS", "LMT", "RTX", "DE",
            # Communication & Media
            "T", "VZ", "CMCSA", "TMUS", "CHTR",
            # Automotive & Transportation
            "F", "GM", "RIVN", "LCID", "NIO", "UBER", "LYFT", "DAL", "UAL", "AAL"
        ]
        
        # Limit to 50 for reasonable scan time
        max_scan = int(prompt_input("How many tickers to scan (1-100)", "50"))
        max_scan = max(1, min(100, max_scan))
        tickers = tickers[:max_scan]
        
        print(f"Will scan {len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}")
        
    elif is_budget_mode:
        # Budget mode setup
        try:
            budget = float(prompt_input("Enter your budget per contract in USD (e.g., 500)", "500"))
            if budget <= 0:
                print("Budget must be positive.")
                sys.exit(1)
        except Exception:
            print("Invalid budget amount.")
            sys.exit(1)
        
        # Default liquid tickers
        default_tickers = "AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,GOOGL,META"
        tickers_input = prompt_input("Enter comma-separated tickers to scan", default_tickers)
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        
        if not tickers:
            print("No valid tickers provided.")
            sys.exit(1)
        
        print(f"\nBudget Mode: Scanning {len(tickers)} tickers with ${budget:.2f} budget...")
    else:
        # Single-stock mode
        if not symbol_input.isalnum():
            print("Please enter a valid alphanumeric ticker (e.g., AAPL).")
            sys.exit(1)
        tickers = [symbol_input]
        print(f"\nSingle-Stock Mode: Analyzing {symbol_input}...")

    logger = setup_logging()
    try:
        max_expiries = int(prompt_input("How many nearest expirations to scan", "4"))
        if max_expiries <= 0 or max_expiries > 12:
            print("Please choose between 1 and 12 expirations.")
            sys.exit(1)
    except Exception:
        print("Invalid number for expirations.")
        sys.exit(1)

    try:
        min_dte = int(prompt_input("Minimum days to expiration (DTE)", "7"))
        max_dte = int(prompt_input("Maximum days to expiration (DTE)", "120"))
        if min_dte < 0 or max_dte <= min_dte:
            print("DTE bounds invalid. Ensure 0 <= min < max.")
            sys.exit(1)
    except Exception:
        print("Invalid DTE inputs.")
        sys.exit(1)

    # Fetch risk-free rate automatically
    print("Fetching current risk-free rate...")
    rfr = get_risk_free_rate()
    print(f"Using risk-free rate: {rfr*100:.2f}% (13-week Treasury)")

    try:
        # Collect data from all tickers (parallel for speed)
        all_frames = []
        errors = []
        def _fetch(tkr: str):
            try:
                df_raw, hv = fetch_options_yfinance(tkr, max_expiries=max_expiries)
                return {"ticker": tkr, "df": df_raw, "hv": hv, "error": None}
            except Exception as e:
                return {"ticker": tkr, "df": None, "hv": None, "error": str(e)}

        max_workers = min(8, len(tickers)) if len(tickers) > 1 else 1
        if max_workers > 1:
            with ThreadPoolExecutor(max_workers=max_workers) as ex:
                futures = {ex.submit(_fetch, t): t for t in tickers}
                for fut in as_completed(futures):
                    res = fut.result()
                    t = res["ticker"]
                    if res["df"] is not None and not res["df"].empty:
                        all_frames.append(res["df"])
                        hv = res["hv"]
                        hv_str = f" HV:{hv*100:.1f}%" if hv else ""
                        print(f"  Fetched {t} ✓{hv_str}")
                    else:
                        print(f"  Fetched {t} (no data)")
                    if res["error"]:
                        errors.append({"ticker": t, "error": res["error"]})
        else:
            res = _fetch(tickers[0])
            if res["df"] is not None and not res["df"].empty:
                all_frames.append(res["df"])
                hv = res["hv"]
                hv_str = f" HV:{hv*100:.1f}%" if hv else ""
                print(f"  Fetched {tickers[0]} ✓{hv_str}")
            elif res["error"]:
                errors.append({"ticker": tickers[0], "error": res["error"]})

        if not all_frames:
            print("\nNo options data retrieved from any ticker.")
            sys.exit(0)

        # Combine all data
        df_combined = pd.concat(all_frames, ignore_index=True)
        print(f"\nProcessing {len(df_combined)} total options contracts...")
        
        # Score and filter
        df_scored = enrich_and_score(df_combined, min_dte=min_dte, max_dte=max_dte, risk_free_rate=rfr)
        if df_scored.empty:
            print("No contracts passed filters (check DTE bounds or liquidity).")
            sys.exit(0)
        
        # Apply budget filter if in budget mode
        if is_budget_mode:
            df_scored["contract_cost"] = df_scored["premium"] * 100
            df_scored = df_scored[df_scored["contract_cost"] <= budget].copy()
            if df_scored.empty:
                print(f"No contracts found within budget of ${budget:.2f}.")
                sys.exit(0)
            print(f"Found {len(df_scored)} contracts within budget.")
            
            # Show budget distribution for user clarity
            print(f"\nBudget Categories:")
            print(f"  LOW:    $0 - ${budget*0.33:.2f} (0-33% of budget)")
            print(f"  MEDIUM: ${budget*0.33:.2f} - ${budget*0.66:.2f} (33-66% of budget)")
            print(f"  HIGH:   ${budget*0.66:.2f} - ${budget:.2f} (66-100% of budget)")
        elif is_discovery_mode:
            # Discovery mode: no budget filter, use quantiles for categorization
            print(f"\nDiscovery Mode: Analyzing {len(df_scored)} quality options across all price ranges...")
        
        # Categorize and pick
        # Use budget categorization for budget mode, quantile for single-stock and discovery
        df_bucketed = categorize_by_premium(df_scored, budget=budget if is_budget_mode else None)
        
        # Diversify tickers in budget and discovery modes
        picks = pick_top_per_bucket(df_bucketed, per_bucket=5, diversify_tickers=(is_budget_mode or is_discovery_mode))
        if picks.empty:
            print("Could not produce picks in the requested buckets.")
            sys.exit(0)
        
        # Get underlying price for report (first ticker in single mode, or 0 in budget mode)
        underlying_price = df_scored.iloc[0]["underlying"] if not df_scored.empty and not is_budget_mode else 0.0
        
        # Print main report
        print_report(picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode, budget)
        
        # Structured JSONL logging of the picks and context
        log_picks_json(
            logger,
            picks,
            context={
                "mode": mode,
                "budget": budget,
                "max_expiries": max_expiries,
                "min_dte": min_dte,
                "max_dte": max_dte,
                "risk_free_rate": rfr,
            },
        )
        # Compute and display top overall pick
        print("\n" + "="*80)
        print("  ⭐ TOP OVERALL PICK")
        print("="*80)
        
        # Enhanced scoring for top pick: balance quality with practical factors
        picks_copy = picks.copy()
        chain_iv_median = picks["impliedVolatility"].median(skipna=True)
        
        # Weighted overall score
        # Favor: high quality, good liquidity, moderate IV, tight spread, balanced delta
        picks_copy["overall_score"] = (
            0.40 * picks_copy["quality_score"] +
            0.20 * picks_copy["liquidity_score"] +
            0.15 * picks_copy["spread_score"] +
            0.15 * picks_copy["delta_quality"] +
            0.10 * picks_copy["iv_quality"]
        )
        
        top_pick = picks_copy.sort_values("overall_score", ascending=False).iloc[0]
        exp = pd.to_datetime(top_pick["expiration"]).date()
        moneyness = determine_moneyness(top_pick)
        dte = int(top_pick["T_years"] * 365)
        
        print(
            f"\n  {top_pick['symbol']} {top_pick['type'].upper()} | "
            f"Strike ${top_pick['strike']:.2f} | Exp {exp} ({dte}d) | {moneyness}\n"
        )
        if mode in ["Budget scan", "Discovery scan"]:
            print(f"  Stock Price: ${top_pick['underlying']:.2f}")
        print(f"  Premium: {format_money(top_pick['premium'])}")
        if mode == "Budget scan":
            contract_cost = top_pick['premium'] * 100
            print(f"  Contract Cost: ${contract_cost:.2f} (within ${budget:.2f} budget)")
        elif mode == "Discovery scan":
            contract_cost = top_pick['premium'] * 100
            print(f"  Contract Cost: ${contract_cost:.2f}")
        print(f"  IV: {format_pct(top_pick['impliedVolatility'])} | Delta: {top_pick['delta']:+.2f} | Quality: {top_pick['quality_score']:.2f}")
        print(f"  Volume: {int(top_pick['volume'])} | OI: {int(top_pick['openInterest'])} | Spread: {format_pct(top_pick['spread_pct'])}")
        
        # Generate justification
        justification_parts = []
        
        # Liquidity assessment
        if top_pick["volume"] > picks["volume"].quantile(0.75):
            justification_parts.append("excellent liquidity")
        elif top_pick["volume"] > picks["volume"].median():
            justification_parts.append("good liquidity")
        
        # IV assessment
        iv_diff = abs(top_pick["impliedVolatility"] - chain_iv_median)
        if iv_diff <= 0.05:
            justification_parts.append("balanced IV near chain median")
        elif top_pick["impliedVolatility"] < chain_iv_median:
            justification_parts.append("favorable IV below median")
        
        # Spread assessment
        if pd.notna(top_pick["spread_pct"]) and top_pick["spread_pct"] < 0.05:
            justification_parts.append("tight bid-ask spread")
        
        # Delta assessment
        if 0.35 <= abs(top_pick["delta"]) <= 0.50:
            justification_parts.append("optimal delta range")
        
        # DTE assessment
        if dte <= 30:
            justification_parts.append("short-term play")
        elif dte <= 60:
            justification_parts.append("medium-term opportunity")
        else:
            justification_parts.append("longer-dated position")
        
        justification = "Chosen for " + ", ".join(justification_parts[:3]) + "."
        if len(justification_parts) > 3:
            justification += f" Also offers {', '.join(justification_parts[3:])}." 
        
        print(f"\n  💡 Rationale: {justification}")
        
        # Summary footer
        print("\n" + "="*80)
        print("  SCAN SUMMARY")
        print("="*80)
        print(f"  Total Picks Displayed: {len(picks)}")
        if mode in ["Budget scan", "Discovery scan"]:
            unique_tickers = picks["symbol"].nunique()
            print(f"  Tickers Covered: {unique_tickers}")
        if mode == "Budget scan":
            print(f"  Budget Constraint: ${budget:.2f} per contract")
        print(f"  Chain Median IV: {format_pct(chain_iv_median)}")
        print(f"  Expirations Scanned: {max_expiries}")
        print(f"  Risk-Free Rate Used: {rfr*100:.2f}%")
        print(f"  DTE Filter: {min_dte}-{max_dte} days")
        print(f"  Mode: {mode}")
        print("="*80)
        print("\n  ⚠️  Not financial advice. Verify all data before trading.")
        print("="*80 + "\n")
        
        # === EXPORT AND LOGGING ===
        export_choice = prompt_input("Export results to CSV? (y/n)", "n").lower()
        if export_choice == "y":
            csv_file = export_to_csv(picks, mode, budget)
            if csv_file:
                print(f"\n  📄 Results exported to: {csv_file}")
        
        log_choice = prompt_input("Log trades for P/L tracking? (y/n)", "n").lower()
        if log_choice == "y":
            log_trade_entry(picks, mode)
        
        print("\n👋 Done! Happy trading!\n")
        
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
