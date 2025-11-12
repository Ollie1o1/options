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
from datetime import datetime, timezone
from typing import Optional, Tuple, List

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


def fetch_options_yfinance(symbol: str, max_expiries: int) -> pd.DataFrame:
    tkr = yf.Ticker(symbol)
    underlying = get_underlying_price(tkr)
    if underlying is None:
        raise ValueError("Could not determine underlying price for ticker.")
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
    return df


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

    # Composite score
    df["quality_score"] = (
        0.35 * liquidity +
        0.25 * spread_score +
        0.20 * delta_quality +
        0.20 * iv_quality
    )

    # Keep helpful computed columns
    df["spread_pct"] = df["spread_pct"].replace([float("inf"), -float("inf")], pd.NA)
    df["liquidity_score"] = liquidity
    df["delta_quality"] = delta_quality
    df["iv_quality"] = iv_quality
    df["spread_score"] = spread_score

    # Basic sanity ordering hints
    df = df.sort_values(["quality_score", "volume", "openInterest"], ascending=[False, False, False]).reset_index(drop=True)
    return df


def categorize_by_premium(df: pd.DataFrame) -> pd.DataFrame:
    # Create low/medium/high bins by premium quantiles
    if df.empty:
        return df
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


def pick_top_per_bucket(df: pd.DataFrame, per_bucket: int = 5) -> pd.DataFrame:
    picks = []
    for bucket in ["LOW", "MEDIUM", "HIGH"]:
        sub = df[df["price_bucket"] == bucket].copy()
        if sub.empty:
            continue
        # Tie-breakers: quality, tighter spreads, more volume, nearer expirations
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
    else:
        print(f"  OPTIONS SCREENER REPORT - {df_picks.iloc[0]['symbol']}")
    print("="*80)
    if mode != "Budget scan":
        print(f"  Stock Price: ${underlying_price:.2f}")
    else:
        print(f"  Budget Constraint: ${budget:.2f} per contract (premium × 100)")
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
        
        # Column headers (add Ticker for multi-stock mode)
        if mode == "Budget scan":
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
            if mode == "Budget scan":
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
            ticker_info = f"${r['underlying']:.2f}" if mode == "Budget scan" else ""
            cost_per_contract = r['premium'] * 100
            if mode == "Budget scan":
                print(f"    → {rationale_row(r, chain_iv_median)} | DTE: {dte}d | Stock: {ticker_info} | Cost: ${cost_per_contract:.2f}\n")
            else:
                print(f"    → {rationale_row(r, chain_iv_median)} | DTE: {dte}d\n")


def prompt_input(prompt: str, default: Optional[str] = None) -> str:
    sfx = f" [{default}]" if default is not None else ""
    val = input(f"{prompt}{sfx}: ").strip()
    return default if (not val and default is not None) else val


def main():
    print("Options Screener (yfinance)")
    print("Note: For personal/informational use only. Review data provider terms.")
    print("\nModes:")
    print("  - Enter a ticker (e.g., AAPL) for single-stock analysis")
    print("  - Enter 'ALL' or leave blank for budget-based multi-stock scan\n")
    
    symbol_input = prompt_input("Enter stock ticker or 'ALL' for budget mode", "").upper()
    
    # Determine mode
    is_budget_mode = (symbol_input == "ALL" or symbol_input == "")
    mode = "Budget scan" if is_budget_mode else "Single-stock"
    budget = None
    tickers = []
    
    if is_budget_mode:
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
        # Collect data from all tickers
        all_frames = []
        for ticker in tickers:
            try:
                print(f"  Fetching {ticker}...", end="")
                df_raw = fetch_options_yfinance(ticker, max_expiries=max_expiries)
                if not df_raw.empty:
                    all_frames.append(df_raw)
                    print(" ✓")
                else:
                    print(" (no data)")
            except Exception as e:
                print(f" (error: {str(e)[:30]})")
                continue
        
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
        
        # Categorize and pick
        df_bucketed = categorize_by_premium(df_scored)
        picks = pick_top_per_bucket(df_bucketed, per_bucket=5)
        if picks.empty:
            print("Could not produce picks in the requested buckets.")
            sys.exit(0)
        
        # Get underlying price for report (first ticker in single mode, or 0 in budget mode)
        underlying_price = df_scored.iloc[0]["underlying"] if not df_scored.empty and not is_budget_mode else 0.0
        
        # Print main report
        print_report(picks, underlying_price, rfr, max_expiries, min_dte, max_dte, mode, budget)
        
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
        if mode == "Budget scan":
            print(f"  Stock Price: ${top_pick['underlying']:.2f}")
        print(f"  Premium: {format_money(top_pick['premium'])}")
        if mode == "Budget scan":
            contract_cost = top_pick['premium'] * 100
            print(f"  Contract Cost: ${contract_cost:.2f} (within ${budget:.2f} budget)")
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
        if mode == "Budget scan":
            unique_tickers = picks["symbol"].nunique()
            print(f"  Tickers Covered: {unique_tickers}")
            print(f"  Budget Constraint: ${budget:.2f} per contract")
        print(f"  Chain Median IV: {format_pct(chain_iv_median)}")
        print(f"  Expirations Scanned: {max_expiries}")
        print(f"  Risk-Free Rate Used: {rfr*100:.2f}%")
        print(f"  DTE Filter: {min_dte}-{max_dte} days")
        print(f"  Mode: {mode}")
        print("="*80)
        print("\n  ⚠️  Not financial advice. Verify all data before trading.")
        print("="*80 + "\n")
    except KeyboardInterrupt:
        print("\nCancelled.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
