#!/usr/bin/env python3
"""
Volatility Analytics — vol cone, IV surface, regime classifier, EWMA forecast.
All functions degrade gracefully when data is unavailable.
"""

import math
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
    from . import formatting as fmt
    from .formatting import Colors, BoxChars, supports_color, colorize
    HAS_FMT = True
except ImportError:
    HAS_FMT = False
    fmt = None

try:
    from .utils import bs_delta, safe_float
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False


def _c(text: str, color: str = "", bold: bool = False) -> str:
    """Color helper that degrades gracefully."""
    if HAS_FMT and fmt and color:
        return fmt.colorize(str(text), color, bold=bold)
    return str(text)


def _sep(width: int = 90) -> str:
    line = "  " + "\u2500" * (width - 2)
    if HAS_FMT and fmt:
        return fmt.colorize(line, fmt.Colors.DIM)
    return line


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


def _ordinal_suffix(n: int) -> str:
    """Return ordinal suffix for an integer: 1->st, 2->nd, 3->rd, else->th."""
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")


def print_vol_cone(ticker: str, current_iv: Optional[float] = None, width: int = 90) -> None:
    """Print a formatted vol cone table to terminal."""
    cone = compute_vol_cone(ticker)
    if cone is None:
        print(f"\n  Vol cone unavailable for {ticker}")
        return

    print()
    header = f"  VOL CONE  \u2014  {ticker}"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(header)
    print(_sep(width))

    col_hdr = f"  {'Window':<8}  {'Min':>7}  {'25th':>7}  {'Median':>7}  {'75th':>7}  {'Max':>7}  {'Current':>8}  {'Pctile':>7}"
    if HAS_FMT and fmt:
        print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
    else:
        print(col_hdr)

    window_labels = {10: "10d", 21: "21d", 30: "30d", 63: "63d", 126: "126d", 252: "252d"}
    for w in sorted(cone.keys()):
        d = cone[w]
        label = window_labels.get(w, f"{w}d")
        pctile_int = int(round(d["pctile"] * 100))
        suf = _ordinal_suffix(pctile_int)
        pctile_str = f"{pctile_int}{suf}"

        # Color current by percentile
        cur_val = d["current"]
        cur_str = f"{cur_val*100:.1f}%"
        if HAS_FMT and fmt:
            if d["pctile"] >= 0.75:
                cur_colored = fmt.colorize(cur_str, fmt.Colors.RED, bold=True)
            elif d["pctile"] >= 0.50:
                cur_colored = fmt.colorize(cur_str, fmt.Colors.YELLOW)
            else:
                cur_colored = fmt.colorize(cur_str, fmt.Colors.GREEN)
        else:
            cur_colored = cur_str

        row = (
            f"  {label:<8}"
            f"  {d['min']*100:>6.1f}%"
            f"  {d['p25']*100:>6.1f}%"
            f"  {d['median']*100:>6.1f}%"
            f"  {d['p75']*100:>6.1f}%"
            f"  {d['max']*100:>6.1f}%"
            f"  {cur_colored:>8}"
            f"  {pctile_str:>7}"
        )
        print(row)

    print(_sep(width))

    # IV vs 30d median HV line
    if current_iv is not None and 30 in cone:
        hv_30_median = cone[30]["median"]
        if hv_30_median > 0:
            ratio = current_iv / hv_30_median
            iv_pct_str = f"{current_iv*100:.1f}%"
            if ratio > 1.20:
                verdict = "RICH"
                iv_color = fmt.Colors.RED if HAS_FMT and fmt else ""
            elif ratio < 0.85:
                verdict = "CHEAP"
                iv_color = fmt.Colors.GREEN if HAS_FMT and fmt else ""
            else:
                verdict = "FAIR"
                iv_color = fmt.Colors.YELLOW if HAS_FMT and fmt else ""
            iv_line = f"  Current IV: {iv_pct_str}  \u2192  {verdict} vs 30d median HV ({ratio:.2f}x)"
            if HAS_FMT and fmt and iv_color:
                print(fmt.colorize(iv_line, iv_color))
            else:
                print(iv_line)
    print()


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
        rows = []
        rfr = 0.045

        import warnings
        for exp_str in expirations[:10]:  # limit to first 10 expirations
            try:
                exp_date = datetime.strptime(exp_str[:10], "%Y-%m-%d").date()
                dte = (exp_date - today).days
                if dte < 1:
                    continue
                T = dte / 365.0

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    chain = tkr.option_chain(exp_str)
                calls_df = chain.calls
                puts_df = chain.puts
                if calls_df.empty or puts_df.empty:
                    continue

                # ATM IV: option with strike closest to spot
                calls_df = calls_df.dropna(subset=["impliedVolatility", "strike"])
                puts_df = puts_df.dropna(subset=["impliedVolatility", "strike"])

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

                rows.append({
                    "expiration": exp_str[:10],
                    "dte": dte,
                    "strike_atm": float(calls_df.loc[atm_call_idx, "strike"]),
                    "atm_iv": atm_iv,
                    "call_25d_iv": call_25d_iv,
                    "put_25d_iv": put_25d_iv,
                    "skew_25d": skew_25d,
                })
            except Exception:
                continue

        if not rows:
            return None
        return pd.DataFrame(rows)
    except Exception:
        return None


def print_iv_surface(ticker: str, spot: Optional[float] = None, width: int = 90) -> None:
    """Print a compact IV surface table."""
    surface = compute_iv_surface(ticker)
    if surface is None or surface.empty:
        print(f"\n  IV surface unavailable for {ticker}")
        return

    spot_str = f"  (spot ${spot:.2f})" if spot else ""
    print()
    header = f"  IV SURFACE  \u2014  {ticker}{spot_str}"
    if HAS_FMT and fmt:
        print(fmt.colorize(header, fmt.Colors.BRIGHT_CYAN, bold=True))
    else:
        print(header)
    print(_sep(width))

    _skew_col = "25\u0394 Skew"
    col_hdr = f"  {'Expiry':<12}  {'DTE':>5}  {'ATM IV':>7}  {_skew_col:>9}  {'Structure'}"
    if HAS_FMT and fmt:
        print(fmt.colorize(col_hdr, fmt.Colors.BOLD))
    else:
        print(col_hdr)

    prev_atm_iv = None
    for _, row in surface.iterrows():
        atm_iv = row["atm_iv"]
        dte = int(row["dte"])
        exp_str = str(row["expiration"])[:10]

        # Term structure label
        if prev_atm_iv is None:
            structure = "FRONT"
            struct_color = fmt.Colors.DIM if HAS_FMT and fmt else ""
        else:
            diff = atm_iv - prev_atm_iv
            if diff > 0.005:
                structure = "CONTANGO"
                struct_color = fmt.Colors.GREEN if HAS_FMT and fmt else ""
            elif diff < -0.005:
                structure = "BACKWARDATION"
                struct_color = fmt.Colors.RED if HAS_FMT and fmt else ""
            else:
                structure = "FLAT"
                struct_color = fmt.Colors.YELLOW if HAS_FMT and fmt else ""

        atm_iv_str = f"{atm_iv*100:.1f}%"

        skew_25d = row.get("skew_25d")
        if skew_25d is not None and not (isinstance(skew_25d, float) and math.isnan(skew_25d)):
            skew_val = float(skew_25d)
            skew_str = f"{skew_val*100:+.1f}%"
            if HAS_FMT and fmt:
                if skew_val > 0.05:
                    skew_colored = fmt.colorize(skew_str, fmt.Colors.RED)
                elif skew_val > 0.02:
                    skew_colored = fmt.colorize(skew_str, fmt.Colors.YELLOW)
                else:
                    skew_colored = fmt.colorize(skew_str, fmt.Colors.GREEN)
            else:
                skew_colored = skew_str
        else:
            skew_colored = "   n/a"

        if HAS_FMT and fmt and struct_color:
            struct_colored = fmt.colorize(structure, struct_color)
        else:
            struct_colored = structure

        print(f"  {exp_str:<12}  {dte:>5}  {atm_iv_str:>7}  {skew_colored:>9}  {struct_colored}")
        prev_atm_iv = atm_iv

    print()


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
                rec = f"NEUTRAL \u2014 consider put spreads or straddles; put skew elevated"
            else:
                rec = f"NEUTRAL \u2014 standard screening approach, {ts.lower()} structure"

        result["recommendation"] = rec

    except Exception:
        pass
    return result


def print_regime_summary(ticker: str, current_iv: Optional[float] = None, width: int = 90) -> None:
    """
    One-line regime summary suitable for embedding in scan output.
    Format:
      Vol Regime [TICKER]: HIGH_IV (82nd pctile) | HV/IV: 0.76x RICH | CONTANGO | PUT_SKEW -> Sell premium
    """
    try:
        reg = classify_vol_regime(ticker, current_iv=current_iv)
        regime = reg.get("regime", "UNKNOWN")
        pctile = reg.get("iv_pctile_30d")
        hv_iv = reg.get("hv_iv_ratio")
        ts = reg.get("term_structure", "")
        skew = reg.get("skew_direction", "")
        rec = reg.get("recommendation", "")

        pctile_str = ""
        if pctile is not None:
            p_int = int(round(pctile * 100))
            suf = _ordinal_suffix(p_int)
            pctile_str = f" ({p_int}{suf} pctile)"

        hv_iv_str = ""
        if hv_iv is not None:
            label = "RICH" if hv_iv < 0.85 else ("FAIR" if hv_iv < 1.15 else "CHEAP")
            hv_iv_str = f" | HV/IV: {hv_iv:.2f}x {label}"

        ts_str = f" | {ts}" if ts and ts != "UNKNOWN" else ""
        skew_str = f" | {skew}" if skew and skew != "UNKNOWN" else ""
        rec_short = rec[:60] if rec else ""
        rec_str = f" \u2192 {rec_short}" if rec_short else ""

        line = f"  Vol Regime [{ticker}]: {regime}{pctile_str}{hv_iv_str}{ts_str}{skew_str}{rec_str}"

        if HAS_FMT and fmt:
            if regime in ("HIGH_IV", "EXTREME"):
                color = fmt.Colors.RED
            elif regime == "LOW_IV":
                color = fmt.Colors.GREEN
            else:
                color = fmt.Colors.YELLOW
            print(fmt.colorize(line, color))
        else:
            print(line)
    except Exception:
        pass


__all__ = [
    "realized_vol",
    "compute_vol_cone",
    "print_vol_cone",
    "compute_iv_surface",
    "print_iv_surface",
    "classify_vol_regime",
    "print_regime_summary",
]
