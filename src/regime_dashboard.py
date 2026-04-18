#!/usr/bin/env python3
"""
Market Regime Dashboard — compact one-box summary shown at startup.
Fetches VIX, VIX3M, SPY 5d return, SPY 30d realized vol, and SPY options PCR.
All fetches degrade gracefully when data is unavailable.
"""

import math
import warnings
from typing import Optional, Dict, Any

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


def _c(text: str, color: str = "", bold: bool = False) -> str:
    if HAS_FMT and fmt and color:
        return fmt.colorize(str(text), color, bold=bold)
    return str(text)


_cached_session = None

def _get_session():
    """Return a timeout-aware session for yfinance calls (cached)."""
    global _cached_session
    if _cached_session is not None:
        return _cached_session
    try:
        from curl_cffi import requests as _cffi
        s = _cffi.Session(impersonate="chrome")
        s.timeout = 6
        _cached_session = s
        return s
    except ImportError:
        import requests as _req
        s = _req.Session()
        s.request = lambda *a, timeout=6, **kw: _req.Session.request(s, *a, timeout=timeout, **kw)
        _cached_session = s
        return s


def _safe_last_price(ticker_sym: str) -> Optional[float]:
    """Fetch last price for a ticker, returning None on failure."""
    if not HAS_YF:
        return None
    try:
        tkr = yf.Ticker(ticker_sym, session=_get_session())
        p = getattr(tkr.fast_info, "last_price", None)
        if p and float(p) > 0:
            return float(p)
    except Exception:
        pass
    return None


def _safe_hist(ticker_sym: str, period: str = "2mo") -> Optional["pd.Series"]:
    """Fetch Close price history Series, returning None on failure."""
    if not HAS_YF or not HAS_PD:
        return None
    try:
        tkr = yf.Ticker(ticker_sym, session=_get_session())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hist = tkr.history(period=period)
        if not hist.empty and "Close" in hist.columns:
            return hist["Close"].dropna()
    except Exception:
        pass
    return None


def fetch_market_regime() -> Dict[str, Any]:
    """
    Fetch market regime indicators and return a structured dict.

    Returns:
        {
            "vix": float | None,
            "vix_3m": float | None,
            "vix_term_structure": "CONTANGO" | "BACKWARDATION" | "FLAT" | "UNKNOWN",
            "spy_ret_5d": float | None,
            "spy_hv_30": float | None,
            "options_pcr": float | None,
            "iv_premium": float | None,
            "posture": "RISK_ON" | "NEUTRAL" | "RISK_OFF" | "DEFENSIVE",
            "posture_rationale": str,
        }
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    result: Dict[str, Any] = {
        "vix": None,
        "vix_3m": None,
        "vix_term_structure": "UNKNOWN",
        "spy_ret_5d": None,
        "spy_hv_30": None,
        "options_pcr": None,
        "iv_premium": None,
        "posture": "NEUTRAL",
        "posture_rationale": "Insufficient data for regime classification",
    }

    def _fetch_pcr() -> Optional[float]:
        if not HAS_YF or not HAS_PD:
            return None
        try:
            spy_tkr = yf.Ticker("SPY", session=_get_session())
            exps = spy_tkr.options
            if exps:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    ch = spy_tkr.option_chain(exps[0])
                calls_oi = float(ch.calls["openInterest"].sum()) if not ch.calls.empty else 0.0
                puts_oi = float(ch.puts["openInterest"].sum()) if not ch.puts.empty else 0.0
                if calls_oi > 0:
                    return round(puts_oi / calls_oi, 3)
        except Exception:
            pass
        return None

    try:
        with ThreadPoolExecutor(max_workers=4) as pool:
            f_vix = pool.submit(_safe_last_price, "^VIX")
            f_vix3m = pool.submit(_safe_last_price, "^VIX3M")
            f_spy = pool.submit(_safe_hist, "SPY", "3mo")
            f_pcr = pool.submit(_fetch_pcr)

            vix = f_vix.result(timeout=6)
            vix3m = f_vix3m.result(timeout=6)
            spy_close = f_spy.result(timeout=6)
            result["options_pcr"] = f_pcr.result(timeout=6)

        result["vix"] = vix
        result["vix_3m"] = vix3m

        # VIX term structure
        if vix is not None and vix3m is not None:
            if vix3m > vix + 0.5:
                result["vix_term_structure"] = "CONTANGO"
            elif vix > vix3m + 0.5:
                result["vix_term_structure"] = "BACKWARDATION"
            else:
                result["vix_term_structure"] = "FLAT"

        # SPY history for 5d return and 30d HV
        if spy_close is not None and len(spy_close) >= 6:
            spy_ret_5d = float(spy_close.iloc[-1] / spy_close.iloc[-6] - 1.0)
            result["spy_ret_5d"] = spy_ret_5d

            if HAS_NP and len(spy_close) >= 32:
                import numpy as _np
                log_rets = _np.log(spy_close / spy_close.shift(1)).dropna()
                hv_30 = float(log_rets.iloc[-30:].std() * math.sqrt(252))
                result["spy_hv_30"] = hv_30

        # IV premium: VIX / SPY_HV_30 - 1
        if result["vix"] is not None and result["spy_hv_30"] is not None and result["spy_hv_30"] > 0:
            vix_frac = result["vix"] / 100.0
            hv30 = result["spy_hv_30"]
            result["iv_premium"] = round(vix_frac / hv30 - 1.0, 4)

        # Posture
        vix_val = result["vix"]
        vts = result["vix_term_structure"]
        spy_ret = result["spy_ret_5d"]
        posture = "NEUTRAL"
        rationale_parts = []

        if vix_val is not None and vts is not None and spy_ret is not None:
            if vix_val < 15 and vts == "CONTANGO" and spy_ret > 0:
                posture = "RISK_ON"
                rationale_parts.append("VIX low (<15)")
                rationale_parts.append("term structure contango (normal)")
                rationale_parts.append("SPY positive momentum")
            elif vix_val > 25 or vts == "BACKWARDATION":
                posture = "RISK_OFF"
                if vix_val > 25:
                    rationale_parts.append(f"VIX elevated ({vix_val:.1f})")
                if vts == "BACKWARDATION":
                    rationale_parts.append("VIX backwardation (stress signal)")
            elif vix_val > 20 or (spy_ret is not None and spy_ret < -0.03):
                posture = "DEFENSIVE"
                if vix_val > 20:
                    rationale_parts.append(f"VIX above 20 ({vix_val:.1f})")
                if spy_ret is not None and spy_ret < -0.03:
                    rationale_parts.append("SPY falling >3% over 5d")
            else:
                posture = "NEUTRAL"
                rationale_parts.append("VIX in normal range" if vts == "CONTANGO" else "mixed signals")
                if spy_ret is not None:
                    rationale_parts.append(f"SPY 5d {spy_ret*100:+.1f}%")
        elif vix_val is not None:
            if vix_val > 25:
                posture = "RISK_OFF"
                rationale_parts.append(f"VIX elevated ({vix_val:.1f})")
            elif vix_val > 20:
                posture = "DEFENSIVE"
                rationale_parts.append(f"VIX moderately elevated ({vix_val:.1f})")
            else:
                posture = "NEUTRAL"
                rationale_parts.append(f"VIX moderate ({vix_val:.1f})")

        # Add IV premium comment
        iv_prem = result["iv_premium"]
        if iv_prem is not None:
            if iv_prem > 0.20:
                rationale_parts.append("IV elevated vs realized")
            elif iv_prem < -0.10:
                rationale_parts.append("IV below realized (options cheap)")

        result["posture"] = posture
        result["posture_rationale"] = ", ".join(rationale_parts) if rationale_parts else "No clear signal"

    except Exception:
        pass

    return result


def print_regime_dashboard(width: int = 90) -> None:
    """
    Print a compact one-box market regime dashboard.

    Format:
      ┌─────────────── MARKET REGIME ──────────────────┐
      │  VIX: 18.2  │  VIX3M: 19.8  │  Term: CONTANGO  │  PCR(SPY): 0.82   │
      │  SPY 5d: -1.2%  │  SPY HV30: 14.2%  │  IV Premium: +28% (options RICH)  │
      │  Posture: NEUTRAL — VIX contango, mild pullback, IV elevated vs realized  │
      └────────────────────────────────────────────────┘

    Box color: GREEN for RISK_ON, YELLOW for NEUTRAL, RED for RISK_OFF/DEFENSIVE.
    Gracefully handles unavailable data fields.
    """
    try:
        data = fetch_market_regime()
    except Exception:
        return

    posture = data.get("posture", "NEUTRAL")

    # Choose box/border color based on posture
    if HAS_FMT and fmt:
        if posture == "RISK_ON":
            box_color = fmt.Colors.GREEN
        elif posture in ("RISK_OFF", "DEFENSIVE"):
            box_color = fmt.Colors.RED if posture == "RISK_OFF" else fmt.Colors.YELLOW
        else:
            box_color = fmt.Colors.YELLOW
    else:
        box_color = ""

    # ── Build content lines ────────────────────────────────────────────────────
    inner_width = width - 4  # account for "  │ " and " │"

    def _field(label: str, val: Any, suffix: str = "") -> str:
        if val is None:
            return f"{label}: n/a"
        return f"{label}: {val}{suffix}"

    vix = data.get("vix")
    vix3m = data.get("vix_3m")
    vts = data.get("vix_term_structure", "?")
    pcr = data.get("options_pcr")
    spy_ret = data.get("spy_ret_5d")
    spy_hv = data.get("spy_hv_30")
    iv_prem = data.get("iv_premium")
    rationale = data.get("posture_rationale", "")

    # Line 1: VIX, VIX3M, term structure, PCR
    vix_str = f"{vix:.1f}" if vix is not None else "n/a"
    vix3m_str = f"{vix3m:.1f}" if vix3m is not None else "n/a"
    pcr_str = f"{pcr:.2f}" if pcr is not None else "n/a"
    line1 = f"VIX: {vix_str}  |  VIX3M: {vix3m_str}  |  Term: {vts}  |  PCR(SPY): {pcr_str}"

    # Line 2: SPY 5d, HV30, IV premium
    spy_ret_str = f"{spy_ret*100:+.1f}%" if spy_ret is not None else "n/a"
    spy_hv_str = f"{spy_hv*100:.1f}%" if spy_hv is not None else "n/a"
    if iv_prem is not None:
        rich_label = "options RICH" if iv_prem > 0.10 else ("options CHEAP" if iv_prem < -0.05 else "near fair")
        iv_prem_str = f"{iv_prem*100:+.0f}% ({rich_label})"
    else:
        iv_prem_str = "n/a"
    line2 = f"SPY 5d: {spy_ret_str}  |  SPY HV30: {spy_hv_str}  |  IV Premium: {iv_prem_str}"

    # Line 3: posture + rationale
    rat_trunc = rationale[:inner_width - 20] if len(rationale) > inner_width - 20 else rationale
    line3 = f"Posture: {posture} \u2014 {rat_trunc}"

    # ── Draw box ───────────────────────────────────────────────────────────────
    title = " MARKET REGIME "
    # Top border with centered title
    side_len = (width - len(title) - 2) // 2
    side_r = width - len(title) - 2 - side_len
    top_bar = (
        BoxChars.TOP_LEFT
        + BoxChars.HORIZONTAL * side_len
        + title
        + BoxChars.HORIZONTAL * side_r
        + BoxChars.TOP_RIGHT
    ) if HAS_FMT else (
        "\u250c" + "\u2500" * side_len + title + "\u2500" * side_r + "\u2510"
    )
    bot_bar = (
        BoxChars.BOTTOM_LEFT + BoxChars.HORIZONTAL * (width - 2) + BoxChars.BOTTOM_RIGHT
    ) if HAS_FMT else (
        "\u2514" + "\u2500" * (width - 2) + "\u2518"
    )

    def _pad_line(content: str) -> str:
        """Pad a content string to fit inside box borders."""
        v = BoxChars.VERTICAL if HAS_FMT else "\u2502"
        padded = f"  {content}"
        # Pad to inner_width
        padded = padded.ljust(inner_width + 2)
        return f"{v}{padded}{v}"

    print()
    if HAS_FMT and fmt and box_color:
        print(fmt.colorize(top_bar, box_color, bold=True))
        print(fmt.colorize(_pad_line(line1), box_color))
        print(fmt.colorize(_pad_line(line2), box_color))
        # Posture line: color posture label distinctly
        posture_colored = line3
        print(fmt.colorize(_pad_line(posture_colored), box_color, bold=True))
        print(fmt.colorize(bot_bar, box_color, bold=True))
    else:
        print(top_bar)
        print(_pad_line(line1))
        print(_pad_line(line2))
        print(_pad_line(line3))
        print(bot_bar)
    print()


__all__ = [
    "fetch_market_regime",
    "print_regime_dashboard",
]
