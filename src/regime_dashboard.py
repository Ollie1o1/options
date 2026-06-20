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
    from . import ui
    from .formatting import Colors, BoxChars, supports_color, colorize
    HAS_FMT = True
except ImportError:
    HAS_FMT = False
    fmt = None
    ui = None


def _print_gate_banner() -> None:
    """Phase 1 gate banner — printed when reports/GATE_STATUS.md exists."""
    from pathlib import Path
    p = Path("reports/GATE_STATUS.md")
    if not p.exists():
        return
    try:
        lines = p.read_text().splitlines()[:5]
    except Exception:
        return
    if not lines:
        return
    if HAS_FMT and ui:
        print(ui.rule(90, title="GATE STATUS"))
        for ln in lines:
            print(ln)
        print(ui.rule(90))
    else:
        border = "═" * 90
        print(border)
        for ln in lines:
            print(ln)
        print(border)


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


def classify_index_direction(
    price: Optional[float],
    ma50: Optional[float],
    ma200: Optional[float],
    ret_5d: Optional[float],
    ret_20d: Optional[float],
) -> Dict[str, Any]:
    """Classify an index's trend as UP / NEUTRAL / DOWN with an explainable reason.

    Transparent, rule-based score in [-1, +1] from four drivers:
      price vs 200d MA (0.40) — the primary "are we in an uptrend" line
      50d vs 200d MA  (0.30) — trend confirmation (golden / death cross)
      20d return      (0.20) — medium-term momentum
      5d return       (0.10) — is it accelerating or rolling over now
    Verdict: score > +0.3 → UP, < -0.3 → DOWN, else NEUTRAL.

    Degrades gracefully when the 200d MA is unavailable (history < ~1y): it
    falls back to price-vs-50d plus momentum and flags "limited history".
    Returns {"verdict": str, "reason": str, "score": float}.
    """
    def _sign(x: Optional[float]) -> int:
        if x is None:
            return 0
        return 1 if x > 0 else (-1 if x < 0 else 0)

    if price is None or ma50 is None:
        return {"verdict": "NEUTRAL", "reason": "insufficient data", "score": 0.0}

    if ma200 is None:
        # Short-history fallback: no 200d MA available.
        score = 0.5 * _sign(price - ma50) + 0.3 * _sign(ret_20d) + 0.2 * _sign(ret_5d)
        parts = [
            f"{'above' if price >= ma50 else 'below'} 50d ({price / ma50 - 1.0:+.1%})",
            f"20d {ret_20d:+.1%}" if ret_20d is not None else "20d n/a",
            "limited history (no 200d)",
        ]
    else:
        score = (
            0.40 * _sign(price - ma200)
            + 0.30 * _sign(ma50 - ma200)
            + 0.20 * _sign(ret_20d)
            + 0.10 * _sign(ret_5d)
        )
        parts = [
            f"{'above' if price >= ma200 else 'below'} 200d ({price / ma200 - 1.0:+.1%})",
            "50>200" if ma50 >= ma200 else "50<200",
            f"20d {ret_20d:+.1%}" if ret_20d is not None else "20d n/a",
            f"5d {ret_5d:+.1%}" if ret_5d is not None else "5d n/a",
        ]

    if score > 0.3:
        verdict = "UP"
    elif score < -0.3:
        verdict = "DOWN"
    else:
        verdict = "NEUTRAL"

    return {
        "verdict": verdict,
        "reason": ", ".join(parts),
        "score": round(score, 3),
        "ret_5d": ret_5d,
        "ret_20d": ret_20d,
    }


def direction_from_closes(closes) -> Dict[str, Any]:
    """Compute the direction verdict from a sequence of closing prices.

    Derives 50/200-day moving averages and 5/20-day returns from the tail of
    the series, then defers to classify_index_direction. Accepts a list or a
    pandas Series. Returns the same dict shape as classify_index_direction.
    """
    try:
        vals = [float(x) for x in list(closes) if x is not None]
    except (TypeError, ValueError):
        vals = []
    if not vals:
        return {"verdict": "NEUTRAL", "reason": "insufficient data", "score": 0.0}

    price = vals[-1]
    ma50 = sum(vals[-50:]) / min(len(vals), 50)
    ma200 = sum(vals[-200:]) / 200 if len(vals) >= 200 else None
    ret_5d = (price / vals[-6] - 1.0) if len(vals) >= 6 else None
    ret_20d = (price / vals[-21] - 1.0) if len(vals) >= 21 else None
    return classify_index_direction(price, ma50, ma200, ret_5d, ret_20d)


# Gauges in display order: broad market first, then the semiconductor complex
# (sector ETF + the two bellwether names most retail semi books are built on).
# Friendly name aids the user.
_INDEX_NAMES = {
    "SPY": "S&P 500",
    "QQQ": "Nasdaq 100",
    "IWM": "Russell 2000",
    "SMH": "Semiconductors",
    "NVDA": "Nvidia",
    "AMD": "AMD",
}

# Symbols above the divider are broad market; below are the semi complex.
_SEMI_SYMBOLS = ("SMH", "NVDA", "AMD")


def fetch_index_directions(symbols=("SPY", "QQQ", "IWM", "SMH", "NVDA", "AMD")) -> Dict[str, Any]:
    """Fetch ~1y of closes for each index and classify its direction.

    Returns an ordered dict {symbol: {"price", "verdict", "reason", "score"}}.
    Symbols that fail to fetch are omitted (graceful degradation), so an empty
    dict means market data was unavailable this run.
    """
    from concurrent.futures import ThreadPoolExecutor

    out: Dict[str, Any] = {}
    if not HAS_YF or not HAS_PD:
        return out

    def _one(sym: str):
        closes = _safe_hist(sym, "1y")
        if closes is None or len(closes) == 0:
            return sym, None
        res = direction_from_closes(closes.tolist())
        res["price"] = float(closes.iloc[-1])
        return sym, res

    try:
        with ThreadPoolExecutor(max_workers=min(4, len(symbols))) as pool:
            for sym, res in pool.map(_one, symbols):
                if res is not None:
                    out[sym] = res
    except Exception:
        pass
    return out


def print_market_direction(width: int = 100) -> None:
    """Print a compact, color-coded MARKET DIRECTION box for the broad indices.

    One row per index: friendly name, ticker, last price, an UP/NEUTRAL/DOWN
    verdict with an arrow, and the plain-English reason. Colors: green=UP,
    yellow=NEUTRAL, red=DOWN. Silently prints nothing if no data is available.
    """
    # The reason string now carries four drivers (200d, cross, 20d, 5d); give it
    # room so the 5d momentum read is never the part that gets truncated away.
    width = max(width, 100)
    data = fetch_index_directions()
    if not data:
        return

    # verdict → (color, arrow)
    if HAS_FMT and fmt:
        style = {
            "UP": (fmt.Colors.GREEN, "▲"),
            "NEUTRAL": (fmt.Colors.YELLOW, "▬"),
            "DOWN": (fmt.Colors.RED, "▼"),
        }
    else:
        style = {"UP": ("", "^"), "NEUTRAL": ("", "="), "DOWN": ("", "v")}

    inner_width = width - 4

    def _row(sym: str, info: Dict[str, Any]) -> str:
        name = _INDEX_NAMES.get(sym, sym)
        price = info.get("price")
        verdict = info.get("verdict", "NEUTRAL")
        reason = info.get("reason", "")
        _, arrow = style.get(verdict, ("", "?"))
        price_str = f"${price:,.2f}" if price is not None else "n/a"
        head = f"{name:<13} {sym:<4} {price_str:>11}   {arrow} {verdict:<7}"
        room = inner_width - len(head) - 2
        if room > 4 and reason:
            if len(reason) > room:
                reason = reason[: room - 1] + "…"
            return f"{head}  {reason}"
        return head

    # ── Section: titled rule + rows, color only the verdict-bearing row ────────
    def _pad_line(content: str) -> str:
        return f"  {content}"

    print()
    if HAS_FMT and ui:
        print(ui.rule(width, title="MARKET DIRECTION"))
    else:
        print("-" * width)
        print("  MARKET DIRECTION")
    printed_divider = False
    for sym, info in data.items():
        # Visual divider where the broad-market block ends and semis begin.
        if sym in _SEMI_SYMBOLS and not printed_divider:
            if HAS_FMT and ui:
                print(ui.rule(width, title="SEMICONDUCTORS"))
            else:
                label = " SEMICONDUCTORS "
                print(_pad_line(label + "─" * max(0, inner_width - len(label))))
            printed_divider = True
        line = _row(sym, info)
        if HAS_FMT and fmt:
            color, _ = style.get(info.get("verdict", "NEUTRAL"), ("", ""))
            print(fmt.colorize(_pad_line(line), color))
        else:
            print(_pad_line(line))

    # ── Synthesis: separate the slow trend from the fast momentum, because a
    # name can be "UP" (above 200d) while actively dipping (negative 5d/20d).
    verdicts = [i.get("verdict") for i in data.values()]
    n_up = verdicts.count("UP")
    n_down = verdicts.count("DOWN")
    r20 = [i.get("ret_20d") for i in data.values() if i.get("ret_20d") is not None]
    r5 = [i.get("ret_5d") for i in data.values() if i.get("ret_5d") is not None]
    avg20 = sum(r20) / len(r20) if r20 else 0.0
    avg5 = sum(r5) / len(r5) if r5 else 0.0
    if n_up > n_down:
        trend = "uptrend intact"
    elif n_down > n_up:
        trend = "downtrend"
    else:
        trend = "mixed trend"
    if avg20 < -0.005 and avg5 < 0:
        mom = f"but short-term momentum NEGATIVE (avg 20d {avg20:+.1%}, 5d {avg5:+.1%}) — pullback"
    elif avg20 > 0.005 and avg5 > 0:
        mom = f"and short-term momentum POSITIVE (avg 20d {avg20:+.1%}, 5d {avg5:+.1%})"
    else:
        mom = f"momentum flat/mixed (avg 20d {avg20:+.1%}, 5d {avg5:+.1%})"
    synth = f"Read: {trend}, {mom}"
    if len(synth) > inner_width:
        synth = synth[: inner_width - 1] + "…"
    if HAS_FMT and fmt:
        print(_pad_line(fmt.style(synth, 'heading')))
        print(ui.rule(width))
    else:
        print(_pad_line(synth))
        print("-" * width)


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
    _print_gate_banner()
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

    # ── Section: titled rule + kv rows; posture semantics live on the value ────
    print()
    if HAS_FMT and fmt and ui:
        posture_style = ('good' if posture == "RISK_ON"
                         else 'bad' if posture == "RISK_OFF"
                         else 'warn')
        print(ui.rule(width, title="MARKET REGIME"))
        print(ui.kv_line("Volatility", [f"VIX: {vix_str}", f"VIX3M: {vix3m_str}",
                                        f"Term: {vts}", f"PCR(SPY): {pcr_str}"]))
        print(ui.kv_line("SPY", [f"5d: {spy_ret_str}", f"HV30: {spy_hv_str}",
                                 f"IV Premium: {iv_prem_str}"]))
        print(ui.kv_line("Posture", [fmt.style(posture, posture_style, bold=True),
                                     fmt.style(rat_trunc, 'muted')], sep='  \u2014  '))
        print(ui.rule(width))
    else:
        print("-" * width)
        print("  MARKET REGIME")
        print(f"  {line1}")
        print(f"  {line2}")
        print(f"  {line3}")
        print("-" * width)
    print()

    # Broad-market direction gauge (SPY / QQQ / IWM) — is the market up or down?
    try:
        print_market_direction(width)
    except Exception:
        pass

    # World-news pulse (trust-weighted multi-source; failure-safe, ~8s budget).
    try:
        print_world_pulse()
    except Exception:
        pass

    # Hard macro / rates from FRED (no key; failure-safe).
    try:
        from src.macro_rates import print_macro_rates
        print_macro_rates(width)
        print()
    except Exception:
        pass

    # Point-in-time news archive coverage (factual one-liner).
    try:
        from src.news_archive import archive_stats, format_stats_line
        stats = archive_stats()
        if stats.get("total"):
            print(format_stats_line(stats))
            print()
    except Exception:
        pass


def print_world_pulse() -> None:
    """One-line world-news pulse under the regime box. Never raises."""
    try:
        from src.worldnews import panel, scoring, sources
        items = sources.fetch_all()
        if not items:
            return
        line = panel.pulse_line(scoring.aggregate(items), sources.fetch_crowd())
        if HAS_FMT and fmt and ui:
            print(ui.kv_line("Pulse", fmt.style(line, 'emph')))
        else:
            print(f"  {line}")
        print()
    except Exception:
        pass


__all__ = [
    "fetch_market_regime",
    "print_regime_dashboard",
    "classify_index_direction",
    "direction_from_closes",
    "fetch_index_directions",
    "print_market_direction",
]
