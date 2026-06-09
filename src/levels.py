"""Support / resistance levels and an empirical bounce-rate for an index or stock.

The point of this module is to turn "is it dipping?" into two concrete,
decision-grade answers:

  1. WHERE — the price levels below (support) and above (resistance) that the
     market actually reacts to: the 50/200-day moving averages and recent swing
     lows/highs, each with its % distance from the current price.

  2. HOW LIKELY / HOW LONG — an *empirical* bounce probability. Rather than
     inventing a number, we ask the only honest question the data can answer:
     in this name's own history, when it had already sold off this much over the
     last few days, how often was it higher 5 / 10 / 20 trading days later, and
     by how much? That base rate, with its sample size, is the bounce odds.

All the math lives in pure functions (no network) so it is unit-tested offline.
`analyze_symbol` / `print_levels` add the yfinance fetch and the display.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    from src import formatting as _fmt
    _HAS_FMT = True
except Exception:  # pragma: no cover
    _fmt = None
    _HAS_FMT = False


def _clean(closes: Sequence) -> List[float]:
    try:
        return [float(x) for x in list(closes) if x is not None]
    except (TypeError, ValueError):
        return []


def rsi(closes: Sequence, period: int = 14) -> Optional[float]:
    """Classic Wilder RSI(period) on a close series. None if too short.

    RSI < 30 is the textbook "oversold / bounce-prone" zone; > 70 is overbought.
    """
    vals = _clean(closes)
    if len(vals) <= period:
        return None
    gains, losses = 0.0, 0.0
    for i in range(len(vals) - period, len(vals)):
        ch = vals[i] - vals[i - 1]
        if ch >= 0:
            gains += ch
        else:
            losses -= ch
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return round(100.0 - 100.0 / (1.0 + rs), 1)


def support_resistance_levels(closes: Sequence, current: Optional[float] = None) -> Dict[str, Any]:
    """Identify support (below) and resistance (above) price levels.

    Candidates: 50d MA, 200d MA, the 20- and 60-day swing low (support) and
    swing high (resistance). Each is classified relative to the current price
    and tagged with its % distance. Supports are returned nearest-first
    (descending price); resistances nearest-first (ascending price).

    Returns {"price", "supports":[{label, level, pct}], "resistances":[...]}.
    pct is the level's distance from price as a signed fraction (support is
    negative, resistance positive).
    """
    vals = _clean(closes)
    if not vals:
        return {"price": None, "supports": [], "resistances": []}
    price = float(current) if current is not None else vals[-1]

    candidates: List[Tuple[str, float]] = []
    candidates.append(("50d MA", sum(vals[-50:]) / min(len(vals), 50)))
    if len(vals) >= 200:
        candidates.append(("200d MA", sum(vals[-200:]) / 200))
    if len(vals) >= 20:
        candidates.append(("20d low", min(vals[-20:])))
        candidates.append(("20d high", max(vals[-20:])))
    if len(vals) >= 60:
        candidates.append(("60d low", min(vals[-60:])))
        candidates.append(("60d high", max(vals[-60:])))

    supports, resistances = [], []
    for label, level in candidates:
        pct = level / price - 1.0
        entry = {"label": label, "level": round(level, 2), "pct": round(pct, 4)}
        if level < price:
            supports.append(entry)
        elif level > price:
            resistances.append(entry)
    # De-dupe levels that collide within 0.1% (e.g. 20d low == 60d low).
    supports = _dedupe(sorted(supports, key=lambda e: -e["level"]))
    resistances = _dedupe(sorted(resistances, key=lambda e: e["level"]))
    return {"price": round(price, 2), "supports": supports, "resistances": resistances}


def _dedupe(levels: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for lv in levels:
        if out and abs(lv["level"] / out[-1]["level"] - 1.0) < 0.001:
            out[-1]["label"] += " / " + lv["label"]
        else:
            out.append(lv)
    return out


def bounce_stats(
    closes: Sequence,
    lookback_drop_days: int = 5,
    horizons: Sequence[int] = (5, 10, 20),
) -> Dict[str, Any]:
    """Empirical bounce odds, conditioned on the current selloff's magnitude.

    Measures the current trailing `lookback_drop_days` return, then scans the
    full history for every day that had fallen *as much or more* over the same
    window. For each such day it records the forward return at each horizon.
    The fraction of those forward returns that are positive is the bounce rate;
    the median is the typical move. Sample size (`n`) is reported so a thin,
    untrustworthy base rate is obvious.

    Returns {"trailing_return", "lookback_days",
             "by_horizon": {h: {"n", "bounce_rate", "median", "p25", "p75"}}}.
    """
    vals = _clean(closes)
    out: Dict[str, Any] = {
        "trailing_return": None,
        "lookback_days": lookback_drop_days,
        "by_horizon": {},
    }
    if len(vals) <= lookback_drop_days + max(horizons):
        return out

    r_now = vals[-1] / vals[-1 - lookback_drop_days] - 1.0
    out["trailing_return"] = round(r_now, 4)
    eps = 1e-9

    for h in horizons:
        fwd: List[float] = []
        # t ranges over days that have both a full trailing window behind them
        # and a full horizon ahead of them.
        for t in range(lookback_drop_days, len(vals) - h):
            trail = vals[t] / vals[t - lookback_drop_days] - 1.0
            if trail <= r_now + eps:  # selloff as bad as (or worse than) now
                fwd.append(vals[t + h] / vals[t] - 1.0)
        if fwd:
            fwd_sorted = sorted(fwd)
            n = len(fwd)
            out["by_horizon"][h] = {
                "n": n,
                "bounce_rate": round(sum(1 for x in fwd if x > 0) / n, 3),
                "median": round(fwd_sorted[n // 2], 4),
                "p25": round(fwd_sorted[int(n * 0.25)], 4),
                "p75": round(fwd_sorted[int(n * 0.75)], 4),
            }
        else:
            out["by_horizon"][h] = {"n": 0, "bounce_rate": None,
                                    "median": None, "p25": None, "p75": None}
    return out


# ── Fetch + display ────────────────────────────────────────────────────────────

_NAMES = {
    "SPY": "S&P 500", "QQQ": "Nasdaq 100", "IWM": "Russell 2000",
    "SMH": "Semiconductors", "NVDA": "Nvidia", "AMD": "AMD",
}


def analyze_symbol(sym: str, period: str = "3y") -> Optional[Dict[str, Any]]:
    """Fetch history and compute levels + bounce stats for one symbol.

    Returns None if data is unavailable. Uses ~3y of daily closes so the bounce
    base rate has enough conditioned samples to be meaningful.
    """
    from src.regime_dashboard import _safe_hist  # reuse the resilient fetcher

    closes = _safe_hist(sym, period)
    if closes is None or len(closes) == 0:
        return None
    series = closes.tolist()
    res = support_resistance_levels(series)
    res["bounce"] = bounce_stats(series)
    res["rsi"] = rsi(series)
    res["symbol"] = sym
    res["name"] = _NAMES.get(sym, sym)
    return res


def print_levels(symbols: Sequence[str] = ("SPY", "SMH", "NVDA")) -> None:
    """Print support/resistance levels and the empirical bounce table per symbol."""
    c = _fmt.Colors if _HAS_FMT else None

    def col(text: str, color: str, bold: bool = False) -> str:
        return _fmt.colorize(text, color, bold=bold) if _HAS_FMT else text

    for sym in symbols:
        info = analyze_symbol(sym)
        if info is None:
            print(f"  {sym}: data unavailable")
            continue
        price = info["price"]
        rsi_v = info["rsi"]
        rsi_tag = ""
        if rsi_v is not None:
            zone = "oversold" if rsi_v < 30 else ("overbought" if rsi_v > 70 else "neutral")
            rsi_tag = f"   RSI {rsi_v} ({zone})"
        header = f"{info['name']} ({sym})  ${price:,.2f}{rsi_tag}"
        print()
        print(col("─" * 78, c.CYAN, True) if _HAS_FMT else "─" * 78)
        print(col(header, c.BRIGHT_CYAN, True) if _HAS_FMT else header)

        # Resistance (above) — listed top-down, furthest first then nearest.
        for r in reversed(info["resistances"]):
            line = f"   ↑ resistance  ${r['level']:>10,.2f}   {r['pct']:+.1%}   {r['label']}"
            print(col(line, c.RED) if _HAS_FMT else line)
        print(f"   • now         ${price:>10,.2f}    0.0%   current price")
        # Support (below) — nearest first going down.
        for s in info["supports"]:
            line = f"   ↓ support     ${s['level']:>10,.2f}   {s['pct']:+.1%}   {s['label']}"
            print(col(line, c.GREEN) if _HAS_FMT else line)

        b = info["bounce"]
        tr = b["trailing_return"]
        if tr is not None:
            print(col(
                f"   Bounce odds — given a {tr:+.1%} drop over the last "
                f"{b['lookback_days']}d, history says:", c.YELLOW)
                if _HAS_FMT else
                f"   Bounce odds (after a {tr:+.1%} / {b['lookback_days']}d drop):")
            print("     horizon   higher?   median move   typical range (25–75%)   n")
            for h, st in b["by_horizon"].items():
                if st["n"] == 0 or st["bounce_rate"] is None:
                    print(f"     {h:>3}d       n/a")
                    continue
                print(
                    f"     {h:>3}d       {st['bounce_rate']*100:>4.0f}%      "
                    f"{st['median']:+6.1%}        "
                    f"{st['p25']:+.1%} … {st['p75']:+.1%}     {st['n']}"
                )
