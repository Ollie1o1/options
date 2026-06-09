"""Pure signal extractors for the Intel Briefing.

Every extractor takes already-fetched primitives (no network) and returns a
`Signal` normalized to [-1, +1] where positive is bullish-for-a-buyer. This
keeps the whole layer unit-testable offline and lets the verdict blend signals
on a common scale. Whether a signal actually *counts* toward the verdict is
decided later by `reliability.py` (a signal with no backtested edge is shown but
given zero weight).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class Signal:
    name: str
    value: float          # [-1, +1]; + bullish for a buyer, - bearish
    label: str            # short word, e.g. "UP", "oversold"
    detail: str           # human-readable reason
    directional: bool = True   # False = informational only (e.g. earnings risk)


def _clamp(x: float) -> float:
    return max(-1.0, min(1.0, x))


def _sign(x: Optional[float]) -> int:
    if x is None:
        return 0
    return 1 if x > 0 else (-1 if x < 0 else 0)


def trend_signal(price: Optional[float], ma50: Optional[float],
                 ma200: Optional[float]) -> Signal:
    """Primary trend from price vs the 200d and the 50/200 cross."""
    if price is None or ma50 is None:
        return Signal("trend", 0.0, "n/a", "insufficient price history")
    if ma200 is None:
        v = 0.6 * _sign(price - ma50)
        label = "up" if v > 0 else ("down" if v < 0 else "flat")
        return Signal("trend", _clamp(v), label,
                      f"price {'above' if price >= ma50 else 'below'} 50d "
                      f"(limited history, no 200d)")
    v = 0.6 * _sign(price - ma200) + 0.4 * _sign(ma50 - ma200)
    if v > 0.3:
        label = "UP"
    elif v < -0.3:
        label = "DOWN"
    else:
        label = "mixed"
    cross = "50>200" if ma50 >= ma200 else "50<200"
    rel = "above" if price >= ma200 else "below"
    return Signal("trend", _clamp(v), label,
                  f"{rel} 200d ({price / ma200 - 1.0:+.1%}), {cross}")


def momentum_signal(ret_5d: Optional[float], ret_20d: Optional[float]) -> Signal:
    """Short/medium momentum. Scaled so a ~5% move maps near full strength."""
    if ret_5d is None and ret_20d is None:
        return Signal("momentum", 0.0, "n/a", "no return data")
    r5 = ret_5d or 0.0
    r20 = ret_20d or 0.0
    v = _clamp((0.6 * r20 + 0.4 * r5) / 0.05)
    if v > 0.15:
        label = "rising"
    elif v < -0.15:
        label = "falling"
    else:
        label = "flat"
    return Signal("momentum", v, label, f"20d {r20:+.1%}, 5d {r5:+.1%}")


def bounce_signal(bounce_rate: Optional[float], n: Optional[int]) -> Signal:
    """Empirical bounce base-rate (from levels.bounce_stats) mapped to ±.

    Centered at 0.5 (a coin flip). A thin sample (n < 10) is de-emphasized
    toward zero rather than trusted.
    """
    if bounce_rate is None or not n:
        return Signal("bounce", 0.0, "n/a", "no bounce sample")
    v = _clamp((bounce_rate - 0.5) * 2.0)
    if n < 10:
        v *= 0.4  # thin sample: discount, don't trust
    label = f"{bounce_rate:.0%} hist."
    return Signal("bounce", v, label,
                  f"{bounce_rate:.0%} higher 10d after similar drops (n={n})")


def rsi_signal(rsi: Optional[float]) -> Signal:
    """Mean-reversion read: oversold (<30) is bullish, overbought (>70) bearish."""
    if rsi is None:
        return Signal("rsi", 0.0, "n/a", "no RSI")
    if rsi < 30:
        v = _clamp((30 - rsi) / 20.0)      # 30→0, 10→+1
        label = "oversold"
    elif rsi > 70:
        v = -_clamp((rsi - 70) / 20.0)     # 70→0, 90→-1
        label = "overbought"
    else:
        v = 0.0
        label = "neutral"
    return Signal("rsi", v, label, f"RSI {rsi:.0f}")


def support_proximity_signal(pct_to_support: Optional[float]) -> Signal:
    """How close price sits above its nearest support (a buy-the-dip cue).

    pct_to_support is a negative fraction (support below price). Closer = more
    bullish (better risk/reward to lean on the level).
    """
    if pct_to_support is None:
        return Signal("support", 0.0, "n/a", "no support level")
    dist = abs(pct_to_support)
    v = _clamp(max(0.0, (0.04 - dist) / 0.04))  # at support→+1, 4%+ away→0
    label = "at support" if dist < 0.015 else ("near support" if dist < 0.04 else "mid-range")
    return Signal("support", v, label, f"{dist:.1%} above nearest support")


def news_sentiment_signal(sentiment: Optional[float]) -> Signal:
    """Aggregate news sentiment in [-1, +1] from news_fetcher.

    Directional weighting is decided by the reliability backtest; by default
    this contributes nothing to the verdict (context only) until proven.
    """
    if sentiment is None:
        return Signal("news", 0.0, "n/a", "no recent news")
    v = _clamp(sentiment)
    if v > 0.15:
        label = "positive"
    elif v < -0.15:
        label = "negative"
    else:
        label = "neutral"
    return Signal("news", v, label, f"sentiment {v:+.2f}")


def analyst_signal(raises: Optional[int], cuts: Optional[int]) -> Signal:
    """Net analyst tone over the trailing window (raises minus cuts)."""
    r = raises or 0
    c = cuts or 0
    if r == 0 and c == 0:
        return Signal("analyst", 0.0, "n/a", "no analyst changes 30d")
    v = _clamp((r - c) / 3.0)
    label = "upgrades" if v > 0 else ("downgrades" if v < 0 else "mixed")
    return Signal("analyst", v, label, f"{r} raises / {c} cuts 30d")


def options_posture_signal(iv_rank: Optional[float], skew: Optional[float]) -> Signal:
    """Options structure read (informational, small directional weight).

    High IV rank = expensive premium (slightly cautionary for long premium);
    iv_rank in [0,1]. skew > 0 means puts bid over calls (downside fear).
    """
    if iv_rank is None and skew is None:
        return Signal("options", 0.0, "n/a", "no options data", directional=False)
    parts = []
    if iv_rank is not None:
        parts.append(f"IV rank {iv_rank:.0%} ({'rich' if iv_rank > 0.7 else 'cheap' if iv_rank < 0.3 else 'mid'})")
    if skew is not None:
        parts.append("skew rich puts" if skew > 0 else "skew flat/calls")
    # Mildly bearish if both expensive and put-skewed (fear regime).
    v = 0.0
    if iv_rank is not None and skew is not None:
        v = _clamp(-(0.5 * max(0.0, iv_rank - 0.5) + 0.5 * max(0.0, skew)))
    return Signal("options", v, "; ".join(parts) or "n/a", "; ".join(parts),
                  directional=False)


def earnings_signal(days_to_earnings: Optional[int]) -> Signal:
    """Event-risk gate (not directional). Negative magnitude = imminent risk."""
    if days_to_earnings is None:
        return Signal("earnings", 0.0, "none scheduled", "no earnings in view",
                      directional=False)
    d = days_to_earnings
    if d < 0:
        return Signal("earnings", 0.0, f"{-d}d ago", f"reported {-d}d ago",
                      directional=False)
    if d <= 3:
        v = -1.0
    elif d <= 10:
        v = -0.5
    else:
        v = -0.1
    return Signal("earnings", v, f"in {d}d", f"earnings in {d} days",
                  directional=False)


# Names of signals that are eligible to feed the verdict composite (the rest are
# informational). reliability.py decides each one's actual weight.
DIRECTIONAL_NAMES = (
    "trend", "momentum", "bounce", "rsi", "support", "news", "analyst",
)


def build_signals(state: Dict[str, Any]) -> Dict[str, Signal]:
    """Assemble all signals from a fetched `state` dict (see briefing.py).

    Missing keys degrade to n/a signals (value 0) rather than raising.
    """
    g = state.get
    return {
        "trend": trend_signal(g("price"), g("ma50"), g("ma200")),
        "momentum": momentum_signal(g("ret_5d"), g("ret_20d")),
        "bounce": bounce_signal(g("bounce_rate"), g("bounce_n")),
        "rsi": rsi_signal(g("rsi")),
        "support": support_proximity_signal(g("pct_to_support")),
        "news": news_sentiment_signal(g("news_sentiment")),
        "analyst": analyst_signal(g("analyst_raises"), g("analyst_cuts")),
        "options": options_posture_signal(g("iv_rank"), g("skew")),
        "earnings": earnings_signal(g("days_to_earnings")),
    }
