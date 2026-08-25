"""Straddle-implied move for the expiry that spans a catalyst.

This is the ONLY number on the board sourced from a price rather than a filing,
and it is the market's estimate, not ours. It answers "what magnitude is
already priced in", which is the honest version of "how big could this be".

Note what it is not: an edge. If the implied move looks large, that is the
market agreeing the event is binary, not a mispricing you have found.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence


@dataclass(frozen=True)
class ImpliedMove:
    expiry: Optional[str] = None
    spot: Optional[float] = None
    straddle: Optional[float] = None
    move_pct: Optional[float] = None


def _as_date(text: str) -> dt.date:
    parts = text.split("-")
    if len(parts) == 2:
        return dt.date(int(parts[0]), int(parts[1]), 1)
    return dt.date(int(parts[0]), int(parts[1]), int(parts[2]))


def pick_expiry(expiries: Sequence[str], event_date: str) -> Optional[str]:
    """First listed expiry at or after the event. None if none reaches it."""
    try:
        target = _as_date(event_date)
    except (ValueError, IndexError):
        return None
    candidates = []
    for text in expiries:
        try:
            parsed = _as_date(text)
        except (ValueError, IndexError):
            continue
        if parsed >= target:
            candidates.append((parsed, text))
    return min(candidates)[1] if candidates else None


def _price(row: Any) -> Optional[float]:
    """Mid when both sides are quoted, else last. A one-sided quote is not a
    mid, and averaging against a zero would halve the premium."""
    bid, ask = row.get("bid"), row.get("ask")
    if bid and ask and bid > 0 and ask > 0:
        return (float(bid) + float(ask)) / 2.0
    last = row.get("lastPrice")
    return float(last) if last else None


def _nearest(rows: Sequence[Any], spot: float) -> Optional[Any]:
    best, best_gap = None, None
    for row in rows:
        strike = row.get("strike")
        if strike is None:
            continue
        gap = abs(float(strike) - spot)
        if best_gap is None or gap < best_gap:
            best, best_gap = row, gap
    return best


def straddle_move(calls: Sequence[Any], puts: Sequence[Any],
                  spot: float) -> Optional[float]:
    """ATM straddle premium as a fraction of spot. None if either leg or the
    spot is missing."""
    if not calls or not puts or not spot or spot <= 0:
        return None
    call, put = _nearest(calls, spot), _nearest(puts, spot)
    if call is None or put is None:
        return None
    call_px, put_px = _price(call), _price(put)
    if call_px is None or put_px is None:
        return None
    return (call_px + put_px) / spot


def _expiries(ticker: str) -> List[str]:
    import yfinance as yf
    return list(yf.Ticker(ticker).options or [])


def _chain(ticker: str, expiry: str) -> Any:
    import yfinance as yf
    return yf.Ticker(ticker).option_chain(expiry)


def _spot(ticker: str) -> Optional[float]:
    import yfinance as yf
    value = yf.Ticker(ticker).fast_info.get("lastPrice")
    return float(value) if value else None


def implied_move(ticker: str, event_date: str) -> ImpliedMove:
    """Implied move for the expiry spanning ``event_date``. Never raises."""
    try:
        expiry = pick_expiry(_expiries(ticker), event_date)
        if not expiry:
            return ImpliedMove()
        spot = _spot(ticker)
        if not spot:
            return ImpliedMove(expiry=expiry)
        chain = _chain(ticker, expiry)
        calls = chain.calls.to_dict("records")
        puts = chain.puts.to_dict("records")
        move = straddle_move(calls, puts, spot)
        straddle = move * spot if move is not None else None
        return ImpliedMove(expiry=expiry, spot=spot, straddle=straddle,
                           move_pct=move)
    except Exception:
        return ImpliedMove()
