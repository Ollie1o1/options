"""Market-cap band for the catalyst universe.

$50M floor, $10B ceiling. Below the floor the names are mostly untradeable
shells; above the ceiling a single readout stops being material to the company,
which is the entire premise of the board.

An UNKNOWN cap excludes a ticker. Defaulting a missing number into the band
would silently admit exactly the delisted and illiquid names the band exists to
remove — APLS and ADVM both return no cap because they are delisted.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

MCAP_LO = 50e6
MCAP_HI = 10e9


def _ticker(symbol: str) -> Any:
    import yfinance as yf
    return yf.Ticker(symbol)


def in_band(mcap: Optional[float], lo: float = MCAP_LO,
            hi: float = MCAP_HI) -> bool:
    """Inclusive on both ends. None is out."""
    if mcap is None:
        return False
    return lo <= mcap <= hi


def market_caps(tickers: Sequence[str]) -> Dict[str, Optional[float]]:
    """Ticker → market cap, None where unavailable. Never raises."""
    out: Dict[str, Optional[float]] = {}
    for symbol in tickers:
        try:
            value = _ticker(symbol).fast_info.get("marketCap")
            out[symbol] = float(value) if value else None
        except Exception:
            out[symbol] = None
    return out
