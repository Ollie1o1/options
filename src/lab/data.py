"""The price substrate: 21,244 symbols, 2017-10-02 to 2026-07-21, 2,212 days.

`data/squeeze_prices.db` was built for the short-interest work but it is the
deepest price history in the repo by an order of magnitude, and it is the only
thing that can support a multi-year test of anything. `data/equity_ohlcv.db`
raises `disk I/O error` on every read including `.backup` and is treated as
absent.

Daily closes only — no intraday, no open/high/low. Ideas that need a stop or an
intraday touch cannot be tested here, which is one reason the harness's default
exit is a fixed hold.
"""
from __future__ import annotations

import sqlite3
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_PRICES = "data/squeeze_prices.db"

Bars = List[Tuple[str, float]]

# Liquid, optionable names that also carry real option marks in at least one of
# the two quote corpora — so a modelled result can be spot-checked against real
# ones where the DTE allows.
ANCHORED = ("AAPL", "MSFT", "NVDA", "AMD", "META", "AMZN", "GOOG", "SPY",
            "QQQ", "IWM", "TSLA", "INTC", "ORCL", "PLTR", "COIN")


# A one-day move this extreme is a corporate action, not a price move. The
# largest genuine single-day drop in a liquid US equity is around -40% (and
# those exist), so the threshold sits below that: adjusting a real crash away
# would be a worse bug than the one being fixed.
_SPLIT_DROP = -0.45
_SPLIT_JUMP = 1.50


def adjust_splits(bars: Bars) -> Bars:
    """Back-adjust raw closes so a split stops looking like a crash.

    `data/squeeze_prices.db` stores RAW prices. NVDA falls 89.9% on 2024-06-10,
    AMZN 94.9% on 2022-06-06, GOOG 95.1% on 2022-07-18, AAPL 74.2% on
    2020-08-31 — every one of them a split. Left alone these destroy long calls
    and manufacture long puts, which is precisely the shape of a backtest
    result that looks like an edge and is an artifact.

    Detected by ratio rather than from a corporate-actions table because no
    such table exists in this repo for the 21k-symbol universe. Everything
    before the event is scaled by the split ratio, so the series becomes
    continuous and total return is preserved."""
    if len(bars) < 2:
        return list(bars)
    # Split the pairs apart rather than mutating a list of mixed-type rows: the
    # dates never change, and keeping the closes in their own float list is what
    # makes the arithmetic below well-typed.
    dates = [d for d, _ in bars]
    closes = [float(c) for _, c in bars]
    # Walk backwards so each adjustment applies to everything already before it.
    for i in range(len(closes) - 1, 0, -1):
        prev, cur = closes[i - 1], closes[i]
        if prev <= 0 or cur <= 0:
            continue
        ratio = cur / prev
        if ratio - 1 <= _SPLIT_DROP or ratio - 1 >= _SPLIT_JUMP:
            for j in range(i):
                closes[j] *= ratio
    return list(zip(dates, closes))


def load_universe(symbols: Sequence[str], db_path: str = DEFAULT_PRICES,
                  start: Optional[str] = None, end: Optional[str] = None,
                  min_bars: int = 300, adjust_splits_: bool = True,
                  **kw) -> Dict[str, Bars]:
    """Daily closes per symbol, in date order.

    Symbols with fewer than `min_bars` are dropped rather than tested on thin
    history: their features would be noisier than everything they are being
    compared against, which shows up as spurious edge."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        out: Dict[str, Bars] = {}
        for sym in symbols:
            q = "SELECT date, close FROM px WHERE symbol=?"
            args: List[object] = [sym]
            if start:
                q += " AND date>=?"
                args.append(start)
            if end:
                q += " AND date<=?"
                args.append(end)
            q += " ORDER BY date"
            bars = [(d, float(c)) for d, c in conn.execute(q, args)
                    if c is not None and float(c) > 0]
            if len(bars) < min_bars:
                continue
            if kw.get("adjust_splits", adjust_splits_):
                bars = adjust_splits(bars)
            out[sym] = bars
        return out
    finally:
        conn.close()


def available_symbols(db_path: str = DEFAULT_PRICES, min_bars: int = 1000,
                      limit: Optional[int] = None) -> List[str]:
    """Symbols with enough history to be worth testing."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        q = ("SELECT symbol, COUNT(*) n FROM px GROUP BY symbol "
             "HAVING n >= ? ORDER BY n DESC")
        if limit:
            q += f" LIMIT {int(limit)}"
        return [r[0] for r in conn.execute(q, (min_bars,))]
    finally:
        conn.close()
