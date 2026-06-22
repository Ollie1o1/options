"""OHLCV cache for the breakout engine. Free daily history (yfinance) cached to
sqlite, fetched incrementally. Pure storage + an injectable fetcher so tests
never hit the network."""
from __future__ import annotations
import sqlite3
from typing import Callable, Dict, List, NamedTuple, Optional
import numpy as np

HORIZONS: Dict[str, int] = {"EOW": 5, "EOM": 21, "3M": 63}
THRESHOLDS = (0.05, 0.10, 0.20)
DEFAULT_DB = "data/equity_ohlcv.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ohlcv (
  ticker TEXT, date TEXT, close REAL, high REAL, low REAL, volume REAL,
  PRIMARY KEY (ticker, date)
)"""


class Series(NamedTuple):
    dates: List[str]
    close: np.ndarray
    high: np.ndarray
    low: np.ndarray
    volume: np.ndarray


def _conn(db_path: str) -> sqlite3.Connection:
    c = sqlite3.connect(db_path)
    c.execute(_SCHEMA)
    return c


def upsert_ohlcv(db_path: str, ticker: str, rows: List[dict]) -> int:
    if not rows:
        return 0
    with _conn(db_path) as c:
        before = c.total_changes
        c.executemany(
            "INSERT OR IGNORE INTO ohlcv(ticker,date,close,high,low,volume) "
            "VALUES (?,?,?,?,?,?)",
            [(ticker, r["date"], r["close"], r["high"], r["low"], r["volume"]) for r in rows],
        )
        return c.total_changes - before


def load_series(db_path: str, ticker: str) -> Optional[Series]:
    with _conn(db_path) as c:
        rows = c.execute(
            "SELECT date,close,high,low,volume FROM ohlcv WHERE ticker=? ORDER BY date",
            (ticker,),
        ).fetchall()
    if not rows:
        return None
    dates = [r[0] for r in rows]
    arr = lambda i: np.array([r[i] for r in rows], dtype=float)
    return Series(dates, arr(1), arr(2), arr(3), arr(4))


def _latest_date(db_path: str, ticker: str) -> Optional[str]:
    with _conn(db_path) as c:
        r = c.execute("SELECT MAX(date) FROM ohlcv WHERE ticker=?", (ticker,)).fetchone()
    return r[0] if r and r[0] else None


def update_universe(db_path: str, tickers: List[str],
                    fetcher: Callable[[str, Optional[str]], List[dict]]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for t in tickers:
        start = _latest_date(db_path, t)
        try:
            rows = fetcher(t, start)
        except Exception:
            rows = []
        if start:  # don't re-insert the boundary day
            rows = [r for r in rows if r["date"] > start]
        out[t] = upsert_ohlcv(db_path, t, rows)
    return out
