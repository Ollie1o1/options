"""Synthetic chain archive and price history for exercising the study end to end.

Nothing here is market data. Spots follow seeded GBM paths; puts and calls are priced by
Black-Scholes with a put skew; each contract's bid-ask width is drawn per contract so the
selector's delta-nearest choice and the baseline's tightest choice often differ. Monthly
(third-Friday) expirations quote tighter, as measured in the real archive. Two coverage
defects are planted: one provider failure and one missing chain.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
from typing import Any, Dict, List, Sequence

import numpy as np

SYMBOLS = ("SYNA", "SYNB", "SYNC", "SYND", "SYNE", "SYNF")


def _ncdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_put(s: float, k: float, t: float, iv: float, r: float = 0.04) -> tuple[float, float]:
    sd = iv * math.sqrt(t)
    d1 = (math.log(s / k) + (r + 0.5 * iv * iv) * t) / sd
    d2 = d1 - sd
    price = k * math.exp(-r * t) * _ncdf(-d2) - s * _ncdf(-d1)
    return max(price, 0.0), _ncdf(d1) - 1.0


def _sessions(start: str, n: int) -> List[str]:
    from src.data_quality import is_trading_day
    out: List[str] = []
    d = dt.date.fromisoformat(start)
    while len(out) < n:
        if is_trading_day(d):
            out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def _fridays(after: dt.date, horizon: int) -> List[dt.date]:
    d = after + dt.timedelta(days=(4 - after.weekday()) % 7 or 7)
    out: List[dt.date] = []
    while (d - after).days <= horizon:
        out.append(d)
        d += dt.timedelta(days=7)
    return out


def build(directory: str, *, sessions: int = 85, start: str = "2026-01-05", seed: int = 7,
          symbols: Sequence[str] = SYMBOLS) -> Dict[str, Any]:
    """Write `chain_archive.db` and `equity_ohlcv.db` under `directory`."""
    from src.chain_archive import ensure_db

    os.makedirs(directory, exist_ok=True)
    chain_db = os.path.join(directory, "chain_archive.db")
    ohlcv_db = os.path.join(directory, "equity_ohlcv.db")
    for p in (chain_db, ohlcv_db):
        if os.path.exists(p):
            os.remove(p)
    ensure_db(chain_db)
    days = _sessions(start, sessions)
    rng = np.random.default_rng(seed)
    base = {s: 50.0 + 40.0 * i for i, s in enumerate(symbols)}
    vols = {s: 0.18 + 0.04 * (i % 3) for i, s in enumerate(symbols)}
    paths: Dict[str, List[float]] = {}
    for s in symbols:
        shocks = rng.normal(0.0002, vols[s] / math.sqrt(252), size=len(days))
        paths[s] = list(base[s] * np.exp(np.cumsum(shocks)))
    width_draw = rng.uniform(0.02, 0.16, size=(len(symbols), 4000))
    rows: List[tuple] = []
    attempts: List[tuple] = []
    planted = {"provider_failure": (symbols[0], days[10]), "missing_chain": (symbols[1], days[20])}
    for si, s in enumerate(symbols):
        step = 1.0 if base[s] < 120 else 2.5
        for di, day in enumerate(days):
            if (s, day) in planted.values():
                if (s, day) == planted["provider_failure"]:
                    attempts.append((day, s, "error", "ConnectionError: synthetic outage", 1, f"{day}T19:45:00Z"))
                continue
            spot = paths[s][di]
            d0 = dt.date.fromisoformat(day)
            ts = f"{day}T19:45:00Z"
            for exp in _fridays(d0, 70):
                dte = (exp - d0).days
                monthly = 15 <= exp.day <= 21
                t = dte / 365.0
                k = math.floor(spot * 0.85 / step) * step
                while k <= spot * 1.15:
                    if k > 0:
                        iv = vols[s] * (1.0 + 0.8 * max(0.0, (spot - k) / spot))
                        price, delta = _bs_put(spot, k, t, iv)
                        cid = int(round(k / step)) % 4000
                        rel = width_draw[si, cid] * (0.6 if monthly else 1.0)
                        half = max(0.01, 0.5 * rel * max(price, 0.05))
                        bid, ask = round(max(0.0, price - half), 2), round(price + half, 2)
                        code = f"{s}{exp:%y%m%d}P{int(k * 1000):08d}"
                        rows.append((s, day, code, "put", k, exp.isoformat(), bid, ask, None, None, iv, delta,
                                     None, None, None, None, 100.0, 10.0, ts, spot, ts, "synthetic"))
                        cprice = price + spot - k * math.exp(-0.04 * t)
                        rows.append((s, day, f"{s}{exp:%y%m%d}C{int(k * 1000):08d}", "call", k, exp.isoformat(),
                                     round(max(0.0, cprice - half), 2), round(cprice + half, 2), None, None,
                                     iv, delta + 1.0, None, None, None, None, 100.0, 10.0, ts, spot, ts,
                                     "synthetic"))
                    k += step
            attempts.append((day, s, "archived", None, 1, ts))
    with sqlite3.connect(chain_db) as con:
        con.executemany("INSERT OR IGNORE INTO chain_snapshots VALUES (" + ",".join("?" * 22) + ")", rows)
        con.executemany("INSERT OR REPLACE INTO archive_attempts VALUES (?,?,?,?,?,?)", attempts)
    with sqlite3.connect(ohlcv_db) as con:
        con.execute("CREATE TABLE ohlcv (ticker TEXT, date TEXT, close REAL, high REAL, low REAL, volume REAL, "
                    "PRIMARY KEY (ticker, date))")
        con.executemany("INSERT INTO ohlcv VALUES (?,?,?,?,?,?)",
                        [(s, d, p, p, p, 1e6) for s in symbols for d, p in zip(days, paths[s])])
    return {"chain_db": chain_db, "ohlcv_db": ohlcv_db, "sessions": days, "symbols": list(symbols),
            "planted": {k: list(v) for k, v in planted.items()}, "rows": len(rows)}
