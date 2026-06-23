"""Load real ATM straddle entries and daily closes from data/dolt_options.db.
Entries carry the real dolt bid/ask (we sell at the bid — honest fills)."""
from __future__ import annotations
import datetime as _dt
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

TARGET_DTE = 30
FREQ_DAYS = 28


@dataclass(frozen=True)
class Entry:
    symbol: str
    date: str
    expiration: str
    strike: float
    spot: float
    straddle_bid: float   # call_bid + put_bid (received on the short)
    straddle_ask: float   # call_ask + put_ask
    iv: float             # ATM iv for the hedge reprice
    dte: int


def days_between(a: str, b: str) -> int:
    da = _dt.datetime.strptime(str(a)[:10], "%Y-%m-%d")
    db = _dt.datetime.strptime(str(b)[:10], "%Y-%m-%d")
    return (db - da).days


def closes(db_path: str, symbol: str) -> Dict[str, float]:
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT date, close FROM stocks_close WHERE symbol=? ORDER BY date",
            (symbol,)).fetchall()
    return {d: float(c) for d, c in rows}


def pick_entry(rows: List[Dict[str, Any]], spot: float, date: str,
               target_dte: int = TARGET_DTE) -> Optional[Entry]:
    valid = [r for r in rows if r.get("iv") is not None and r.get("strike") is not None
             and r.get("expiration") and r.get("bid") is not None and r.get("ask") is not None]
    if not valid or not spot:
        return None
    exp_dte: Dict[str, int] = {}
    for r in valid:
        d = days_between(date, r["expiration"])
        if d >= 0:
            exp_dte.setdefault(r["expiration"], d)
    if not exp_dte:
        return None
    best_exp = min(exp_dte, key=lambda e: abs(exp_dte[e] - target_dte))
    leg = [r for r in valid if r["expiration"] == best_exp]
    strike = min({float(r["strike"]) for r in leg}, key=lambda k: abs(k - float(spot)))
    calls = [r for r in leg if str(r["type"]).lower() == "call" and float(r["strike"]) == strike]
    puts = [r for r in leg if str(r["type"]).lower() == "put" and float(r["strike"]) == strike]
    if not calls or not puts:
        return None
    c, p = calls[0], puts[0]
    ivs = [float(x["iv"]) for x in (c, p) if x["iv"] and float(x["iv"]) > 0]
    if not ivs:
        return None
    return Entry(symbol=str(rows and ""), date=date, expiration=best_exp, strike=strike,
                 spot=float(spot), straddle_bid=float(c["bid"]) + float(p["bid"]),
                 straddle_ask=float(c["ask"]) + float(p["ask"]),
                 iv=sum(ivs) / len(ivs), dte=exp_dte[best_exp])


def straddle_entries(db_path: str, symbol: str, target_dte: int = TARGET_DTE,
                     freq_days: int = FREQ_DAYS) -> List[Entry]:
    px = closes(db_path, symbol)
    out: List[Entry] = []
    last_date: Optional[str] = None
    with sqlite3.connect(db_path) as conn:
        dates = [r[0] for r in conn.execute(
            "SELECT DISTINCT date FROM dolt_chain WHERE symbol=? ORDER BY date", (symbol,))]
        for d in dates:
            if last_date is not None and days_between(last_date, d) < freq_days:
                continue
            spot = px.get(d)
            if spot is None:
                continue
            rows = [dict(zip(("type", "strike", "expiration", "bid", "ask", "iv"), r))
                    for r in conn.execute(
                        "SELECT type,strike,expiration,bid,ask,iv FROM dolt_chain "
                        "WHERE symbol=? AND date=?", (symbol, d))]
            e = pick_entry(rows, spot, d, target_dte)
            if e is None:
                continue
            out.append(Entry(symbol=symbol, date=e.date, expiration=e.expiration,
                             strike=e.strike, spot=e.spot, straddle_bid=e.straddle_bid,
                             straddle_ask=e.straddle_ask, iv=e.iv, dte=e.dte))
            last_date = d
    return out
