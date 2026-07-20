"""SQLite record of real TFSA fills (data/longterm.db).

Separate DB on purpose — zero schema-migration risk to paper_trades.db.
A tranche is "filled" when a row here references its (ticker, level)."""
import contextlib
import datetime as _dt
import os
import sqlite3
from dataclasses import dataclass
from typing import List, Optional, Set

DEFAULT_DB = "data/longterm.db"

_SCHEMA = """CREATE TABLE IF NOT EXISTS fills (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    level REAL NOT NULL,
    shares REAL NOT NULL,
    price REAL NOT NULL,
    fill_date TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT ''
)"""


@dataclass
class Fill:
    id: int
    ticker: str
    level: float
    shares: float
    price: float
    fill_date: str
    note: str


@contextlib.contextmanager
def _conn(db_path: str = DEFAULT_DB):
    """Yield a sqlite3 connection; commits on success, rolls back on error, always closes."""
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute(_SCHEMA)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def record_fill(ticker: str, level: float, shares: float, price: float,
                fill_date: Optional[str] = None, note: str = "",
                db_path: str = DEFAULT_DB) -> int:
    if shares <= 0 or price <= 0:
        raise ValueError("shares and price must be positive")
    when = fill_date or _dt.date.today().isoformat()
    with _conn(db_path) as conn:
        cur = conn.execute(
            "INSERT INTO fills (ticker, level, shares, price, fill_date, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (str(ticker).upper(), float(level), float(shares), float(price), when, note))
        return int(cur.lastrowid)


def fills_for(ticker: Optional[str] = None, db_path: str = DEFAULT_DB) -> List[Fill]:
    with _conn(db_path) as conn:
        if ticker:
            rows = conn.execute(
                "SELECT id, ticker, level, shares, price, fill_date, note "
                "FROM fills WHERE ticker = ? ORDER BY id", (str(ticker).upper(),)).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, ticker, level, shares, price, fill_date, note "
                "FROM fills ORDER BY id").fetchall()
    return [Fill(*r) for r in rows]


def filled_levels(ticker: str, db_path: str = DEFAULT_DB) -> Set[float]:
    return {f.level for f in fills_for(ticker, db_path=db_path)}


def book(db_path: str = DEFAULT_DB) -> dict:
    out: dict = {}
    for f in fills_for(db_path=db_path):
        slot = out.setdefault(f.ticker, {"shares": 0.0, "cost": 0.0, "avg_price": 0.0})
        slot["shares"] += f.shares
        slot["cost"] += f.shares * f.price
    for slot in out.values():
        slot["avg_price"] = slot["cost"] / slot["shares"] if slot["shares"] else 0.0
    return out


def deployed_usd(db_path: str = DEFAULT_DB) -> float:
    return sum(s["cost"] for s in book(db_path).values())
