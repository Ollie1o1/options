"""Point-in-time shares outstanding from SEC EDGAR XBRL.

FINRA reports shares short but not float, and the grader's gate is short
interest as a fraction of float. EDGAR's ``dei:EntityCommonStockSharesOutstanding``
concept returns a company's whole filing history in one small request, and — the
reason to use it over yfinance — the filings of companies that later delisted do
not disappear.

Two point-in-time hazards are handled explicitly, because both would silently
corrupt the gate:

1. **Look-ahead.** A value is only usable once it has been *filed*, so lookups
   select the latest point with ``filed <= asof``, never the latest ``end``.
2. **Ticker reuse.** ``company_tickers.json`` is a present-day snapshot, so a
   recycled ticker maps to whoever holds it now (BBBY today resolves to Beyond,
   Inc., not the Bed Bath & Beyond that went bankrupt in 2023). A CIK whose
   filing history does not straddle the date being asked about is treated as
   unusable rather than silently wrong.

Shares outstanding exceeds float — it includes insider and restricted stock — so
the raw ratio is NOT directly comparable to a 15%/20%-of-float threshold. Use
``panel.grade(..., si_scale=...)`` to state the assumed float fraction, and sweep
it to show whether a conclusion survives wherever that line is drawn.

CLI:
    python -m src.squeeze.backtest.shares --backfill
    python -m src.squeeze.backtest.shares --stats
"""
from __future__ import annotations

import bisect
import sqlite3
import time
from typing import Dict, Iterable, List, Optional, Tuple

from src.squeeze.backtest import DEFAULT_DB, SHARES_DB

CONCEPT_URL = ("https://data.sec.gov/api/xbrl/companyconcept/"
               "CIK{cik:010d}/dei/EntityCommonStockSharesOutstanding.json")

_DDL = [
    """CREATE TABLE IF NOT EXISTS shares_out (
        symbol  TEXT NOT NULL,
        filed   TEXT NOT NULL,
        end_date TEXT,
        shares  REAL,
        PRIMARY KEY (symbol, filed)
    )""",
    """CREATE TABLE IF NOT EXISTS shares_fetch (
        symbol TEXT PRIMARY KEY,
        cik    INTEGER,
        points INTEGER,
        status TEXT
    )""",
]


def connect(db_path: str = SHARES_DB) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=60)
    conn.execute("PRAGMA journal_mode=WAL")
    for ddl in _DDL:
        conn.execute(ddl)
    conn.commit()
    return conn


def fetch_symbol(symbol: str, cik: int) -> List[Tuple[str, str, float]]:
    """(filed, end, shares) points for one CIK; [] if the concept is absent."""
    from src.insider.edgar import _get
    resp = _get(CONCEPT_URL.format(cik=cik), timeout=30)
    data = resp.json()
    pts = (data.get("units") or {}).get("shares") or []
    out = []
    for p in pts:
        filed, val = p.get("filed"), p.get("val")
        if not filed or val is None:
            continue
        try:
            out.append((filed, p.get("end"), float(val)))
        except (TypeError, ValueError):
            continue
    out.sort()
    return out


def backfill(symbols: Iterable[str], db_path: str = SHARES_DB,
             verbose: bool = True) -> Dict[str, int]:
    """Fetch shares-outstanding history for every symbol not already tried."""
    from src.insider.edgar import _ticker_map
    tmap = _ticker_map()
    conn = connect(db_path)
    done = {r[0] for r in conn.execute("SELECT symbol FROM shares_fetch")}
    conn.close()
    todo = sorted(s for s in symbols if s not in done)
    counts = {"ok": 0, "no_cik": 0, "no_data": 0, "error": 0}

    # Rows are buffered and flushed in short bursts. Holding a write transaction
    # open across the network fetches would lock the database for minutes at a
    # time and starve anything else writing to it.
    buf_points: List[Tuple[str, str, Optional[str], float]] = []
    buf_status: List[Tuple[str, Optional[int], int, str]] = []

    def flush():
        if not buf_points and not buf_status:
            return
        c = connect(db_path)
        try:
            if buf_points:
                c.executemany(
                    "INSERT OR REPLACE INTO shares_out(symbol,filed,end_date,shares) "
                    "VALUES(?,?,?,?)", buf_points)
            if buf_status:
                c.executemany("INSERT OR REPLACE INTO shares_fetch VALUES(?,?,?,?)",
                              buf_status)
            c.commit()
        finally:
            c.close()
        buf_points.clear()
        buf_status.clear()

    for i, sym in enumerate(todo, 1):
        cik = tmap.get(sym)
        if cik is None:
            buf_status.append((sym, None, 0, "no_cik"))
            counts["no_cik"] += 1
        else:
            try:
                pts = fetch_symbol(sym, cik)
                if pts:
                    buf_points.extend((sym, f, e, v) for f, e, v in pts)
                    counts["ok"] += 1
                    status = "ok"
                else:
                    counts["no_data"] += 1
                    status = "no_data"
                buf_status.append((sym, cik, len(pts), status))
            except Exception as exc:
                msg = str(exc)
                status = "no_data" if "404" in msg else "error"
                counts["no_data" if status == "no_data" else "error"] += 1
                buf_status.append((sym, cik, 0, status))
        if i % 200 == 0:
            flush()
            if verbose:
                print(f"  [{i}/{len(todo)}] {counts}", flush=True)
    flush()
    return counts


# ── point-in-time lookup ────────────────────────────────────────────────────
class SharesLookup:
    """As-of-date shares outstanding, loaded once into memory.

    ``get(symbol, asof)`` returns the most recent value *filed* on or before
    ``asof``, or None when the symbol has no filing by then — which is the
    correct answer for a company that had not yet IPO'd, and the safe answer
    for a recycled ticker whose current CIK was filing under a different name.
    """

    def __init__(self, db_path: str = SHARES_DB):
        conn = sqlite3.connect(db_path, timeout=60)
        rows = conn.execute(
            "SELECT symbol, filed, shares FROM shares_out ORDER BY symbol, filed"
        ).fetchall()
        conn.close()
        self._by_symbol: Dict[str, Tuple[List[str], List[float]]] = {}
        for sym, filed, shares in rows:
            filings, values = self._by_symbol.setdefault(sym, ([], []))
            filings.append(filed)
            values.append(shares)

    def get(self, symbol: str, asof: str) -> Optional[float]:
        entry = self._by_symbol.get(symbol)
        if not entry:
            return None
        filings, values = entry
        idx = bisect.bisect_right(filings, asof) - 1
        if idx < 0:
            return None
        val = values[idx]
        return val if val and val > 0 else None

    def __len__(self) -> int:
        return len(self._by_symbol)


def stats(db_path: str = SHARES_DB) -> dict:
    conn = connect(db_path)
    by_status = dict(conn.execute(
        "SELECT status, COUNT(*) FROM shares_fetch GROUP BY status").fetchall())
    pts = conn.execute("SELECT COUNT(*), COUNT(DISTINCT symbol) FROM shares_out").fetchone()
    conn.close()
    return {"attempted": sum(by_status.values()), **by_status,
            "points": pts[0], "symbols_with_data": pts[1]}


def main(argv: Optional[Iterable[str]] = None) -> int:
    import argparse
    p = argparse.ArgumentParser(description="EDGAR shares-outstanding backfill")
    p.add_argument("--backfill", action="store_true")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--db", default=DEFAULT_DB, help="FINRA DB defining the universe")
    p.add_argument("--shares-db", default=SHARES_DB, help="where shares data is written")
    args = p.parse_args(list(argv) if argv is not None else None)

    if args.backfill:
        from src.squeeze.backtest.universe import study_symbols
        syms = study_symbols(args.db)
        print(f"universe: {len(syms):,} symbols")
        print(backfill(syms, args.shares_db))
    for k, v in stats(args.shares_db).items():
        print(f"  {k:18s} {v:,}" if isinstance(v, int) else f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
