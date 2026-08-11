"""Load optionsDX EOD option-chain CSVs into a local cache.

WHY THIS EXISTS
---------------
The Dolt cache is 12.1M rows across 121 symbols and 2020-2026, and it has two
hard walls: **DTE 10-67** and ~2.6 expirations per symbol-day. Those walls are
not a specification problem, they are missing data. `docs/HOLDOUT_20260809.md`
records the cost:

    term_slope   in-sample +0.0431   holdout -0.0568   FLIPS

...on a slope measured 10d against 60d, when the literature result is 1M/3M.
The doc's own verdict: "That is a data wall, not a fixable specification —
nothing on disk can test the real quantity. Revisit only if optionsDX lands,
which carries all expirations."

optionsDX is free (registration, no billing) and carries **all strikes and all
expirations** for SPY/SPX/QQQ/VIX/AAPL/NVDA/TSLA, 2010-2023, with `C_VOLUME`
and `C_SIZE`. It makes three things testable for the first time: a real term
slope, a depth-based liquidity filter, and anything longer-dated than 67 days.

WHY A SEPARATE TABLE
--------------------
`odx_chain` is deliberately NOT `dolt_chain`. The two sources have different
DTE coverage, so a query spanning both would silently compare a 10-67 window
before 2020 against an all-expirations window after it, and any feature whose
value depends on tenor would pick that up as signal. Provenance stays explicit
and joins across sources have to be written on purpose.

The file format is the standard optionsDX EOD chain export: one row per
(quote date, expiry, strike) carrying BOTH the call and the put, with
bracketed, space-padded headers. Each input row becomes two output rows.
"""
from __future__ import annotations

import csv
import glob
import os
import re
import sqlite3
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

DEFAULT_CACHE = os.path.join("data", "optionsdx.db")

# Columns shared with `dolt_chain` so downstream code can read either with the
# same field names, plus the four optionsDX adds.
_DDL_CHAIN = """
CREATE TABLE IF NOT EXISTS odx_chain (
    symbol TEXT, date TEXT, expiration TEXT, strike REAL, type TEXT,
    bid REAL, ask REAL, mid REAL, iv REAL,
    delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
    volume REAL, bid_size REAL, ask_size REAL,
    underlying REAL, dte REAL,
    PRIMARY KEY (symbol, date, expiration, strike, type)
)
"""

# Coverage log, mirroring `dolt_fetched`. A file is recorded only after its rows
# are committed, so an interrupted run resumes at the file boundary rather than
# re-reading everything or, worse, skipping a partially-loaded file.
_DDL_LOADED = """
CREATE TABLE IF NOT EXISTS odx_loaded (
    source_file TEXT PRIMARY KEY,
    symbol TEXT, min_date TEXT, max_date TEXT,
    n_rows INTEGER, loaded_at TEXT
)
"""

_DDL_INDEX = (
    "CREATE INDEX IF NOT EXISTS idx_odx_sym_date ON odx_chain(symbol, date)",
    "CREATE INDEX IF NOT EXISTS idx_odx_dte ON odx_chain(symbol, date, dte)",
)


def _norm_header(name: str) -> str:
    """`[C_BID]` / ` [C_BID] ` / `c_bid` -> `c_bid`."""
    return re.sub(r"[\[\]\s]", "", str(name)).lower()


def _num(value: Any) -> Optional[float]:
    """A finite float, or None. optionsDX writes blanks and stray spaces."""
    if value is None:
        return None
    s = str(value).strip()
    if not s or s in {"", "-", "NA", "N/A", "nan"}:
        return None
    try:
        f = float(s)
    except ValueError:
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _sizes(value: Any) -> Tuple[Optional[float], Optional[float]]:
    """`C_SIZE`/`P_SIZE` is quoted depth as "12 x 30" — bid size by ask size.

    Returned as two numbers because a single string is useless to a filter.
    This is the column that makes H2 (does quoted depth predict realized
    friction?) testable at all, so a silent mis-parse here would quietly
    answer a question with noise.
    """
    if value is None:
        return None, None
    s = str(value).strip()
    if not s:
        return None, None
    parts = re.split(r"\s*[xX]\s*", s)
    if len(parts) != 2:
        return None, None
    return _num(parts[0]), _num(parts[1])


@dataclass
class LoadReport:
    """What a load actually did. Counts are rows written, not rows read."""
    files: int = 0
    rows_read: int = 0
    rows_written: int = 0
    skipped_files: List[str] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)

    def note(self, reason: str) -> None:
        self.rejected[reason] = self.rejected.get(reason, 0) + 1

    def summary(self) -> str:
        out = [f"{self.files} file(s), {self.rows_read:,} read, "
               f"{self.rows_written:,} written"]
        if self.skipped_files:
            out.append(f"{len(self.skipped_files)} already loaded")
        for reason, n in sorted(self.rejected.items(), key=lambda kv: -kv[1]):
            out.append(f"  rejected {n:,}: {reason}")
        return "\n".join(out)


REQUIRED_COLUMNS = ("quote_date", "expire_date", "strike", "underlying_last", "dte")


def parse_rows(handle: Iterable[str], symbol: str,
               report: Optional[LoadReport] = None) -> Iterator[Dict[str, Any]]:
    """Yield one dict per (quote, expiry, strike, side) from an optionsDX CSV.

    Each input row carries a call and a put; both are emitted. A row missing
    the identifying columns is rejected and counted rather than guessed at —
    a chain row with no strike is not a contract.
    """
    rep = report if report is not None else LoadReport()
    reader = csv.reader(handle)
    try:
        raw_header = next(reader)
    except StopIteration:
        rep.note("empty file")
        return
    header = [_norm_header(h) for h in raw_header]
    idx = {name: i for i, name in enumerate(header)}

    missing = [c for c in REQUIRED_COLUMNS if c not in idx]
    if missing:
        raise ValueError(
            f"not an optionsDX chain export — missing {missing}. "
            f"Saw columns: {header[:8]}...")

    def cell(row: List[str], name: str) -> Any:
        i = idx.get(name)
        return row[i] if (i is not None and i < len(row)) else None

    for row in reader:
        if not row or all(not str(c).strip() for c in row):
            continue
        rep.rows_read += 1

        date = str(cell(row, "quote_date") or "").strip()
        expiration = str(cell(row, "expire_date") or "").strip()
        strike = _num(cell(row, "strike"))
        if not date or not expiration or strike is None:
            rep.note("missing date/expiry/strike")
            continue

        underlying = _num(cell(row, "underlying_last"))
        dte = _num(cell(row, "dte"))

        for side, prefix in (("call", "c"), ("put", "p")):
            bid = _num(cell(row, f"{prefix}_bid"))
            ask = _num(cell(row, f"{prefix}_ask"))
            if bid is None and ask is None:
                rep.note(f"{side}: no quote")
                continue
            bid_size, ask_size = _sizes(cell(row, f"{prefix}_size"))
            mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
            yield {
                "symbol": symbol.upper(),
                "date": date,
                "expiration": expiration,
                "strike": strike,
                "type": side,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "iv": _num(cell(row, f"{prefix}_iv")),
                "delta": _num(cell(row, f"{prefix}_delta")),
                "gamma": _num(cell(row, f"{prefix}_gamma")),
                "theta": _num(cell(row, f"{prefix}_theta")),
                "vega": _num(cell(row, f"{prefix}_vega")),
                "rho": _num(cell(row, f"{prefix}_rho")),
                "volume": _num(cell(row, f"{prefix}_volume")),
                "bid_size": bid_size,
                "ask_size": ask_size,
                "underlying": underlying,
                "dte": dte,
            }


def ensure_cache(db_path: str = DEFAULT_CACHE) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL_CHAIN)
        conn.execute(_DDL_LOADED)
        for ddl in _DDL_INDEX:
            conn.execute(ddl)


def symbol_from_filename(path: str) -> Optional[str]:
    """optionsDX names files like `spy_eod_202301.txt` / `SPY_2023_01.csv`.

    Returns None rather than guessing when the leading token is not a plausible
    ticker — loading a whole month under the wrong symbol is worse than
    refusing to load it.
    """
    base = os.path.basename(path)
    token = re.split(r"[^A-Za-z]", base, maxsplit=1)[0]
    return token.upper() if 1 <= len(token) <= 6 else None


def load_file(path: str, db_path: str = DEFAULT_CACHE,
              symbol: Optional[str] = None,
              report: Optional[LoadReport] = None) -> LoadReport:
    """Load one CSV. Idempotent: a file already in `odx_loaded` is skipped.

    Rows are written in a single transaction per file, and the coverage row is
    written inside it, so the cache never records a file it did not finish.
    """
    rep = report if report is not None else LoadReport()
    ensure_cache(db_path)
    key = os.path.basename(path)

    with sqlite3.connect(db_path) as conn:
        seen = conn.execute(
            "SELECT 1 FROM odx_loaded WHERE source_file=?", (key,)).fetchone()
    if seen:
        rep.skipped_files.append(key)
        return rep

    sym = symbol or symbol_from_filename(path)
    if not sym:
        raise ValueError(f"cannot infer symbol from {key!r}; pass symbol=")

    with open(path, "r", newline="", encoding="utf-8-sig") as fh:
        rows = list(parse_rows(fh, sym, rep))

    if not rows:
        rep.note("no usable rows")
        return rep

    dates = [r["date"] for r in rows]
    with sqlite3.connect(db_path) as conn:
        conn.executemany(
            "INSERT OR REPLACE INTO odx_chain (symbol,date,expiration,strike,type,"
            "bid,ask,mid,iv,delta,gamma,theta,vega,rho,volume,bid_size,ask_size,"
            "underlying,dte) VALUES (:symbol,:date,:expiration,:strike,:type,"
            ":bid,:ask,:mid,:iv,:delta,:gamma,:theta,:vega,:rho,:volume,"
            ":bid_size,:ask_size,:underlying,:dte)", rows)
        conn.execute(
            "INSERT OR REPLACE INTO odx_loaded (source_file,symbol,min_date,"
            "max_date,n_rows,loaded_at) VALUES (?,?,?,?,?,datetime('now'))",
            (key, sym, min(dates), max(dates), len(rows)))

    rep.files += 1
    rep.rows_written += len(rows)
    return rep


def load_directory(pattern: str, db_path: str = DEFAULT_CACHE,
                   symbol: Optional[str] = None) -> LoadReport:
    """Load every file matching a glob, in name order. Resumable by design."""
    rep = LoadReport()
    for path in sorted(glob.glob(pattern)):
        load_file(path, db_path=db_path, symbol=symbol, report=rep)
    return rep


def coverage(db_path: str = DEFAULT_CACHE) -> List[Dict[str, Any]]:
    """Per-symbol coverage, including the DTE reach that motivated the load.

    `max_dte` is the column to read: the whole point of this source is that it
    exceeds the Dolt cache's 67-day ceiling. A load that comes back at 67 has
    silently reproduced the wall it was meant to break.
    """
    ensure_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        cur = conn.execute(
            "SELECT symbol, COUNT(*) n, MIN(date) d0, MAX(date) d1, "
            "COUNT(DISTINCT date) days, COUNT(DISTINCT expiration) expiries, "
            "MIN(dte) min_dte, MAX(dte) max_dte "
            "FROM odx_chain GROUP BY symbol ORDER BY symbol")
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def expiries_per_day(db_path: str = DEFAULT_CACHE,
                     symbol: Optional[str] = None) -> Optional[float]:
    """Mean distinct expirations per symbol-day — the Dolt cache runs ~2.6.

    The direct measure of whether the second wall is gone.
    """
    ensure_cache(db_path)
    sql = ("SELECT AVG(k) FROM (SELECT COUNT(DISTINCT expiration) k "
           "FROM odx_chain {} GROUP BY symbol, date)")
    where, args = ("WHERE symbol=?", (symbol.upper(),)) if symbol else ("", ())
    with sqlite3.connect(db_path) as conn:
        row = conn.execute(sql.format(where), args).fetchone()
    return float(row[0]) if row and row[0] is not None else None


# ── feeding the allocation engine ───────────────────────────────────────────
class OdxChainSource:
    """`odx_chain` behind the engine's `ChainSource` protocol.

    Deliberately here rather than in `src/alloc/engine.py`. The engine asks
    only for an object with `.chain(symbol, date)`, so putting the optionsDX
    schema knowledge in the module that owns the schema means the engine gains
    no import of, or opinion about, this source.

    The reason to reuse the engine at all rather than write a fresh harness:
    every earlier result in this repo came out of `replay`, and a second
    harness would differ from it in ways nobody could see. A new hypothesis
    measured a new way is not comparable to the record it is meant to correct.

    Carries `volume`/`bid_size`/`ask_size` on top of the Dolt column set. Those
    three exist in no other source here and H2 is built on them.
    """

    _COLS = ("symbol", "date", "expiration", "strike", "type", "bid", "ask",
             "mid", "iv", "delta", "gamma", "theta", "vega", "rho",
             "volume", "bid_size", "ask_size", "underlying", "dte")

    def __init__(self, db_path: str = DEFAULT_CACHE, cache_dates: int = 8):
        self.db_path = db_path
        # BOUNDED, unlike the Dolt source's cache. That one holds every chain
        # for the whole replay, which is affordable at 117 symbols x ~230
        # dates. This source replays one symbol across 3,500 dates averaging
        # 5,400 contracts each, so keeping them all would mean ~19M live dicts.
        # The engine only ever reads the date it is standing on, so a handful
        # is enough; `OrderedDict` gives the eviction order for free.
        self._cache: "OrderedDict[Tuple[str, str], List[Dict[str, Any]]]" = OrderedDict()
        self._cache_dates = max(1, int(cache_dates))
        self._conn: Optional[Any] = None

    def _connection(self) -> Any:
        # One connection for the whole replay: a full window is thousands of
        # chain reads and reopening per read is both slow and needlessly
        # contended against a concurrent load.
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, timeout=30.0)
        return self._conn

    def chain(self, symbol: str, date: str) -> List[Dict[str, Any]]:
        key = (symbol, date)
        if key not in self._cache:
            cur = self._connection().execute(
                f"SELECT {','.join(self._COLS)} FROM odx_chain "
                "WHERE symbol=? AND date=?", (symbol.upper(), date))
            self._cache[key] = [dict(zip(self._COLS, r)) for r in cur.fetchall()]
            while len(self._cache) > self._cache_dates:
                self._cache.popitem(last=False)     # oldest out
        else:
            self._cache.move_to_end(key)
        return self._cache[key]

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def available_dates(db_path: str, symbol: str,
                    start: Optional[str] = None,
                    end: Optional[str] = None) -> List[str]:
    """Every quote date this symbol actually has, ascending.

    Read from the data rather than generated from a calendar. A date the
    source does not cover is MISSING, and scoring it as "no opportunity that
    day" is what quietly biases an always-on benchmark upward — the same trap
    `universe.usable_dates` exists to avoid on the Dolt side.
    """
    ensure_cache(db_path)
    sql = "SELECT DISTINCT date FROM odx_chain WHERE symbol=?"
    args: List[Any] = [symbol.upper()]
    if start:
        sql += " AND date>=?"
        args.append(start)
    if end:
        sql += " AND date<=?"
        args.append(end)
    with sqlite3.connect(db_path) as conn:
        return [r[0] for r in conn.execute(sql + " ORDER BY date", args)]


def terminal_date(db_path: str, symbol: str) -> Optional[str]:
    """Last date with data, so the engine can close rather than drop.

    A position still open past this point cannot be marked. Dropping it would
    delete exactly the trades that ran into the end of the window.
    """
    ensure_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        row = conn.execute("SELECT MAX(date) FROM odx_chain WHERE symbol=?",
                           (symbol.upper(),)).fetchone()
    return row[0] if row and row[0] else None


# ── CLI ─────────────────────────────────────────────────────────────────────
def _print_coverage(db_path: str) -> None:
    rows = coverage(db_path)
    if not rows:
        print(f"  {db_path}: empty — nothing loaded yet")
        return
    print(f"  {'symbol':<8}{'rows':>12}{'days':>7}{'expiries':>10}"
          f"{'DTE min':>9}{'DTE max':>9}  window")
    print("  " + "-" * 78)
    for r in rows:
        print(f"  {r['symbol']:<8}{r['n']:>12,}{r['days']:>7}"
              f"{r['expiries']:>10}{(r['min_dte'] or 0):>9.0f}"
              f"{(r['max_dte'] or 0):>9.0f}  {r['d0']}..{r['d1']}")
    per_day = expiries_per_day(db_path)
    print()
    # The two numbers this whole loader exists to move.
    worst = max((r["max_dte"] or 0) for r in rows)
    print(f"  mean expiries per symbol-day: {per_day:.1f}"
          f"   (Dolt cache: ~2.6)" if per_day else "")
    print(f"  deepest DTE loaded:           {worst:.0f}"
          f"   (Dolt cache ceiling: 67)")
    if worst <= 67:
        print("\n  WARNING: nothing past 67 DTE — this load has reproduced the"
              "\n  wall it was meant to break. Check you downloaded the full"
              "\n  chain export rather than a near-dated sample.")


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m src.optionsdx",
        description="Load optionsDX EOD chain CSVs into a local cache.")
    p.add_argument("--load", metavar="GLOB",
                   help="glob of CSV/TXT files, e.g. 'data/optionsdx_raw/*.txt'")
    p.add_argument("--symbol", default=None,
                   help="override the symbol inferred from each filename")
    p.add_argument("--db", default=DEFAULT_CACHE, help=f"cache (default {DEFAULT_CACHE})")
    p.add_argument("--coverage", action="store_true",
                   help="report what is loaded and whether the DTE wall is gone")
    args = p.parse_args(argv)

    if not args.load and not args.coverage:
        p.print_help()
        return 1

    if args.load:
        rep = load_directory(args.load, db_path=args.db, symbol=args.symbol)
        print(rep.summary())
        if not rep.files and not rep.skipped_files:
            print(f"  no files matched {args.load!r}")
            return 1
        print()

    _print_coverage(args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
