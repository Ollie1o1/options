"""Real EOD option chains from the public DoltHub dataset
`post-no-preference/options` (no auth needed for reads), cached locally.

Coverage 2019-02-09..2026-06-12, trading days only. Queries must be scoped by
`date` (leading PK column) or they hit the API's ~30s deadline. A (symbol,date)
chain is ~200 rows in <1s.

CLI:
    python -m src.dolt_options --probe
    python -m src.dolt_options --backfill --symbols AAPL,SPY --start 2024-01-01 --end 2024-12-31 --weekly
    python -m src.dolt_options --stats
"""
from __future__ import annotations

import datetime as _dtmod
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

import requests

API_BASE = "https://www.dolthub.com/api/v1alpha1/post-no-preference/options/master"
COVERAGE_MIN = "2019-02-09"
COVERAGE_MAX = "2026-06-12"
DEFAULT_CACHE = os.path.join("data", "dolt_options.db")
_THROTTLE_S = 0.30          # polite spacing between live calls


class DoltQueryError(RuntimeError):
    pass


# ── SQL client ──────────────────────────────────────────────────────────────
# HTTP statuses worth retrying with backoff: rate-limit (403/429) + transient 5xx.
_RETRY_STATUS = (403, 429, 500, 502, 503, 504)
_MAX_RETRIES = 5


class DoltRateLimited(DoltQueryError):
    """Raised when DoltHub keeps rate-limiting (403/429) after backoff retries."""


def _query(sql: str, timeout: int = 45) -> List[Dict[str, Any]]:
    """Run one SQL query against the DoltHub API. Returns rows (list of dicts).
    Retries deadline/network errors AND rate-limit/5xx HTTP statuses with
    exponential backoff. Raises DoltRateLimited if 403/429 persists, else
    DoltQueryError. Sends NO auth header (DoltHub 400s if one is present)."""
    last_exc: Optional[Exception] = None
    rate_limited = False
    for attempt in range(_MAX_RETRIES):
        try:
            resp = requests.get(API_BASE, params={"q": sql}, timeout=timeout)
            if resp.status_code == 200:
                j = resp.json()
                status = j.get("query_execution_status")
                if status == "Success":
                    return j.get("rows") or []
                msg = j.get("query_execution_message", "")
                if "deadline" in msg.lower() and attempt < _MAX_RETRIES - 1:
                    last_exc = DoltQueryError(msg)
                    time.sleep(1.0 * (attempt + 1))
                    continue
                raise DoltQueryError(msg or f"status={status}")
            if resp.status_code in _RETRY_STATUS:
                rate_limited = resp.status_code in (403, 429)
                last_exc = DoltQueryError(f"HTTP {resp.status_code}")
                # Longer backoff for rate limits than for plain 5xx.
                base = 8.0 if rate_limited else 1.5
                time.sleep(base * (attempt + 1))
                continue
            raise DoltQueryError(f"HTTP {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(1.0 * (attempt + 1))
    if rate_limited:
        raise DoltRateLimited(f"rate-limited after {_MAX_RETRIES} attempts: {last_exc}")
    raise DoltQueryError(f"query failed after {_MAX_RETRIES} attempts: {last_exc}")


# ── Normalization + coverage helpers ────────────────────────────────────────
_NUM_FIELDS = ("bid", "ask", "vol", "delta", "gamma", "theta", "vega", "rho")


def _f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _normalize(row: Dict[str, Any]) -> Dict[str, Any]:
    """DoltHub option_chain row (all strings) → typed contract dict."""
    out = {
        "date": row.get("date"),
        "symbol": (row.get("act_symbol") or "").upper(),
        "expiration": row.get("expiration"),
        "strike": _f(row.get("strike")),
        "type": (row.get("call_put") or "").strip().lower(),  # 'call'/'put'
    }
    for k in _NUM_FIELDS:
        out[k] = _f(row.get(k))
    b, a = out.get("bid"), out.get("ask")
    out["iv"] = out.pop("vol")  # rename for clarity
    out["mid"] = round((b + a) / 2, 4) if (b is not None and a is not None) else None
    return out


def _clamp_date(date: str) -> str:
    """Clamp an ISO date into the dataset's coverage window."""
    if date < COVERAGE_MIN:
        return COVERAGE_MIN
    if date > COVERAGE_MAX:
        return COVERAGE_MAX
    return date


def _dte(asof: str, expiration: str) -> int:
    a = _dtmod.date.fromisoformat(asof)
    e = _dtmod.date.fromisoformat(expiration)
    return (e - a).days


# ── Local cache ─────────────────────────────────────────────────────────────
_DDL_CHAIN = """
CREATE TABLE IF NOT EXISTS dolt_chain (
    symbol TEXT, date TEXT, expiration TEXT, strike REAL, type TEXT,
    bid REAL, ask REAL, mid REAL, iv REAL,
    delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
    PRIMARY KEY (symbol, date, expiration, strike, type)
)
"""
_DDL_FETCHED = """
CREATE TABLE IF NOT EXISTS dolt_fetched (
    symbol TEXT, date TEXT, n_rows INTEGER, fetched_at TEXT,
    PRIMARY KEY (symbol, date)
)
"""


def _ensure_cache(db_path: str) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(_DDL_CHAIN)
        conn.execute(_DDL_FETCHED)


def _cache_read(db_path: str, symbol: str, date: str):
    with sqlite3.connect(db_path) as conn:
        done = conn.execute(
            "SELECT n_rows FROM dolt_fetched WHERE symbol=? AND date=?",
            (symbol, date)).fetchone()
        if done is None:
            return None  # never fetched
        cur = conn.execute(
            "SELECT symbol,date,expiration,strike,type,bid,ask,mid,iv,"
            "delta,gamma,theta,vega,rho FROM dolt_chain WHERE symbol=? AND date=?",
            (symbol, date))
        cols = [d[0] for d in cur.description]
        return [dict(zip(cols, r)) for r in cur.fetchall()]


def _cache_write(db_path: str, symbol: str, date: str, chain) -> None:
    with sqlite3.connect(db_path) as conn:
        for c in chain:
            conn.execute(
                "INSERT OR REPLACE INTO dolt_chain "
                "(symbol,date,expiration,strike,type,bid,ask,mid,iv,delta,gamma,theta,vega,rho) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (symbol, date, c["expiration"], c["strike"], c["type"], c["bid"], c["ask"],
                 c["mid"], c["iv"], c["delta"], c["gamma"], c["theta"], c["vega"], c["rho"]))
        conn.execute(
            "INSERT OR REPLACE INTO dolt_fetched (symbol,date,n_rows,fetched_at) VALUES (?,?,?,?)",
            (symbol, date, len(chain),
             _dtmod.datetime.now().isoformat(timespec="seconds")))


def get_chain(symbol: str, date: str, db_path: str = DEFAULT_CACHE) -> List[Dict[str, Any]]:
    """Real option chain for (symbol, date). Cache-first; one live API call per
    new (symbol,date), empty days cached as misses. Returns [] for non-trading days."""
    symbol = symbol.upper()
    date = _clamp_date(date)
    _ensure_cache(db_path)
    cached = _cache_read(db_path, symbol, date)
    if cached is not None:
        return cached
    rows = _query(
        "SELECT date,act_symbol,expiration,strike,call_put,bid,ask,vol,"
        "delta,gamma,theta,vega,rho FROM option_chain "
        f"WHERE act_symbol='{symbol}' AND date='{date}'")
    chain = [_normalize(r) for r in rows]
    _cache_write(db_path, symbol, date, chain)
    time.sleep(_THROTTLE_S)
    return chain


def get_chain_near(symbol: str, date: str, max_skip: int = 4,
                   db_path: str = DEFAULT_CACHE, direction: int = 1):
    """Return (actual_date, chain) for the nearest date that HAS data, searching
    up to `max_skip` days out, preferring `direction` (+1 forward / -1 back).
    Handles gaps in the DoltHub dataset (some trading days are missing). Returns
    (None, []) if nothing found in the window."""
    symbol = symbol.upper()
    base = _dtmod.date.fromisoformat(_clamp_date(date))
    offsets = [0]
    for k in range(1, max_skip + 1):
        offsets += [direction * k, -direction * k]
    for off in offsets:
        d = (base + _dtmod.timedelta(days=off)).isoformat()
        if d < COVERAGE_MIN or d > COVERAGE_MAX:
            continue
        ch = get_chain(symbol, d, db_path=db_path)
        if ch:
            return d, ch
    return None, []


# ── Contract selection ──────────────────────────────────────────────────────
def nearest_contract(chain: List[Dict[str, Any]], opt_type: str, target_strike: float,
                     asof: str, target_dte: int = 30,
                     min_dte: int = 7) -> Optional[Dict[str, Any]]:
    """From a chain, pick the contract of `opt_type` minimizing a combined
    distance in (strike, DTE) from the targets. Skips expiries under min_dte."""
    opt_type = opt_type.lower()
    best, best_cost = None, float("inf")
    for c in chain:
        if c.get("type") != opt_type or c.get("strike") is None:
            continue
        d = _dte(asof, c["expiration"])
        if d < min_dte:
            continue
        s_err = abs(c["strike"] - target_strike) / max(target_strike, 1e-6)
        d_err = abs(d - target_dte) / 30.0
        cost = s_err + d_err
        if cost < best_cost:
            best, best_cost = c, cost
    return best


# ── Backfill + stats + CLI ──────────────────────────────────────────────────
def _already_fetched(db_path: str, symbol: str, date: str) -> bool:
    _ensure_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        return conn.execute("SELECT 1 FROM dolt_fetched WHERE symbol=? AND date=?",
                            (symbol, date)).fetchone() is not None


def backfill(symbols, dates, db_path: str = DEFAULT_CACHE, verbose: bool = False) -> int:
    """Populate the cache for every (symbol, date). Resumable: skips pairs already
    fetched. Returns the number of NEW (symbol,date) pairs fetched."""
    fetched = 0
    for date in dates:
        date = _clamp_date(date)
        for symbol in symbols:
            symbol = symbol.upper()
            if _already_fetched(db_path, symbol, date):
                continue
            chain = get_chain(symbol, date, db_path=db_path)  # writes cache + throttles
            fetched += 1
            if verbose:
                print(f"  {symbol} {date}: {len(chain)} contracts")
    return fetched


def stats(db_path: str = DEFAULT_CACHE) -> Dict[str, Any]:
    _ensure_cache(db_path)
    with sqlite3.connect(db_path) as conn:
        pairs = conn.execute("SELECT COUNT(*) FROM dolt_fetched").fetchone()[0]
        rows = conn.execute("SELECT COUNT(*) FROM dolt_chain").fetchone()[0]
        syms = conn.execute("SELECT COUNT(DISTINCT symbol) FROM dolt_fetched").fetchone()[0]
        rng = conn.execute("SELECT MIN(date), MAX(date) FROM dolt_fetched").fetchone()
    return {"symbol_days": pairs, "rows": rows, "symbols": syms,
            "first": rng[0], "last": rng[1]}


def _config_basket():
    import json
    try:
        with open("config.json") as f:
            return (json.load(f).get("dolt_options") or {}).get("basket") or ["AAPL", "SPY"]
    except Exception:
        return ["AAPL", "SPY"]


def _date_range(start: str, end: str, weekly: bool = False):
    """Calendar dates in [start, end] (coverage-clamped). When weekly, only
    Fridays. Steps day-by-day so a non-Friday start still yields all Fridays."""
    start = _clamp_date(start or COVERAGE_MIN)
    end = _clamp_date(end or COVERAGE_MAX)
    s = _dtmod.date.fromisoformat(start)
    e = _dtmod.date.fromisoformat(end)
    out, d = [], s
    while d <= e:
        if not weekly or d.weekday() == 4:  # 4 = Friday
            out.append(d.isoformat())
        d += _dtmod.timedelta(days=1)
    return out


def _cli():
    import argparse
    import json
    ap = argparse.ArgumentParser(description="DoltHub real option chains (cached)")
    ap.add_argument("--probe", action="store_true", help="Fetch one AAPL chain and print a sample")
    ap.add_argument("--backfill", action="store_true", help="Backfill basket x dates")
    ap.add_argument("--symbols", default="", help="Comma list (default: config basket)")
    ap.add_argument("--start")
    ap.add_argument("--end")
    ap.add_argument("--weekly", action="store_true", help="Sample Fridays in [start,end]")
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--db", default=DEFAULT_CACHE)
    args = ap.parse_args()

    if args.probe:
        chain = get_chain("AAPL", COVERAGE_MAX, db_path=args.db)
        print(f"AAPL {COVERAGE_MAX}: {len(chain)} contracts")
        for c in chain[:3]:
            print("  ", {k: c[k] for k in ("type", "strike", "expiration", "bid", "ask", "iv", "delta")})
        return
    if args.stats:
        print(json.dumps(stats(db_path=args.db), indent=1))
        return
    if args.backfill:
        syms = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] or _config_basket()
        dates = _date_range(args.start, args.end, weekly=args.weekly)
        print(f"Backfilling {len(syms)} symbols x {len(dates)} dates ...")
        n = backfill(syms, dates, db_path=args.db, verbose=True)
        print(f"Done: {n} new symbol-days. {json.dumps(stats(db_path=args.db))}")


if __name__ == "__main__":
    _cli()
