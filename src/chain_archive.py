"""Daily option-chain snapshots — the free historical dataset that compounds.

Real per-contract option history is the single biggest data gap for honest
backtesting (the lottery study had to model premiums with an IV proxy; paid
feeds run $29-90/mo). This module takes the other path: archive the *current*
chain once per trading day from the free CBOE feed, and in a few months the
backtests have real bid/ask/IV/Greeks history for the symbols we actually
trade — cost: $0 and patience.

Storage: ``data/chain_archive.db`` table ``chain_snapshots``, one row per
contract per day, filtered to tradeable territory (DTE ≤ max, moneyness within
a band) to keep growth sane (~10-15k rows/day for ~15 liquid symbols ≈
a few MB/month). Runs inside startup maintenance (parent only), throttled to
once per weekday afternoon — same self-healing model as the other jobs.

Config (config.json → data_archive): enabled, symbols, max_dte, moneyness_band.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Sequence
from src.paths import repo_path

DEFAULT_DB = repo_path(os.path.join("data", "chain_archive.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS chain_snapshots (
    symbol TEXT NOT NULL,
    snap_date TEXT NOT NULL,
    contract TEXT NOT NULL,
    type TEXT, strike REAL, expiration TEXT,
    bid REAL, ask REAL, bid_size REAL, ask_size REAL,
    iv REAL, delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
    open_interest REAL, volume REAL,
    last_trade_time TEXT, spot REAL, snapshot_ts TEXT, source TEXT,
    PRIMARY KEY (contract, snap_date)
);
CREATE INDEX IF NOT EXISTS idx_snap_symbol_date
    ON chain_snapshots (symbol, snap_date);
"""

_COLUMNS = ["symbol", "snap_date", "contract", "type", "strike", "expiration",
            "bid", "ask", "bid_size", "ask_size", "iv", "delta", "gamma",
            "theta", "vega", "rho", "open_interest", "volume",
            "last_trade_time", "spot", "snapshot_ts", "source"]


def ensure_db(db_path: str = DEFAULT_DB) -> None:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.executescript(_SCHEMA)


def filter_rows(rows: List[Dict[str, Any]], today: str,
                max_dte: int = 120, moneyness_band: float = 0.15,
                min_open_interest: float = 1) -> List[Dict[str, Any]]:
    """Keep contracts a swing trader could actually touch: 0 < DTE ≤ max_dte,
    strike within ±moneyness_band of spot, and some open interest (zero-OI far
    wings are junk and triple the storage). Pure.

    ±15% covers every realistic swing strike at these DTEs — even a 2.5σ
    lottery wing at 14d/20%vol sits ~12% OTM."""
    t0 = datetime.strptime(today, "%Y-%m-%d")
    kept = []
    for r in rows:
        try:
            dte = (datetime.strptime(str(r["expiration"])[:10], "%Y-%m-%d") - t0).days
            spot, strike = float(r.get("spot") or 0), float(r.get("strike") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if dte <= 0 or dte > max_dte:
            continue
        if spot <= 0 or strike <= 0 or abs(strike / spot - 1.0) > moneyness_band:
            continue
        if float(r.get("open_interest") or 0) < min_open_interest:
            continue
        kept.append(r)
    return kept


def symbols_for_run(config: Dict[str, Any],
                    extra: Optional[Sequence[str]] = None) -> List[str]:
    """Symbols to archive on this run: the configured basket plus *extra*.

    The configured list is a fixed set of liquid names chosen for backtest
    coverage. The squeeze cohort changes every FINRA cycle and cannot be
    enumerated in advance, so it is passed in per run rather than written to
    config. Configured symbols keep their order and win on collision, so
    turning the sleeve on can never quietly reorder or drop the standing basket.
    """
    out: List[str] = []
    seen = set()
    configured = (config.get("data_archive") or {}).get("symbols") or []
    for group in (configured, extra or []):
        for raw in group:
            sym = str(raw or "").strip().upper()
            if sym and sym not in seen:
                seen.add(sym)
                out.append(sym)
    return out


def archive_symbols(symbols: List[str], db_path: str = DEFAULT_DB,
                    today: Optional[str] = None,
                    fetcher: Optional[Callable] = None,
                    max_dte: int = 120, moneyness_band: float = 0.15,
                    min_open_interest: float = 1) -> int:
    """Snapshot each symbol once for `today`. Idempotent per (symbol, day);
    per-symbol failures are isolated. Returns rows written."""
    from src import cboe_client
    today = today or datetime.now().strftime("%Y-%m-%d")
    fetcher = fetcher or cboe_client.fetch_chain
    ensure_db(db_path)
    written = 0
    with sqlite3.connect(db_path) as conn:
        for sym in symbols:
            sym = sym.upper()
            try:
                done = conn.execute(
                    "SELECT 1 FROM chain_snapshots WHERE symbol=? AND snap_date=? LIMIT 1",
                    (sym, today)).fetchone()
                if done:
                    continue
                rows = filter_rows(fetcher(sym) or [], today,
                                   max_dte=max_dte, moneyness_band=moneyness_band,
                                   min_open_interest=min_open_interest)
                for r in rows:
                    values = [sym, today] + [r.get(c) for c in _COLUMNS[2:]]
                    conn.execute(
                        f"INSERT OR IGNORE INTO chain_snapshots ({','.join(_COLUMNS)}) "
                        f"VALUES ({','.join('?' * len(_COLUMNS))})", values)
                    written += 1
                conn.commit()
            except Exception:
                continue
    return written


def archive_universe(symbols: Sequence[str], ledger_path: Optional[str],
                     lookback_days: int = 30,
                     today: Optional[str] = None) -> List[str]:
    """The configured symbols plus every ticker the book has recently touched.

    The archive is the only independent record of the prices the ledger
    exits at. It covered 15 fixed symbols while the sized-era book traded 39,
    so 26 names — ABT, AVGO, JPM, MU, UNH among them — could never be
    reconciled. A ticker joins while a position on it is open, or when it was
    entered or exited inside `lookback_days`, so its exit day is archived too.

    Configured symbols come first, in their order, so a run that dies partway
    has already covered the core list. The ledger is opened read-only, and a
    missing or unreadable ledger leaves the configured list unchanged.
    """
    from datetime import timedelta
    out: List[str] = []
    for s in symbols:
        s = str(s).upper()
        if s and s not in out:
            out.append(s)
    if not ledger_path or not lookback_days or not os.path.exists(ledger_path):
        return out
    day = datetime.strptime(today, "%Y-%m-%d") if today else datetime.now()
    cutoff = (day - timedelta(days=int(lookback_days))).strftime("%Y-%m-%d")
    try:
        with sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True) as conn:
            rows = conn.execute(
                "SELECT DISTINCT UPPER(ticker) FROM trades "
                "WHERE ticker IS NOT NULL AND (status = 'OPEN' "
                "   OR date(date) >= ? OR date(exit_date) >= ?) "
                "ORDER BY 1", (cutoff, cutoff)).fetchall()
    except sqlite3.Error:
        return out
    for (t,) in rows:
        # Crypto pairs are not on the CBOE equity-options feed.
        if t and "-" not in t and t not in out:
            out.append(t)
    return out


def health_lines(db_path: str = DEFAULT_DB, days: int = 14,
                 today: Optional[str] = None) -> List[str]:
    """Trading days in the last `days` with no snapshot at all.

    The scheduled-jobs line asks only how old the NEWEST snapshot is, so a day
    lost in the middle is invisible to it: 09-22, 09-23, 09-24, 10-06 and
    10-09 2026 — five of 22 sessions — went missing behind an OK. A snapshot
    cannot be taken retroactively, so this is the only place a lost day shows.

    WARN, not CRITICAL: nothing live reads the archive. Today is excluded —
    the run is not due until the afternoon. Days with no rows at all are the
    unit; one symbol failing on an otherwise good day is ordinary.
    """
    from datetime import timedelta
    from .candidate_marks import missing_trading_days
    today = today or datetime.now().strftime("%Y-%m-%d")
    since = (datetime.strptime(today, "%Y-%m-%d")
             - timedelta(days=days)).strftime("%Y-%m-%d")
    if not os.path.exists(db_path):
        return []
    try:
        with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
            per_day = conn.execute(
                "SELECT snap_date, COUNT(DISTINCT symbol) FROM chain_snapshots "
                "WHERE snap_date >= ? GROUP BY snap_date", (since,)).fetchall()
    except sqlite3.Error:
        return ["  archive gaps   unreadable                        [WARN]"]
    gaps = missing_trading_days([d for d, _n in per_day], today)
    syms = max((n for _d, n in per_day), default=0)
    sev = "WARN" if gaps else "OK"
    out = [f"  {'archive gaps':<14} {len(per_day)} days / up to {syms} symbols "
           f"in {days}d{'':<6}[{sev}]"]
    if gaps:
        out.append(f"     NO CHAIN SNAPSHOT ON {len(gaps)} TRADING DAY(S) — "
                   f"{', '.join(gaps)} — lost for good")
    return out


def due_chain_archive(state: dict, today: str, weekday: int, hhmm: int) -> bool:
    """Once per weekday, in the afternoon (≥14:00 — close enough to EOD for a
    delayed feed). weekday: 1=Mon..7=Sun."""
    if weekday > 5 or hhmm < 1400:
        return False
    # A weekday holiday is not a session. The archive ran on Labor Day 2026
    # and stored closed-market quotes under that date.
    from .data_quality import is_trading_day
    if not is_trading_day(today):
        return False
    return (state or {}).get("last_chain_archive") != today


def archive_stats(db_path: str = DEFAULT_DB) -> Dict[str, Any]:
    """Quick inventory for display/CLI."""
    if not os.path.exists(db_path):
        return {"rows": 0, "symbols": 0, "days": 0, "first": None, "last": None}
    with sqlite3.connect(db_path) as conn:
        rows, symbols, days, first, last = conn.execute(
            "SELECT COUNT(*), COUNT(DISTINCT symbol), COUNT(DISTINCT snap_date), "
            "MIN(snap_date), MAX(snap_date) FROM chain_snapshots").fetchone()
    return {"rows": rows, "symbols": symbols, "days": days,
            "first": first, "last": last}


if __name__ == "__main__":
    import json
    print(json.dumps(archive_stats(), indent=1))
