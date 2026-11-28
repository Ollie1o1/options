"""Cost-guarded institutional Databento OPRA historical client (`src/databento_opra.py`).

Uses Databento's Historical HTTP API directly via `requests` (HTTP Basic Auth with
`DATABENTO_API_KEY:` — zero extra pip dependencies).

Safety invariants:
  * Every download MUST estimate cost first via `/v0/metadata.get_cost`.
  * Refuses to spend if estimated cost exceeds `max_usd` (default $2.00 per call)
    or if `dry_run=True` (CLI default is `--estimate`).
  * Logs every actual billed pull into `data/databento_opra.db::opra_spend_log`
    so cumulative spend against the $125 signup credit is always transparent.

CLI:
    python -m src.databento_opra --estimate --symbols SPY QQQ IWM --dates 2025-11-03 2025-11-04
    python -m src.databento_opra --download --symbols SPY --dates 2025-11-03 --max-usd 0.50
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import math
import os
import sqlite3
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.cboe_client import parse_occ_symbol
from src.paths import repo_path

HIST_BASE = "https://hist.databento.com/v0"
DEFAULT_DATASET = "OPRA.PILLAR"
DEFAULT_SCHEMA = "cbbo-1m"
DEFAULT_DB = "data/databento_opra.db"

_DDL = """
CREATE TABLE IF NOT EXISTS opra_chain (
    symbol TEXT NOT NULL,
    date TEXT NOT NULL,
    clock_et TEXT NOT NULL DEFAULT '',
    ts_event TEXT NOT NULL,
    contract TEXT NOT NULL,
    expiration TEXT NOT NULL,
    strike REAL NOT NULL,
    type TEXT NOT NULL,
    bid REAL,
    ask REAL,
    mid REAL,
    bid_size REAL,
    ask_size REAL,
    dataset TEXT NOT NULL DEFAULT 'OPRA.PILLAR',
    schema TEXT NOT NULL DEFAULT 'cbbo-1m',
    PRIMARY KEY (contract, date, clock_et)
);
CREATE INDEX IF NOT EXISTS idx_opra_sym_date ON opra_chain (symbol, date, clock_et);
CREATE TABLE IF NOT EXISTS opra_spend_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    logged_at TEXT NOT NULL,
    dataset TEXT NOT NULL,
    schema TEXT NOT NULL,
    symbols TEXT NOT NULL,
    start_iso TEXT NOT NULL,
    end_iso TEXT NOT NULL,
    est_usd REAL NOT NULL,
    rows_written INTEGER NOT NULL DEFAULT 0
);
"""


def load_api_key(env_path: Optional[Path] = None) -> str:
    """Load `DATABENTO_API_KEY` from environment or `.env`."""
    key = (os.environ.get("DATABENTO_API_KEY") or "").strip()
    if key:
        return key
    path = env_path or Path(repo_path(".env"))
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s or s.startswith("#") or "=" not in s:
                continue
            k, v = s.split("=", 1)
            if k.strip() == "DATABENTO_API_KEY":
                val = v.strip().strip('"').strip("'")
                if val:
                    return val
    raise RuntimeError("DATABENTO_API_KEY not set (environment or .env)")


def ensure_db(db_path: str = DEFAULT_DB) -> str:
    full = repo_path(db_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    con = sqlite3.connect(full)
    try:
        con.executescript(_DDL)
        con.commit()
    finally:
        con.close()
    return full


def parent_symbol(ticker: str) -> str:
    """Convert underlying ticker to Databento OPRA parent symbology (`SPY` -> `SPY.OPT`)."""
    t = str(ticker or "").upper().strip()
    return t if t.endswith(".OPT") else f"{t}.OPT"


def snapshot_window_utc(trade_date: str, *, hhmm_et: str = "15:45", minutes: int = 1) -> Tuple[str, str]:
    """Return `(start_iso, end_iso)` UTC window for an ET intraday snapshot (DST-aware)."""
    from zoneinfo import ZoneInfo

    hh, mm = (int(x) for x in hhmm_et.split(":"))
    d = dt.date.fromisoformat(trade_date[:10])
    et = ZoneInfo("America/New_York")
    start_local = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=et)
    end_local = start_local + dt.timedelta(minutes=max(1, int(minutes)))
    s_utc = start_local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    e_utc = end_local.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return s_utc, e_utc


def estimate_cost(
    symbols: Sequence[str],
    start_iso: str,
    end_iso: str,
    *,
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
    stype_in: str = "parent",
    api_key: Optional[str] = None,
    get: Optional[Callable[..., Any]] = None,
    timeout: int = 90,
) -> float:
    """Run Databento `/v0/metadata.get_cost` ($0 dry-run) and return estimated USD cost."""
    import time

    key = api_key or load_api_key()
    if get is None:
        import requests

        get = requests.get
    sym_csv = ",".join(parent_symbol(s) if stype_in == "parent" else str(s).upper() for s in symbols)
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            resp = get(
                f"{HIST_BASE}/metadata.get_cost",
                auth=(key, ""),
                params={
                    "dataset": dataset,
                    "schema": schema,
                    "symbols": sym_csv,
                    "stype_in": stype_in,
                    "start": start_iso,
                    "end": end_iso,
                },
                timeout=timeout,
            )
            if int(getattr(resp, "status_code", 0)) == 200:
                val = float(str(resp.text).strip())
                if not math.isfinite(val) or val < 0:
                    raise ValueError(f"Invalid cost returned by Databento: {resp.text!r}")
                return val
            if int(getattr(resp, "status_code", 0)) in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            raise RuntimeError(f"Databento get_cost HTTP {resp.status_code}: {str(resp.text)[:200]}")
        except Exception as exc:
            last_exc = exc
            if attempt < 2 and "timeout" in str(exc).lower():
                time.sleep(2.0)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Databento get_cost failed")


def _f(val: Any) -> Optional[float]:
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "null", ""):
        return None
    try:
        x = float(s)
    except ValueError:
        return None
    return x if math.isfinite(x) else None


def parse_opra_csv(
    csv_text: str,
    *,
    trade_date: str,
    clock_et: str = "",
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
) -> List[Dict[str, Any]]:
    """Parse Databento OPRA CSV (`pretty_px=true`, `pretty_ts=true`, `map_symbols=true`) to chain rows.

    Note: in Databento `cbbo-1m`, `ts_recv` is the exact snapshot clock while `ts_event` is the
    time of the last underlying NBBO quote update for that contract (which may be earlier in the
    session for resting quotes). Tagging `clock_et` preserves each snapshot clock losslessly.
    """
    if not csv_text or not csv_text.strip():
        return []
    reader = csv.DictReader(io.StringIO(csv_text))
    if not reader.fieldnames:
        return []
    rows: List[Dict[str, Any]] = []
    for rec in reader:
        raw_sym = str(rec.get("symbol") or "").replace(" ", "").upper()
        parsed = parse_occ_symbol(raw_sym)
        if not parsed:
            continue
        root, expiration, opt_type, strike = parsed
        bid = _f(rec.get("bid_px_00") or rec.get("bid_px"))
        ask = _f(rec.get("ask_px_00") or rec.get("ask_px"))
        if bid is None and ask is None:
            continue
        mid = round(0.5 * (bid + ask), 4) if (bid is not None and ask is not None and ask >= bid) else None
        rows.append(
            {
                "symbol": root,
                "date": trade_date[:10],
                "clock_et": clock_et or str(rec.get("ts_recv") or rec.get("ts_event") or "")[:16],
                "ts_event": str(rec.get("ts_recv") or rec.get("ts_event") or trade_date),
                "contract": raw_sym,
                "expiration": expiration,
                "strike": float(strike),
                "type": opt_type,
                "bid": bid,
                "ask": ask,
                "mid": mid,
                "bid_size": _f(rec.get("bid_sz_00") or rec.get("bid_sz")),
                "ask_size": _f(rec.get("ask_sz_00") or rec.get("ask_sz")),
                "dataset": dataset,
                "schema": schema,
            }
        )
    return rows


def cumulative_spend_usd(db_path: str = DEFAULT_DB) -> float:
    full = repo_path(db_path)
    if not os.path.exists(full):
        return 0.0
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        row = con.execute("SELECT COALESCE(SUM(est_usd), 0.0) FROM opra_spend_log").fetchone()
        return float(row[0] or 0.0) if row else 0.0
    except sqlite3.Error:
        return 0.0
    finally:
        con.close()


def fetch_opra_snapshots(
    symbols: Sequence[str],
    dates: Sequence[str],
    *,
    hhmm_et: str = "15:45",
    minutes: int = 1,
    dataset: str = DEFAULT_DATASET,
    schema: str = DEFAULT_SCHEMA,
    max_usd: float = 2.0,
    dry_run: bool = True,
    db_path: str = DEFAULT_DB,
    api_key: Optional[str] = None,
    get: Optional[Callable[..., Any]] = None,
    post: Optional[Callable[..., Any]] = None,
    max_workers: int = 4,
    precomputed_plan: Optional[Sequence[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Estimate cost across `(symbols x dates)` and optionally download & archive OPRA snapshots.

    Never downloads if `dry_run=True` or if total estimated cost exceeds `max_usd`.
    """
    import time
    from concurrent.futures import ThreadPoolExecutor

    key = api_key or load_api_key()
    use_default_http = get is None and post is None
    if get is None or post is None:
        import requests

        get = get or requests.get
        post = post or requests.post

    def _est_one(d: str) -> Dict[str, Any]:
        s_iso, e_iso = snapshot_window_utc(d, hhmm_et=hhmm_et, minutes=minutes)
        cost = estimate_cost(
            symbols, s_iso, e_iso, dataset=dataset, schema=schema, api_key=key, get=get
        )
        return {"date": d, "start": s_iso, "end": e_iso, "est_usd": round(cost, 6)}

    workers = max(1, int(max_workers)) if (use_default_http and len(dates) > 1) else 1
    if precomputed_plan is not None:
        plan = list(precomputed_plan)
    elif workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            plan = list(ex.map(_est_one, dates))
    else:
        plan = [_est_one(d) for d in dates]
    total_usd = sum(float(item["est_usd"]) for item in plan)

    result: Dict[str, Any] = {
        "dataset": dataset,
        "schema": schema,
        "symbols": [s.upper() for s in symbols],
        "plan": plan,
        "total_est_usd": round(total_usd, 6),
        "max_usd": float(max_usd),
        "dry_run": bool(dry_run),
        "rows_written": 0,
        "prior_spend_usd": round(cumulative_spend_usd(db_path), 4),
    }
    if dry_run:
        return result
    if total_usd > max_usd:
        raise RuntimeError(
            f"Refused Databento download: estimated cost ${total_usd:.4f} exceeds max_usd=${max_usd:.2f}"
        )

    full = ensure_db(db_path)
    sym_csv = ",".join(parent_symbol(s) for s in symbols)
    now_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    total_rows = 0

    def _pull_one(item: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
        last_exc: Optional[Exception] = None
        for attempt in range(3):
            try:
                resp = post(
                    f"{HIST_BASE}/timeseries.get_range",
                    auth=(key, ""),
                    data={
                        "dataset": dataset,
                        "schema": schema,
                        "symbols": sym_csv,
                        "stype_in": "parent",
                        "start": item["start"],
                        "end": item["end"],
                        "encoding": "csv",
                        "pretty_px": "true",
                        "pretty_ts": "true",
                        "map_symbols": "true",
                    },
                    timeout=(15, 90),
                )
                code = int(getattr(resp, "status_code", 0))
                if code == 200:
                    return item, parse_opra_csv(
                        resp.text,
                        trade_date=item["date"],
                        clock_et=hhmm_et,
                        dataset=dataset,
                        schema=schema,
                    )
                if code in (429, 500, 502, 503, 504) and attempt < 2:
                    time.sleep(2.5 * (attempt + 1))
                    continue
                raise RuntimeError(f"Databento get_range HTTP {code}: {str(resp.text)[:200]}")
            except Exception as exc:
                last_exc = exc
                if attempt < 2 and "timeout" in str(exc).lower():
                    time.sleep(2.5)
                    continue
                raise
        raise last_exc or RuntimeError("Databento get_range failed")

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            pulled = list(ex.map(_pull_one, plan))
    else:
        pulled = [_pull_one(item) for item in plan]

    con = sqlite3.connect(full)
    try:
        for item, rows in pulled:
            if rows:
                con.executemany(
                    """
                    INSERT OR REPLACE INTO opra_chain
                    (symbol, date, clock_et, ts_event, contract, expiration, strike, type, bid, ask, mid,
                     bid_size, ask_size, dataset, schema)
                    VALUES (:symbol, :date, :clock_et, :ts_event, :contract, :expiration, :strike, :type,
                            :bid, :ask, :mid, :bid_size, :ask_size, :dataset, :schema)
                    """,
                    rows,
                )
            con.execute(
                """
                INSERT INTO opra_spend_log
                (logged_at, dataset, schema, symbols, start_iso, end_iso, est_usd, rows_written)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (now_ts, dataset, schema, sym_csv, item["start"], item["end"], item["est_usd"], len(rows)),
            )
            total_rows += len(rows)
        con.commit()
    finally:
        con.close()

    result["rows_written"] = total_rows
    result["new_cumulative_spend_usd"] = round(cumulative_spend_usd(db_path), 4)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Cost-guarded Databento OPRA snapshot tool")
    ap.add_argument("--symbols", nargs="+", required=True, help="Underlying tickers (e.g. SPY QQQ IWM)")
    ap.add_argument("--dates", nargs="+", required=True, help="Trade dates YYYY-MM-DD")
    ap.add_argument("--time", default="15:45", help="ET snapshot time HH:MM (default 15:45)")
    ap.add_argument("--schema", default=DEFAULT_SCHEMA)
    ap.add_argument("--max-usd", type=float, default=2.0)
    ap.add_argument("--download", action="store_true", help="Actually execute download (default is dry-run estimate)")
    ap.add_argument("--db", default=DEFAULT_DB)
    a = ap.parse_args(list(argv) if argv is not None else None)

    res = fetch_opra_snapshots(
        a.symbols, a.dates, hhmm_et=a.time, schema=a.schema, max_usd=a.max_usd, dry_run=not a.download, db_path=a.db
    )
    import json

    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_DATASET",
    "DEFAULT_SCHEMA",
    "DEFAULT_DB",
    "load_api_key",
    "ensure_db",
    "parent_symbol",
    "snapshot_window_utc",
    "estimate_cost",
    "parse_opra_csv",
    "cumulative_spend_usd",
    "fetch_opra_snapshots",
]
