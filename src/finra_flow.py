"""Free, zero-auth FINRA Daily Reg SHO Short-Sale Volume & Weekly ATS Dark-Pool collector.

Extends the `api.finra.org` pattern in `src/squeeze/backtest/finra.py` (which only pulls
bi-monthly consolidated short interest) to daily off-exchange short-sale volume (`regShoDaily`)
and weekly ATS dark-pool blocks (`weeklySummary`).

Storage: `data/finra_flow.db` (`regsho_daily`, `ats_weekly`).
Evidence label: WATCHLIST (descriptive microstructure flow; preregister OOS before sizing).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.paths import repo_path

REGSHO_URL = "https://api.finra.org/data/group/otcMarket/name/regShoDaily"
ATS_WEEKLY_URL = "https://api.finra.org/data/group/otcMarket/name/weeklySummary"
DEFAULT_DB = "data/finra_flow.db"
PAGE_SIZE = 5000

_DDL = """
CREATE TABLE IF NOT EXISTS regsho_daily (
    trade_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    short_vol REAL NOT NULL,
    short_exempt_vol REAL NOT NULL DEFAULT 0,
    total_vol REAL NOT NULL,
    short_ratio REAL NOT NULL,
    facilities INTEGER NOT NULL DEFAULT 1,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (trade_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_regsho_sym_dt ON regsho_daily (symbol, trade_date DESC);

CREATE TABLE IF NOT EXISTS ats_weekly (
    week_start TEXT NOT NULL,
    symbol TEXT NOT NULL,
    ats_shares REAL NOT NULL,
    ats_trades REAL NOT NULL,
    avg_trade_size REAL NOT NULL,
    venues INTEGER NOT NULL DEFAULT 1,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (week_start, symbol)
);
CREATE INDEX IF NOT EXISTS idx_ats_sym_wk ON ats_weekly (symbol, week_start DESC);
"""


@dataclass(frozen=True)
class FinraFlowSignal:
    symbol: str
    trade_date: str
    short_vol: float
    total_vol: float
    short_ratio: float
    avg_20d_ratio: Optional[float]
    ratio_z: Optional[float]
    ats_avg_block: Optional[float]
    ats_week: Optional[str]
    signal_label: str  # "HIGH_SHORT_VOL" | "LOW_SHORT_VOL_ACCUM" | "DARK_BLOCK_ACTIVE" | "NEUTRAL"
    evidence_label: str = "WATCHLIST"
    source: str = "FINRA RegSHO + ATS"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


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


def _post_json(url: str, payload: Dict[str, Any], timeout: int = 60) -> List[Dict[str, Any]]:
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    last_exc: Optional[Exception] = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8", errors="replace"))
                return data if isinstance(data, list) else []
        except urllib.error.HTTPError as exc:
            last_exc = exc
            if exc.code in (429, 500, 502, 503, 504) and attempt < 2:
                time.sleep(1.5 * (2**attempt))
                continue
            raise
        except Exception as exc:
            last_exc = exc
            if attempt < 2:
                time.sleep(1.5)
                continue
            raise
    if last_exc is not None:
        raise last_exc
    return []


def aggregate_regsho_rows(rows: Sequence[Dict[str, Any]], *, fetched_at: Optional[str] = None) -> List[Dict[str, Any]]:
    """Aggregate facility-level FINRA Reg SHO records into one row per `(trade_date, symbol)`."""
    now_ts = fetched_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    buckets: Dict[tuple[str, str], Dict[str, float]] = {}
    for r in rows or []:
        sym = str(r.get("securitiesInformationProcessorSymbolIdentifier") or r.get("symbol") or "").strip().upper()
        d = str(r.get("tradeReportDate") or r.get("date") or "")[:10]
        if not sym or len(d) != 10:
            continue
        try:
            sv = float(r.get("shortParQuantity") or 0.0)
            sev = float(r.get("shortExemptParQuantity") or 0.0)
            tv = float(r.get("totalParQuantity") or 0.0)
        except (TypeError, ValueError):
            continue
        if tv <= 0:
            continue
        b = buckets.setdefault((d, sym), {"sv": 0.0, "sev": 0.0, "tv": 0.0, "n": 0.0})
        b["sv"] += max(0.0, sv)
        b["sev"] += max(0.0, sev)
        b["tv"] += tv
        b["n"] += 1.0

    out: List[Dict[str, Any]] = []
    for (d, sym), b in sorted(buckets.items()):
        ratio = round(b["sv"] / b["tv"], 4) if b["tv"] > 0 else 0.0
        out.append(
            {
                "trade_date": d,
                "symbol": sym,
                "short_vol": round(b["sv"], 1),
                "short_exempt_vol": round(b["sev"], 1),
                "total_vol": round(b["tv"], 1),
                "short_ratio": ratio,
                "facilities": int(b["n"]),
                "fetched_at": now_ts,
            }
        )
    return out


def aggregate_ats_rows(rows: Sequence[Dict[str, Any]], *, fetched_at: Optional[str] = None) -> List[Dict[str, Any]]:
    """Aggregate venue-level FINRA weekly ATS dark-pool rows per `(week_start, symbol)`."""
    now_ts = fetched_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    buckets: Dict[tuple[str, str], Dict[str, float]] = {}
    for r in rows or []:
        sym = str(r.get("issueSymbolIdentifier") or r.get("symbol") or "").strip().upper()
        wk = str(r.get("weekStartDate") or "")[:10]
        if not sym or len(wk) != 10:
            continue
        try:
            sh = float(r.get("totalWeeklyShareQuantity") or 0.0)
            tr = float(r.get("totalWeeklyTradeCount") or 0.0)
        except (TypeError, ValueError):
            continue
        if sh <= 0 or tr <= 0:
            continue
        b = buckets.setdefault((wk, sym), {"sh": 0.0, "tr": 0.0, "n": 0.0})
        b["sh"] += sh
        b["tr"] += tr
        b["n"] += 1.0

    out: List[Dict[str, Any]] = []
    for (wk, sym), b in sorted(buckets.items()):
        out.append(
            {
                "week_start": wk,
                "symbol": sym,
                "ats_shares": round(b["sh"], 1),
                "ats_trades": round(b["tr"], 1),
                "avg_trade_size": round(b["sh"] / b["tr"], 1) if b["tr"] > 0 else 0.0,
                "venues": int(b["n"]),
                "fetched_at": now_ts,
            }
        )
    return out


def fetch_regsho_for_symbols(
    symbols: Sequence[str],
    *,
    start_date: str,
    end_date: Optional[str] = None,
    poster: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
    db_path: str = DEFAULT_DB,
) -> Dict[str, Any]:
    """Fetch FINRA Reg SHO daily short-sale volume for specific symbols in `[start_date, end_date]`."""
    post = poster or _post_json
    syms = sorted({str(s).upper().strip() for s in symbols if s})
    end_d = end_date or dt.date.today().isoformat()
    raw_all: List[Dict[str, Any]] = []

    for i in range(0, len(syms), 25):
        chunk = syms[i : i + 25]
        offset = 0
        while True:
            payload: Dict[str, Any] = {
                "limit": PAGE_SIZE,
                "offset": offset,
                "domainFilters": [
                    {
                        "fieldName": "securitiesInformationProcessorSymbolIdentifier",
                        "values": chunk,
                    }
                ],
                "dateRangeFilters": [
                    {
                        "fieldName": "tradeReportDate",
                        "startDate": start_date[:10],
                        "endDate": end_d[:10],
                    }
                ],
            }
            batch = post(REGSHO_URL, payload)
            if not batch:
                break
            raw_all.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(0.2)

    agg = aggregate_regsho_rows(raw_all)
    if agg:
        full = ensure_db(db_path)
        con = sqlite3.connect(full)
        try:
            con.executemany(
                """
                INSERT OR REPLACE INTO regsho_daily
                (trade_date, symbol, short_vol, short_exempt_vol, total_vol, short_ratio, facilities, fetched_at)
                VALUES (:trade_date, :symbol, :short_vol, :short_exempt_vol, :total_vol, :short_ratio, :facilities, :fetched_at)
                """,
                agg,
            )
            con.commit()
        finally:
            con.close()
    return {"symbols": len(syms), "raw_rows": len(raw_all), "saved": len(agg), "start": start_date, "end": end_d}


def fetch_ats_for_symbols(
    symbols: Sequence[str],
    *,
    start_date: str,
    end_date: Optional[str] = None,
    poster: Optional[Callable[[str, Dict[str, Any]], List[Dict[str, Any]]]] = None,
    db_path: str = DEFAULT_DB,
) -> Dict[str, Any]:
    """Fetch FINRA weekly ATS dark-pool volume & block trade size for specific symbols."""
    post = poster or _post_json
    syms = sorted({str(s).upper().strip() for s in symbols if s})
    end_d = end_date or dt.date.today().isoformat()
    raw_all: List[Dict[str, Any]] = []

    for i in range(0, len(syms), 25):
        chunk = syms[i : i + 25]
        offset = 0
        while True:
            payload: Dict[str, Any] = {
                "limit": PAGE_SIZE,
                "offset": offset,
                "compareFilters": [
                    {"fieldName": "summaryTypeCode", "fieldValue": "ATS_W_SMBL", "compareType": "equal"}
                ],
                "domainFilters": [{"fieldName": "issueSymbolIdentifier", "values": chunk}],
                "dateRangeFilters": [
                    {
                        "fieldName": "weekStartDate",
                        "startDate": start_date[:10],
                        "endDate": end_d[:10],
                    }
                ],
            }
            batch = post(ATS_WEEKLY_URL, payload)
            if not batch:
                break
            raw_all.extend(batch)
            if len(batch) < PAGE_SIZE:
                break
            offset += PAGE_SIZE
            time.sleep(0.2)

    agg = aggregate_ats_rows(raw_all)
    if agg:
        full = ensure_db(db_path)
        con = sqlite3.connect(full)
        try:
            con.executemany(
                """
                INSERT OR REPLACE INTO ats_weekly
                (week_start, symbol, ats_shares, ats_trades, avg_trade_size, venues, fetched_at)
                VALUES (:week_start, :symbol, :ats_shares, :ats_trades, :avg_trade_size, :venues, :fetched_at)
                """,
                agg,
            )
            con.commit()
        finally:
            con.close()
    return {"symbols": len(syms), "raw_rows": len(raw_all), "saved": len(agg), "start": start_date, "end": end_d}


def scan_finra_flow_signals(
    *,
    db_path: str = DEFAULT_DB,
    symbols: Optional[Sequence[str]] = None,
    min_total_vol: float = 100_000.0,
    limit: int = 30,
) -> List[FinraFlowSignal]:
    """Rank symbols by abnormal daily Reg SHO short-volume ratio and dark-pool block size."""
    import numpy as np

    full = repo_path(db_path)
    if not os.path.exists(full):
        return []
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "regsho_daily" not in tables:
            return []
        sym_filter = {s.upper().strip() for s in symbols} if symbols else None
        latest_rows = con.execute(
            """
            SELECT r.* FROM regsho_daily r
            INNER JOIN (SELECT symbol, MAX(trade_date) AS mx FROM regsho_daily GROUP BY symbol) m
            ON r.symbol = m.symbol AND r.trade_date = m.mx
            """
        ).fetchall()

        ats_map: Dict[str, tuple[str, float]] = {}
        if "ats_weekly" in tables:
            for ar in con.execute(
                """
                SELECT a.symbol, a.week_start, a.avg_trade_size FROM ats_weekly a
                INNER JOIN (SELECT symbol, MAX(week_start) AS mx FROM ats_weekly GROUP BY symbol) m
                ON a.symbol = m.symbol AND a.week_start = m.mx
                """
            ).fetchall():
                ats_map[str(ar["symbol"]).upper()] = (str(ar["week_start"]), float(ar["avg_trade_size"]))

        out: List[FinraFlowSignal] = []
        for r in latest_rows:
            sym = str(r["symbol"]).upper()
            if sym_filter and sym not in sym_filter:
                continue
            tv = float(r["total_vol"] or 0.0)
            if not sym_filter and tv < min_total_vol:
                continue
            sv = float(r["short_vol"] or 0.0)
            ratio = float(r["short_ratio"] or 0.0)
            hist = [
                float(x[0])
                for x in con.execute(
                    "SELECT short_ratio FROM regsho_daily WHERE symbol = ? AND trade_date < ? "
                    "ORDER BY trade_date DESC LIMIT 20",
                    (sym, str(r["trade_date"])),
                ).fetchall()
            ]
            avg20 = round(float(np.mean(hist)), 4) if len(hist) >= 5 else None
            sd20 = float(np.std(hist, ddof=1)) if len(hist) >= 5 else 0.0
            z = round((ratio - avg20) / sd20, 2) if (avg20 is not None and sd20 > 1e-4) else None
            ats_wk, ats_blk = ats_map.get(sym, (None, None))

            label = "NEUTRAL"
            if (z is not None and z >= 1.8) or ratio >= 0.62:
                label = "HIGH_SHORT_VOL"
            elif (z is not None and z <= -1.8) or (ratio <= 0.32 and (ats_blk or 0.0) >= 200.0):
                label = "LOW_SHORT_VOL_ACCUM"
            elif ats_blk is not None and ats_blk >= 350.0:
                label = "DARK_BLOCK_ACTIVE"

            out.append(
                FinraFlowSignal(
                    symbol=sym,
                    trade_date=str(r["trade_date"]),
                    short_vol=sv,
                    total_vol=tv,
                    short_ratio=ratio,
                    avg_20d_ratio=avg20,
                    ratio_z=z,
                    ats_avg_block=ats_blk,
                    ats_week=ats_wk,
                    signal_label=label,
                )
            )
    finally:
        con.close()

    out.sort(key=lambda s: (0 if s.signal_label != "NEUTRAL" else 1, -abs(s.ratio_z or 0.0), -s.short_ratio))
    return out[:limit]


__all__ = [
    "DEFAULT_DB",
    "FinraFlowSignal",
    "ensure_db",
    "aggregate_regsho_rows",
    "aggregate_ats_rows",
    "fetch_regsho_for_symbols",
    "fetch_ats_for_symbols",
    "scan_finra_flow_signals",
]
