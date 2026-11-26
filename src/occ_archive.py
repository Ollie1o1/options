"""Free Whole-Market OCC Daily Volume & Open Interest Archiver (`src/occ_archive.py`).

Collects free daily equity/ETF option volume & open interest across the entire US options
market from OCC public volume/OI endpoints (with graceful fallback to summarizing local
`data/chain_archive.db` when OCC endpoints are rate-limited or offline).

Storage: `data/occ_oi.db` table `occ_daily_oi`.
Evidence label: WATCHLIST (descriptive UOA / OI-delta telemetry until >=1yr history accrues).
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import os
import sqlite3
import urllib.request
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.paths import repo_path

DEFAULT_OCC_DB = "data/occ_oi.db"
# OCC public daily volume/OI download endpoint (free, keyless)
OCC_DAILY_CSV_URL = "https://marketdata.theocc.com/cont-volume-download?reportDate={ymd}&format=csv"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS occ_daily_oi (
    trade_date TEXT NOT NULL,
    symbol TEXT NOT NULL,
    call_vol REAL NOT NULL DEFAULT 0,
    put_vol REAL NOT NULL DEFAULT 0,
    call_oi REAL NOT NULL DEFAULT 0,
    put_oi REAL NOT NULL DEFAULT 0,
    total_vol REAL NOT NULL DEFAULT 0,
    total_oi REAL NOT NULL DEFAULT 0,
    source TEXT NOT NULL DEFAULT 'occ',
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (trade_date, symbol)
);
CREATE INDEX IF NOT EXISTS idx_occ_sym_date ON occ_daily_oi (symbol, trade_date DESC);
CREATE INDEX IF NOT EXISTS idx_occ_date_vol ON occ_daily_oi (trade_date DESC, total_vol DESC);
"""


@dataclass(frozen=True)
class OccUoaSignal:
    symbol: str
    trade_date: str
    total_vol: float
    total_oi: float
    oi_delta_1d: Optional[float]
    oi_delta_pct: Optional[float]
    vol_to_oi: Optional[float]
    pcr_vol: Optional[float]
    call_share_pct: Optional[float]
    signal_label: str       # "CALL_OI_SURGE" | "PUT_OI_SURGE" | "HIGH_VOL_OI" | "NEUTRAL"
    source: str
    evidence_label: str = "WATCHLIST"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_occ_db(db_path: str = DEFAULT_OCC_DB) -> str:
    full = repo_path(db_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    con = sqlite3.connect(full)
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()
    return full


def _fetch_occ_text(url: str, timeout: int = 20) -> str:
    req = urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; options-desk/2.0)",
            "Accept": "text/csv,text/plain,*/*",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_occ_volume_csv(text: str, trade_date: str) -> List[Dict[str, Any]]:
    """Aggregate OCC per-series or per-symbol CSV rows into daily symbol-level Call/Put Vol & OI."""
    if not text or "<html" in text[:300].lower():
        return []
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return []

    agg: Dict[str, Dict[str, float]] = {}
    for row in reader:
        lowered = {str(k).strip().lower(): str(v or "").strip() for k, v in row.items() if k}
        sym = (
            lowered.get("underlying")
            or lowered.get("symbol")
            or lowered.get("root")
            or lowered.get("product")
            or ""
        ).upper().strip()
        if not sym or not sym.replace("-", "").replace(".", "").isalnum() or len(sym) > 8:
            continue

        def _num(*keys: str) -> float:
            for k in keys:
                raw = lowered.get(k, "").replace(",", "")
                if raw:
                    try:
                        return max(0.0, float(raw))
                    except ValueError:
                        continue
            return 0.0

        slot = agg.setdefault(sym, {"call_vol": 0.0, "put_vol": 0.0, "call_oi": 0.0, "put_oi": 0.0})
        c_vol = _num("call_volume", "call volume", "calls_vol", "callvol")
        p_vol = _num("put_volume", "put volume", "puts_vol", "putvol")
        c_oi = _num("call_oi", "call open interest", "call_open_interest", "calloi")
        p_oi = _num("put_oi", "put open interest", "put_open_interest", "putoi")

        if c_vol == 0.0 and p_vol == 0.0 and c_oi == 0.0 and p_oi == 0.0:
            cp = (lowered.get("call/put") or lowered.get("type") or lowered.get("side") or "").upper()
            v_val = _num("volume", "total_volume", "total volume", "quantity")
            oi_val = _num("open_interest", "open interest", "oi")
            if cp.startswith("C"):
                c_vol, c_oi = v_val, oi_val
            elif cp.startswith("P"):
                p_vol, p_oi = v_val, oi_val
            else:
                continue

        slot["call_vol"] += c_vol
        slot["put_vol"] += p_vol
        slot["call_oi"] += c_oi
        slot["put_oi"] += p_oi

    now_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out: List[Dict[str, Any]] = []
    for sym, v in sorted(agg.items()):
        t_vol = v["call_vol"] + v["put_vol"]
        t_oi = v["call_oi"] + v["put_oi"]
        if t_vol <= 0 and t_oi <= 0:
            continue
        out.append(
            {
                "trade_date": trade_date,
                "symbol": sym,
                "call_vol": round(v["call_vol"], 2),
                "put_vol": round(v["put_vol"], 2),
                "call_oi": round(v["call_oi"], 2),
                "put_oi": round(v["put_oi"], 2),
                "total_vol": round(t_vol, 2),
                "total_oi": round(t_oi, 2),
                "source": "occ",
                "fetched_at": now_ts,
            }
        )
    return out


def rollup_from_chain_archive(
    chain_db: str = "data/chain_archive.db",
    *,
    trade_date: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Aggregate symbol-level Call/Put Vol & OI from local `data/chain_archive.db` as fallback."""
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return []
    now_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "chain_snapshots" not in tables:
            return []
        if trade_date is None:
            row = con.execute("SELECT MAX(snap_date) FROM chain_snapshots").fetchone()
            if not row or not row[0]:
                return []
            trade_date = str(row[0])[:10]
        rows = con.execute(
            """
            SELECT symbol,
                   SUM(CASE WHEN LOWER(type) LIKE 'c%' THEN COALESCE(volume, 0) ELSE 0 END) AS c_vol,
                   SUM(CASE WHEN LOWER(type) LIKE 'p%' THEN COALESCE(volume, 0) ELSE 0 END) AS p_vol,
                   SUM(CASE WHEN LOWER(type) LIKE 'c%' THEN COALESCE(open_interest, 0) ELSE 0 END) AS c_oi,
                   SUM(CASE WHEN LOWER(type) LIKE 'p%' THEN COALESCE(open_interest, 0) ELSE 0 END) AS p_oi
            FROM chain_snapshots
            WHERE snap_date = ?
            GROUP BY symbol
            """,
            (trade_date,),
        ).fetchall()
    finally:
        con.close()

    out: List[Dict[str, Any]] = []
    for sym, cv, pv, coi, poi in rows:
        cv_f, pv_f, coi_f, poi_f = float(cv or 0), float(pv or 0), float(coi or 0), float(poi or 0)
        tv, toi = cv_f + pv_f, coi_f + poi_f
        if tv <= 0 and toi <= 0:
            continue
        out.append(
            {
                "trade_date": trade_date,
                "symbol": str(sym).upper(),
                "call_vol": round(cv_f, 2),
                "put_vol": round(pv_f, 2),
                "call_oi": round(coi_f, 2),
                "put_oi": round(poi_f, 2),
                "total_vol": round(tv, 2),
                "total_oi": round(toi, 2),
                "source": "chain_archive_rollup",
                "fetched_at": now_ts,
            }
        )
    return out


def archive_occ_daily(
    *,
    db_path: str = DEFAULT_OCC_DB,
    chain_db: str = "data/chain_archive.db",
    trade_date: Optional[str] = None,
    fetcher: Optional[Callable[[str], str]] = None,
) -> Dict[str, Any]:
    """Fetch OCC daily whole-market volume/OI (falling back to chain_archive rollup) and persist."""
    full = ensure_occ_db(db_path)
    d_obj = dt.date.fromisoformat(trade_date[:10]) if trade_date else dt.date.today()
    # Roll back weekend to Friday
    while d_obj.weekday() >= 5:
        d_obj -= dt.timedelta(days=1)
    d_str = d_obj.isoformat()
    ymd = d_obj.strftime("%Y%m%d")

    get = fetcher or _fetch_occ_text
    rows: List[Dict[str, Any]] = []
    source_used = "occ"
    errors: List[str] = []

    try:
        txt = get(OCC_DAILY_CSV_URL.format(ymd=ymd))
        rows = parse_occ_volume_csv(txt, d_str)
        if not rows:
            errors.append("occ_empty_or_html_response")
    except Exception as exc:
        errors.append(f"occ_fetch: {exc}")

    if not rows:
        rows = rollup_from_chain_archive(chain_db=chain_db, trade_date=None)
        source_used = "chain_archive_rollup" if rows else "none"

    saved = 0
    if rows:
        con = sqlite3.connect(full)
        try:
            for r in rows:
                con.execute(
                    """
                    INSERT OR REPLACE INTO occ_daily_oi (
                        trade_date, symbol, call_vol, put_vol, call_oi, put_oi,
                        total_vol, total_oi, source, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        r["trade_date"],
                        r["symbol"],
                        r["call_vol"],
                        r["put_vol"],
                        r["call_oi"],
                        r["put_oi"],
                        r["total_vol"],
                        r["total_oi"],
                        r["source"],
                        r["fetched_at"],
                    ),
                )
                saved += 1
            con.commit()
        finally:
            con.close()

    return {
        "trade_date": rows[0]["trade_date"] if rows else d_str,
        "saved": saved,
        "source": source_used,
        "errors": errors,
    }


def scan_occ_uoa(
    *,
    db_path: str = DEFAULT_OCC_DB,
    symbols: Optional[Sequence[str]] = None,
    min_total_vol: float = 500.0,
    limit: int = 30,
) -> List[OccUoaSignal]:
    """Rank symbols by Vol/OI ratio and 1-day OI delta from `data/occ_oi.db`."""
    full = repo_path(db_path)
    if not os.path.exists(full):
        return []
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "occ_daily_oi" not in tables:
            return []
        dates = [
            str(r[0])
            for r in con.execute(
                "SELECT DISTINCT trade_date FROM occ_daily_oi ORDER BY trade_date DESC LIMIT 2"
            ).fetchall()
        ]
        if not dates:
            return []
        latest_d = dates[0]
        prev_d = dates[1] if len(dates) > 1 else None

        sym_filter = {s.upper().strip() for s in symbols} if symbols else None
        cur_rows = con.execute(
            "SELECT * FROM occ_daily_oi WHERE trade_date = ?",
            (latest_d,),
        ).fetchall()
        prev_map: Dict[str, Dict[str, Any]] = {}
        if prev_d:
            for r in con.execute("SELECT * FROM occ_daily_oi WHERE trade_date = ?", (prev_d,)).fetchall():
                prev_map[str(r["symbol"]).upper()] = dict(r)
    finally:
        con.close()

    signals: List[OccUoaSignal] = []
    for r in cur_rows:
        sym = str(r["symbol"]).upper()
        if sym_filter and sym not in sym_filter:
            continue
        t_vol = float(r["total_vol"] or 0.0)
        t_oi = float(r["total_oi"] or 0.0)
        if not sym_filter and t_vol < min_total_vol:
            continue
        c_vol = float(r["call_vol"] or 0.0)
        p_vol = float(r["put_vol"] or 0.0)
        vol_oi = round(t_vol / t_oi, 3) if t_oi > 0 else None
        pcr = round(p_vol / c_vol, 2) if c_vol > 0 else None
        call_share = round((c_vol / t_vol) * 100.0, 1) if t_vol > 0 else None

        oi_delta = oi_delta_pct = None
        prev = prev_map.get(sym)
        if prev and float(prev.get("total_oi") or 0) > 0:
            p_oi_prev = float(prev["total_oi"])
            oi_delta = round(t_oi - p_oi_prev, 1)
            oi_delta_pct = round((oi_delta / p_oi_prev) * 100.0, 2)

        label = "NEUTRAL"
        if oi_delta_pct is not None and oi_delta_pct >= 10.0 and (call_share or 0) >= 60.0:
            label = "CALL_OI_SURGE"
        elif oi_delta_pct is not None and oi_delta_pct >= 10.0 and (call_share or 100) <= 40.0:
            label = "PUT_OI_SURGE"
        elif vol_oi is not None and vol_oi >= 0.50:
            label = "HIGH_VOL_OI"

        signals.append(
            OccUoaSignal(
                symbol=sym,
                trade_date=latest_d,
                total_vol=t_vol,
                total_oi=t_oi,
                oi_delta_1d=oi_delta,
                oi_delta_pct=oi_delta_pct,
                vol_to_oi=vol_oi,
                pcr_vol=pcr,
                call_share_pct=call_share,
                signal_label=label,
                source=str(r["source"] or "occ"),
            )
        )

    signals.sort(
        key=lambda s: (
            0 if s.signal_label != "NEUTRAL" else 1,
            -(s.vol_to_oi or 0.0),
            -s.total_vol,
        )
    )
    return signals[:limit]
