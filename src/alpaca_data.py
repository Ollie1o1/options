"""Unified Alpaca Market Data client for options snapshots, historical option bars,
Benzinga news, and corporate actions (`src/alpaca_data.py`).

Reuses `APCA_API_KEY_ID` / `APCA_API_SECRET_KEY` via `src.intraday.alpaca_bars.load_credentials`.
Free-tier notes (honest provenance):
  * Option snapshots default to `feed="indicative"` (15-min delayed/derived quotes with
    Greeks, IV, and sizes). Always stamped `source="alpaca_indicative"` so research and
    fill studies never mistake indicative quotes for live OPRA NBBO.
  * Historical option bars cover Feb 2024+ across all expirations (including 60-120 DTE
    and LEAPS), breaking DoltHub's 67-DTE ceiling.
  * HTTP `get` is injectable so every parser and paginator is unit-testable offline.
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from src.cboe_client import parse_occ_symbol
from src.intraday.alpaca_bars import clamp_end, load_credentials
from src.paths import repo_path

OPTIONS_SNAPSHOT_URL = "https://data.alpaca.markets/v1beta1/options/snapshots/{symbol}"
OPTIONS_BARS_URL = "https://data.alpaca.markets/v1beta1/options/bars"
NEWS_URL = "https://data.alpaca.markets/v1beta1/news"
CORP_ACTIONS_URL = "https://data.alpaca.markets/v1/corporate-actions"
DEFAULT_NEWS_DB = "data/alpaca_news.db"

_NEWS_SCHEMA = """
CREATE TABLE IF NOT EXISTS alpaca_news (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    updated_at TEXT,
    headline TEXT NOT NULL,
    summary TEXT,
    source TEXT,
    url TEXT,
    symbols TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_alpaca_news_created ON alpaca_news (created_at DESC);
CREATE TABLE IF NOT EXISTS alpaca_news_symbols (
    news_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (news_id, symbol)
);
CREATE INDEX IF NOT EXISTS idx_alpaca_news_sym_dt ON alpaca_news_symbols (symbol, created_at DESC);
"""


def has_credentials(env_path: Optional[Path] = None) -> bool:
    """Return True if Alpaca key ID and secret are configured."""
    try:
        key, secret = load_credentials(env_path=env_path)
        return bool(key and secret)
    except Exception:
        return False


def _auth_headers(env_path: Optional[Path] = None) -> Dict[str, str]:
    key, secret = load_credentials(env_path=env_path)
    return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}


def _f(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        x = float(val)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def _request_json(
    get: Callable[..., Any],
    url: str,
    *,
    headers: Dict[str, str],
    params: Dict[str, Any],
    timeout: int = 25,
    max_retries: int = 3,
) -> Dict[str, Any]:
    last_err: Optional[str] = None
    for attempt in range(max_retries):
        resp = get(url, headers=headers, params=dict(params), timeout=timeout)
        status = int(getattr(resp, "status_code", 200))
        if status == 200:
            body = resp.json()
            return body if isinstance(body, dict) else {}
        if status in (429, 500, 502, 503, 504) and attempt + 1 < max_retries:
            time.sleep(min(8.0, 1.0 * (2**attempt)))
            continue
        last_err = f"HTTP {status}: {str(getattr(resp, 'text', ''))[:200]}"
        break
    raise RuntimeError(f"Alpaca request failed for {url}: {last_err or 'unknown error'}")


# ── 1. Option Chain Snapshots ────────────────────────────────────────────────


def parse_option_snapshots(
    snapshots: Dict[str, Any],
    *,
    symbol: str,
    spot: Optional[float] = None,
    feed: str = "indicative",
) -> List[Dict[str, Any]]:
    """Normalize Alpaca `/v1beta1/options/snapshots/{symbol}` payload to chain_archive schema."""
    sym_u = symbol.upper().strip()
    src_tag = f"alpaca_{feed.lower()}"
    out: List[Dict[str, Any]] = []
    for occ, snap in (snapshots or {}).items():
        if not isinstance(snap, dict):
            continue
        parsed = parse_occ_symbol(str(occ))
        if not parsed:
            continue
        root, expiration, opt_type, strike = parsed
        q = snap.get("latestQuote") or {}
        tr = snap.get("latestTrade") or {}
        bar = snap.get("dailyBar") or {}
        gr = snap.get("greeks") or {}

        bid = _f(q.get("bp"))
        ask = _f(q.get("ap"))
        bid_size = _f(q.get("bs"))
        ask_size = _f(q.get("as"))
        iv = _f(snap.get("impliedVolatility"))
        if iv is not None and iv > 5.0:
            iv = iv / 100.0
        vol = _f(bar.get("v")) or 0.0
        oi = _f(snap.get("openInterest")) or 0.0

        out.append(
            {
                "symbol": root or sym_u,
                "contract": str(occ),
                "type": opt_type,
                "strike": float(strike),
                "expiration": expiration,
                "bid": bid,
                "ask": ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "iv": iv,
                "delta": _f(gr.get("delta")),
                "gamma": _f(gr.get("gamma")),
                "theta": _f(gr.get("theta")),
                "vega": _f(gr.get("vega")),
                "rho": _f(gr.get("rho")),
                "open_interest": oi,
                "volume": vol,
                "last_trade_time": str(tr.get("t")) if tr.get("t") else None,
                "spot": spot,
                "snapshot_ts": str(q.get("t") or tr.get("t") or "") or None,
                "source": src_tag,
            }
        )
    out.sort(key=lambda r: (str(r["expiration"]), str(r["type"]), float(r["strike"])))
    return out


def fetch_option_chain_snapshots(
    symbol: str,
    *,
    feed: str = "indicative",
    min_dte: int = 1,
    max_dte: int = 150,
    spot: Optional[float] = None,
    moneyness_band: Optional[float] = 0.25,
    opt_type: Optional[str] = None,
    limit_per_page: int = 1000,
    max_pages: int = 6,
    today: Optional[dt.date] = None,
    get: Optional[Callable[..., Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch full or DTE/strike-bounded option chain snapshots from Alpaca.

    Returns [] cleanly if credentials are missing or the request fails.
    """
    sym = (symbol or "").upper().strip()
    if not sym:
        return []
    if headers is None:
        try:
            headers = _auth_headers()
        except Exception:
            return []
    if get is None:
        import requests

        get = requests.get

    today = today or dt.date.today()
    exp_gte = (today + dt.timedelta(days=max(0, int(min_dte)))).isoformat()
    exp_lte = (today + dt.timedelta(days=max(int(min_dte), int(max_dte)))).isoformat()

    params: Dict[str, Any] = {
        "feed": feed,
        "expiration_date_gte": exp_gte,
        "expiration_date_lte": exp_lte,
        "limit": max(1, min(int(limit_per_page), 1000)),
    }
    if opt_type and opt_type.lower() in ("call", "put"):
        params["type"] = opt_type.lower()
    if spot is not None and spot > 0 and moneyness_band is not None and moneyness_band > 0:
        params["strike_price_gte"] = round(spot * (1.0 - moneyness_band), 2)
        params["strike_price_lte"] = round(spot * (1.0 + moneyness_band), 2)

    all_snaps: Dict[str, Any] = {}
    token: Optional[str] = None
    url = OPTIONS_SNAPSHOT_URL.format(symbol=sym)
    for _ in range(max(1, int(max_pages))):
        if token:
            params["page_token"] = token
        else:
            params.pop("page_token", None)
        try:
            body = _request_json(get, url, headers=headers, params=params, timeout=20)
        except Exception:
            break
        snaps = body.get("snapshots") or {}
        if isinstance(snaps, dict):
            all_snaps.update(snaps)
        token = body.get("next_page_token")
        if not token:
            break

    return parse_option_snapshots(all_snaps, symbol=sym, spot=spot, feed=feed)


# ── 2. Historical Option Bars (Feb 2024+) ────────────────────────────────────


def fetch_option_bars(
    contracts: Sequence[str],
    *,
    start: str,
    end: Optional[str] = None,
    timeframe: str = "1Day",
    limit: int = 10000,
    max_pages: int = 10,
    get: Optional[Callable[..., Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch historical option OHLCV bars for up to N OCC contracts (batched by 100).

    `start`/`end` accept `YYYY-MM-DD` or ISO-8601 UTC timestamps.
    """
    syms = [str(c).upper().strip() for c in contracts if c]
    if not syms:
        return []
    if headers is None:
        headers = _auth_headers()
    if get is None:
        import requests

        get = requests.get

    start_iso = start if "T" in start else f"{start}T00:00:00Z"
    end_raw = end or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    end_iso = clamp_end(end_raw if "T" in end_raw else f"{end_raw}T23:59:59Z")

    rows: List[Dict[str, Any]] = []
    for i in range(0, len(syms), 100):
        chunk = syms[i : i + 100]
        params: Dict[str, Any] = {
            "symbols": ",".join(chunk),
            "timeframe": timeframe,
            "start": start_iso,
            "end": end_iso,
            "limit": max(1, min(int(limit), 10000)),
            "sort": "asc",
        }
        token: Optional[str] = None
        for _ in range(max(1, int(max_pages))):
            if token:
                params["page_token"] = token
            else:
                params.pop("page_token", None)
            body = _request_json(get, OPTIONS_BARS_URL, headers=headers, params=params, timeout=30)
            bars_map = body.get("bars") or {}
            if isinstance(bars_map, dict):
                for occ, blist in bars_map.items():
                    for b in blist or []:
                        rows.append(
                            {
                                "contract": occ,
                                "t": b.get("t"),
                                "o": _f(b.get("o")),
                                "h": _f(b.get("h")),
                                "l": _f(b.get("l")),
                                "c": _f(b.get("c")),
                                "v": _f(b.get("v")) or 0.0,
                                "n": int(b.get("n") or 0),
                                "vw": _f(b.get("vw")),
                            }
                        )
            token = body.get("next_page_token")
            if not token:
                break
    return rows


# ── 3. Benzinga Historical News (2015+) ──────────────────────────────────────


def ensure_news_db(db_path: str = DEFAULT_NEWS_DB) -> str:
    full = repo_path(db_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    con = sqlite3.connect(full)
    try:
        con.executescript(_NEWS_SCHEMA)
        con.commit()
    finally:
        con.close()
    return full


def fetch_news(
    symbols: Sequence[str],
    *,
    start: Optional[str] = None,
    end: Optional[str] = None,
    limit: int = 50,
    max_pages: int = 10,
    include_content: bool = False,
    get: Optional[Callable[..., Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch timestamped Benzinga news items from Alpaca `/v1beta1/news`."""
    syms = [str(s).upper().strip() for s in symbols if s]
    if headers is None:
        headers = _auth_headers()
    if get is None:
        import requests

        get = requests.get

    params: Dict[str, Any] = {
        "limit": max(1, min(int(limit), 50)),
        "sort": "desc",
        "include_content": "true" if include_content else "false",
    }
    if syms:
        params["symbols"] = ",".join(syms)
    if start:
        params["start"] = start if "T" in start else f"{start}T00:00:00Z"
    if end:
        params["end"] = end if "T" in end else f"{end}T23:59:59Z"

    items: List[Dict[str, Any]] = []
    token: Optional[str] = None
    for _ in range(max(1, int(max_pages))):
        if token:
            params["page_token"] = token
        else:
            params.pop("page_token", None)
        body = _request_json(get, NEWS_URL, headers=headers, params=params, timeout=20)
        batch = body.get("news") or []
        if isinstance(batch, list):
            for n in batch:
                if isinstance(n, dict) and n.get("id") is not None:
                    items.append(n)
        token = body.get("next_page_token")
        if not token:
            break
    return items


def save_news_archive(items: Sequence[Dict[str, Any]], db_path: str = DEFAULT_NEWS_DB) -> int:
    """Persist fetched Alpaca Benzinga news items idempotently into SQLite."""
    if not items:
        return 0
    full = ensure_news_db(db_path)
    now_ts = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    con = sqlite3.connect(full)
    saved = 0
    try:
        for n in items:
            nid = str(n.get("id") or "")
            created = str(n.get("created_at") or "")
            headline = str(n.get("headline") or "")
            if not nid or not created or not headline:
                continue
            syms = [str(s).upper().strip() for s in (n.get("symbols") or []) if s]
            con.execute(
                """
                INSERT OR IGNORE INTO alpaca_news
                (id, created_at, updated_at, headline, summary, source, url, symbols, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    nid,
                    created,
                    str(n.get("updated_at") or ""),
                    headline,
                    str(n.get("summary") or ""),
                    str(n.get("source") or "benzinga"),
                    str(n.get("url") or ""),
                    ",".join(syms),
                    now_ts,
                ),
            )
            for s in syms:
                con.execute(
                    "INSERT OR IGNORE INTO alpaca_news_symbols (news_id, symbol, created_at) VALUES (?, ?, ?)",
                    (nid, s, created),
                )
            saved += 1
        con.commit()
    finally:
        con.close()
    return saved


# ── 4. Corporate Actions (Ex-Dividend & Split Guards) ────────────────────────


def fetch_corporate_actions(
    symbols: Sequence[str],
    *,
    types: Sequence[str] = ("cash_dividend", "stock_dividend", "forward_split", "reverse_split", "spin_off"),
    start: Optional[str] = None,
    end: Optional[str] = None,
    get: Optional[Callable[..., Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Fetch upcoming or historical corporate actions from Alpaca (best-effort, fails soft)."""
    syms = [str(s).upper().strip() for s in symbols if s]
    if not syms:
        return []
    if headers is None:
        try:
            headers = _auth_headers()
        except Exception:
            return []
    if get is None:
        import requests

        get = requests.get

    today = dt.date.today()
    params: Dict[str, Any] = {
        "symbols": ",".join(syms[:100]),
        "types": ",".join(types),
        "start": start or today.isoformat(),
        "end": end or (today + dt.timedelta(days=120)).isoformat(),
        "limit": 1000,
    }
    try:
        body = _request_json(get, CORP_ACTIONS_URL, headers=headers, params=params, timeout=15, max_retries=1)
    except Exception:
        return []

    actions_root = body.get("corporate_actions") or {}
    out: List[Dict[str, Any]] = []
    if isinstance(actions_root, dict):
        for action_kind, rows in actions_root.items():
            for row in rows or []:
                if isinstance(row, dict):
                    out.append({"action_type": str(action_kind), **row})
    return out


def upcoming_ex_dividend_dates(
    symbols: Sequence[str],
    *,
    horizon_days: int = 120,
    get: Optional[Callable[..., Any]] = None,
    headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Tuple[str, Optional[float]]]:
    """Return `{SYMBOL: (ex_date_iso, cash_amount)}` for upcoming dividends within `horizon_days`."""
    today = dt.date.today()
    rows = fetch_corporate_actions(
        symbols,
        types=("cash_dividend",),
        start=today.isoformat(),
        end=(today + dt.timedelta(days=max(1, int(horizon_days)))).isoformat(),
        get=get,
        headers=headers,
    )
    out: Dict[str, Tuple[str, Optional[float]]] = {}
    for r in rows:
        sym = str(r.get("symbol") or "").upper().strip()
        ex_d = str(r.get("ex_date") or "")[:10]
        amt = _f(r.get("rate") or r.get("cash"))
        if not sym or len(ex_d) != 10:
            continue
        prev = out.get(sym)
        if prev is None or ex_d < prev[0]:
            out[sym] = (ex_d, amt)
    return out


__all__ = [
    "DEFAULT_NEWS_DB",
    "has_credentials",
    "parse_option_snapshots",
    "fetch_option_chain_snapshots",
    "fetch_option_bars",
    "ensure_news_db",
    "fetch_news",
    "save_news_archive",
    "fetch_corporate_actions",
    "upcoming_ex_dividend_dates",
]
