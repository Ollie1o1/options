"""SEC EDGAR Forced-Flow Corporate Events & XBRL Quality Scanner (`src/alpha/edgar_events.py`).

100% free, keyless, point-in-time SEC EDGAR feeds tailored for low-turnover Wealthsimple TFSA shares:
  1. Form 10 / 10-12B Spinoffs   — institutional forced-selling opportunities.
  2. Schedule 13D Activist Stakes — filed within 5 business days (unlike stale 45d 13Fs).
  3. XBRL 10-Q/10-K Net Buyback Yield + Gross Profitability (GP/A) — true net share
     count contraction (automatically net of stock-based compensation dilution!).

Storage: local SQLite `data/edgar_events.db` (zero network during daily `opt` / Web reads;
refresh on-demand via `opt alpha edgar --refresh`).
Evidence label: WATCHLIST (structural literature; pre-register OOS before allocating size).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import urllib.parse
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence

from src.paths import repo_path

DEFAULT_EDGAR_DB = "data/edgar_events.db"
EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS edgar_events (
    event_id TEXT PRIMARY KEY,
    event_type TEXT NOT NULL,   -- SPINOFF_FORM10 | ACTIVIST_13D | NET_BUYBACK_QUALITY
    ticker TEXT NOT NULL,
    entity_name TEXT,
    filed_date TEXT NOT NULL,
    summary TEXT NOT NULL,
    metric_value REAL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_edgar_type_date ON edgar_events (event_type, filed_date DESC);
"""


@dataclass(frozen=True)
class EdgarEvent:
    event_id: str
    event_type: str        # "SPINOFF_FORM10" | "ACTIVIST_13D" | "NET_BUYBACK_QUALITY"
    ticker: str
    entity_name: str
    filed_date: str
    summary: str
    metric_value: Optional[float]
    source: str
    fetched_at: str
    evidence_label: str = "WATCHLIST"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def ensure_edgar_db(db_path: str = DEFAULT_EDGAR_DB) -> str:
    full = repo_path(db_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    con = sqlite3.connect(full)
    try:
        con.executescript(_SCHEMA)
        con.commit()
    finally:
        con.close()
    return full


def _edgar_get_json(url: str, timeout: int = 20) -> Dict[str, Any]:
    from src.insider.edgar import _get

    resp = _get(url, timeout=timeout)
    data = resp.json()
    return data if isinstance(data, dict) else {}


def parse_efts_hits(
    payload: Dict[str, Any],
    *,
    event_type: str,
    source_label: str,
    fetched_at: Optional[str] = None,
) -> List[EdgarEvent]:
    """Parse SEC EDGAR EFTS `hits.hits` response into normalized `EdgarEvent` rows."""
    now_ts = fetched_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hits = ((payload or {}).get("hits") or {}).get("hits") or []
    out: List[EdgarEvent] = []
    seen = set()
    for h in hits:
        src = h.get("_source") or {}
        adsh = str(src.get("adsh") or h.get("_id") or "")
        filed = str(src.get("file_date") or "")[:10]
        form = str(src.get("form") or "")
        names = src.get("display_names") or []
        entity = str(names[0]) if names else ""
        # Extract ticker if display_names looks like "COMPANY INC  (TICK)  (CIK ...)"
        ticker = ""
        if "(" in entity and ")" in entity:
            for chunk in entity.split("(")[1:]:
                cand = chunk.split(")")[0].strip().upper()
                if cand.isalpha() and 1 <= len(cand) <= 5 and not cand.startswith("CIK"):
                    ticker = cand
                    break
        # An entity name without a listed symbol remains an unresolved event.
        eid = f"{event_type}:{adsh or ((ticker or entity) + ':' + filed)}"
        if eid in seen or not filed:
            continue
        seen.add(eid)
        clean_ent = entity.split("(CIK")[0].strip() if "(CIK" in entity else entity
        summary = f"{form} · {clean_ent[:72]}"
        out.append(
            EdgarEvent(
                event_id=eid,
                event_type=event_type,
                ticker=ticker,
                entity_name=clean_ent[:80],
                filed_date=filed,
                summary=summary,
                metric_value=None,
                source=source_label,
                fetched_at=now_ts,
            )
        )
    return out


def compute_xbrl_buyback_quality(
    ticker: str,
    facts_payload: Dict[str, Any],
    *,
    fetched_at: Optional[str] = None,
) -> Optional[EdgarEvent]:
    """Compute true YoY Net Share Count Reduction % & Gross Profitability (GP/A) from SEC XBRL facts."""
    now_ts = fetched_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    facts = (facts_payload or {}).get("facts") or {}
    dei = facts.get("dei") or {}
    gaap = facts.get("us-gaap") or {}

    # 1. Net share count series (prefer DEI EntityCommonStockSharesOutstanding, else us-gaap)
    share_pts: List[tuple[str, float]] = []
    for ns, tag in ((dei, "EntityCommonStockSharesOutstanding"), (gaap, "CommonStockSharesOutstanding")):
        units = (ns.get(tag) or {}).get("units") or {}
        for row in units.get("shares") or []:
            d = str(row.get("end") or "")[:10]
            v = row.get("val")
            if d and isinstance(v, (int, float)) and v > 0:
                share_pts.append((d, float(v)))
        if share_pts:
            break

    if len(share_pts) < 2:
        return None
    # Deduplicate by date keeping latest observation
    by_d: Dict[str, float] = {}
    for d, v in share_pts:
        by_d[d] = v
    ordered = sorted(by_d.items())
    latest_d, latest_sh = ordered[-1]

    # Find observation roughly 1 year prior (270..460 days earlier)
    try:
        d_last = dt.date.fromisoformat(latest_d)
    except ValueError:
        return None
    prior_cands = []
    for d, v in ordered[:-1]:
        try:
            gap = (d_last - dt.date.fromisoformat(d)).days
        except ValueError:
            continue
        if 260 <= gap <= 470:
            prior_cands.append((abs(gap - 365), d, v))
    if not prior_cands:
        return None
    prior_cands.sort(key=lambda x: x[0])
    _gap, prior_d, prior_sh = prior_cands[0]
    net_buyback_pct = round((1.0 - latest_sh / prior_sh) * 100.0, 2)

    # 2. Optional Gross Profitability (GP / Total Assets)
    def _latest_usd(tag: str) -> Optional[float]:
        units = (gaap.get(tag) or {}).get("units") or {}
        rows = [
            (str(r.get("end") or "")[:10], float(r["val"]))
            for r in (units.get("USD") or [])
            if r.get("end") and isinstance(r.get("val"), (int, float))
            and str(r.get("form") or "") in ("10-K", "10-Q")
        ]
        if not rows:
            return None
        rows.sort(key=lambda x: x[0])
        return rows[-1][1]

    gp = _latest_usd("GrossProfit")
    assets = _latest_usd("Assets")
    gpa_pct = round((gp / assets) * 100.0, 1) if (gp is not None and assets and assets > 0) else None
    gpa_s = f" · GP/A {gpa_pct:.1f}%" if gpa_pct is not None else ""
    entity = str(facts_payload.get("entityName") or ticker.upper())

    return EdgarEvent(
        event_id=f"NET_BUYBACK_QUALITY:{ticker.upper()}:{latest_d}",
        event_type="NET_BUYBACK_QUALITY",
        ticker=ticker.upper(),
        entity_name=entity[:80],
        filed_date=latest_d,
        summary=f"Net shares YoY {-net_buyback_pct:+.1f}% (buyback yield {net_buyback_pct:+.1f}%){gpa_s} vs {prior_d}",
        metric_value=net_buyback_pct,
        source="SEC EDGAR XBRL companyfacts",
        fetched_at=now_ts,
    )


def save_edgar_events(events: Sequence[EdgarEvent], db_path: str = DEFAULT_EDGAR_DB) -> int:
    if not events:
        return 0
    full = ensure_edgar_db(db_path)
    con = sqlite3.connect(full)
    try:
        rows = [
            (
                e.event_id,
                e.event_type,
                e.ticker,
                e.entity_name,
                e.filed_date,
                e.summary,
                e.metric_value,
                e.source,
                e.fetched_at,
            )
            for e in events
        ]
        con.executemany(
            "INSERT INTO edgar_events "
            "(event_id, event_type, ticker, entity_name, filed_date, summary, metric_value, source, fetched_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(event_id) DO UPDATE SET ticker=excluded.ticker, entity_name=excluded.entity_name, "
            "filed_date=excluded.filed_date, summary=excluded.summary, metric_value=excluded.metric_value, "
            "source=excluded.source, fetched_at=MIN(edgar_events.fetched_at, excluded.fetched_at)",
            rows,
        )
        con.commit()
        return len(rows)
    finally:
        con.close()


def load_cached_edgar_events(
    *,
    db_path: str = DEFAULT_EDGAR_DB,
    event_type: Optional[str] = None,
    limit: int = 40,
) -> List[EdgarEvent]:
    full = repo_path(db_path)
    if not os.path.exists(full):
        return []
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        if event_type:
            rows = con.execute(
                "SELECT * FROM edgar_events WHERE event_type = ? ORDER BY filed_date DESC LIMIT ?",
                (event_type, int(limit)),
            ).fetchall()
        else:
            rows = con.execute(
                "SELECT * FROM edgar_events ORDER BY filed_date DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        return [
            EdgarEvent(
                event_id=str(r["event_id"]),
                event_type=str(r["event_type"]),
                ticker=str(r["ticker"]),
                entity_name=str(r["entity_name"] or ""),
                filed_date=str(r["filed_date"]),
                summary=str(r["summary"]),
                metric_value=r["metric_value"],
                source=str(r["source"]),
                fetched_at=str(r["fetched_at"]),
            )
            for r in rows
        ]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def refresh_edgar_events(
    *,
    db_path: str = DEFAULT_EDGAR_DB,
    buyback_tickers: Sequence[str] = ("AAPL", "META", "GOOGL", "MSFT", "NVDA", "XOM", "CVX", "JPM", "BAC", "V", "MA", "HD"),
    getter: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fetch fresh SEC Form 10 spinoffs, Schedule 13D activists, and XBRL net buybacks."""
    fetch = getter or _edgar_get_json
    today = dt.date.today()
    start_90 = (today - dt.timedelta(days=90)).isoformat()
    start_21 = (today - dt.timedelta(days=21)).isoformat()
    end_d = today.isoformat()
    all_ev: List[EdgarEvent] = []
    errors: List[str] = []

    # 1. Form 10-12B Spinoffs (last 90 days)
    try:
        q = urllib.parse.urlencode(
            {"forms": "10-12B", "dateRange": "custom", "startdt": start_90, "enddt": end_d}
        )
        payload = fetch(f"{EFTS_SEARCH_URL}?{q}")
        all_ev.extend(parse_efts_hits(payload, event_type="SPINOFF_FORM10", source_label="SEC EFTS Form 10-12B")[:15])
    except Exception as exc:
        errors.append(f"form10: {exc}")

    # 2. Schedule 13D Activist Filings (last 21 days)
    try:
        q = urllib.parse.urlencode(
            {"forms": "SC 13D", "dateRange": "custom", "startdt": start_21, "enddt": end_d}
        )
        payload = fetch(f"{EFTS_SEARCH_URL}?{q}")
        all_ev.extend(parse_efts_hits(payload, event_type="ACTIVIST_13D", source_label="SEC EFTS Schedule 13D")[:15])
    except Exception as exc:
        errors.append(f"sc13d: {exc}")

    # 3. Form 8-K Item 2.02 / 7.01 Guidance & Outlook Updates (last 14 days)
    start_14 = (today - dt.timedelta(days=14)).isoformat()
    try:
        q = urllib.parse.urlencode(
            {
                "q": '"raises guidance" OR "raises full-year" OR "updates outlook" OR "lowers guidance"',
                "forms": "8-K",
                "dateRange": "custom",
                "startdt": start_14,
                "enddt": end_d,
            }
        )
        payload = fetch(f"{EFTS_SEARCH_URL}?{q}")
        all_ev.extend(parse_efts_hits(payload, event_type="GUIDANCE_8K", source_label="SEC EFTS Form 8-K Guidance")[:15])
    except Exception as exc:
        errors.append(f"guidance_8k: {exc}")

    # 4. Form 10-K Major Customer / Supply-Chain Concentration Disclosures (last 60 days)
    start_60 = (today - dt.timedelta(days=60)).isoformat()
    try:
        q = urllib.parse.urlencode(
            {
                "q": '"accounted for more than 10%" AND "revenue"',
                "forms": "10-K",
                "dateRange": "custom",
                "startdt": start_60,
                "enddt": end_d,
            }
        )
        payload = fetch(f"{EFTS_SEARCH_URL}?{q}")
        all_ev.extend(parse_efts_hits(payload, event_type="SUPPLY_CHAIN_10K", source_label="SEC EFTS 10-K Supply Chain")[:10])
    except Exception as exc:
        errors.append(f"supply_chain_10k: {exc}")

    # 5. XBRL Net Share Buyback Yield + GP/A on liquid TFSA universe
    try:
        from src.insider.edgar import cik_for

        for tk in buyback_tickers:
            cik = cik_for(tk)
            if not cik:
                continue
            try:
                facts = fetch(XBRL_FACTS_URL.format(cik=int(cik)))
                ev = compute_xbrl_buyback_quality(tk, facts)
                if ev is not None and (ev.metric_value or 0.0) >= 1.5:
                    all_ev.append(ev)
            except Exception:
                continue
    except Exception as exc:
        errors.append(f"xbrl: {exc}")

    saved = save_edgar_events(all_ev, db_path=db_path)
    return {"saved": saved, "errors": errors, "events": [e.to_dict() for e in all_ev]}
