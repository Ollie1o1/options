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
DEFAULT_FUNDAMENTALS_DB = "data/sec_fundamentals.db"
EFTS_SEARCH_URL = "https://efts.sec.gov/LATEST/search-index"
XBRL_FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"
COMPANYFACTS_ZIP_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"

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

_FUNDAMENTALS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sec_fundamentals (
    ticker TEXT NOT NULL,
    filed_date TEXT NOT NULL,
    period_end TEXT NOT NULL,
    entity_name TEXT,
    net_buyback_pct REAL,
    gross_profit_assets_pct REAL,
    fcf_assets_pct REAL,
    shares_latest REAL,
    shares_prior REAL,
    source TEXT NOT NULL,
    fetched_at TEXT NOT NULL,
    PRIMARY KEY (ticker, period_end)
);
CREATE INDEX IF NOT EXISTS idx_sec_fund_ticker ON sec_fundamentals (ticker, filed_date DESC);
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


def _reverse_cik_map() -> Dict[int, str]:
    """Invert cached SEC `company_tickers.json` (`ticker -> cik`) into `cik -> ticker`."""
    try:
        from src.insider.edgar import _ticker_map

        tmap = _ticker_map()
        rev: Dict[int, str] = {}
        for tk, cik in tmap.items():
            c_int = int(cik)
            # Prefer shorter canonical symbol (e.g. BRK-B over warrants) when multiple share a CIK
            if c_int not in rev or len(tk) < len(rev[c_int]):
                rev[c_int] = tk
        return rev
    except Exception:
        return {}


def parse_efts_hits(
    payload: Dict[str, Any],
    *,
    event_type: str,
    source_label: str,
    fetched_at: Optional[str] = None,
    cik_to_ticker: Optional[Dict[int, str]] = None,
) -> List[EdgarEvent]:
    """Parse SEC EDGAR EFTS `hits.hits` response into normalized `EdgarEvent` rows."""
    import re

    now_ts = fetched_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    hits = ((payload or {}).get("hits") or {}).get("hits") or []
    cik_map = cik_to_ticker if cik_to_ticker is not None else {}
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
        # Fallback: resolve CIK via SEC company_tickers.json when display_names omits ticker
        if not ticker and cik_map:
            ciks_raw = src.get("ciks") or []
            if not ciks_raw:
                m_cik = re.search(r"CIK\s*0*(\d+)", entity, re.IGNORECASE)
                if m_cik:
                    ciks_raw = [m_cik.group(1)]
            for raw_c in ciks_raw:
                try:
                    resolved = cik_map.get(int(str(raw_c).lstrip("0") or "0"))
                except ValueError:
                    resolved = None
                if resolved:
                    ticker = resolved
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


def ensure_fundamentals_db(db_path: str = DEFAULT_FUNDAMENTALS_DB) -> str:
    full = repo_path(db_path)
    os.makedirs(os.path.dirname(full) or ".", exist_ok=True)
    con = sqlite3.connect(full)
    try:
        con.executescript(_FUNDAMENTALS_SCHEMA)
        con.commit()
    finally:
        con.close()
    return full


def _parse_iso_date(s: Any) -> Optional[dt.date]:
    txt = str(s or "")[:10]
    if len(txt) != 10:
        return None
    try:
        return dt.date.fromisoformat(txt)
    except ValueError:
        return None


def extract_xbrl_pit_series(
    ticker: str,
    facts_payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract split-safe, point-in-time (`filed_date`) historical XBRL series for `ticker`.

    Same-filing (`accn`) comparative `WeightedAverageNumberOfDilutedSharesOutstanding`
    restates both current and prior-year share counts on the identical split-adjusted basis,
    eliminating split artifacts (e.g., AAPL 4:1, NVDA 4:1 / 10:1) while automatically
    capturing stock-based compensation (SBC) dilution.
    """
    facts = (facts_payload or {}).get("facts") or {}
    gaap = facts.get("us-gaap") or {}
    entity = str((facts_payload or {}).get("entityName") or ticker.upper())[:80]

    # 1. Split-safe same-filing YoY Net Diluted Buyback Yield (`nby`)
    nby_events: List[Dict[str, Any]] = []
    for tag in (
        "WeightedAverageNumberOfDilutedSharesOutstanding",
        "WeightedAverageNumberOfSharesOutstandingBasic",
    ):
        rows = ((gaap.get(tag) or {}).get("units") or {}).get("shares") or []
        by_acc: Dict[tuple[str, str, str], List[tuple[dt.date, dt.date, float]]] = {}
        for r in rows:
            form = str(r.get("form") or "")
            if form not in ("10-K", "10-Q"):
                continue
            accn = str(r.get("accn") or "")
            filed = str(r.get("filed") or "")[:10]
            ds = _parse_iso_date(r.get("start"))
            de = _parse_iso_date(r.get("end"))
            val = r.get("val")
            if accn and len(filed) == 10 and ds and de and isinstance(val, (int, float)) and val > 0:
                by_acc.setdefault((accn, filed, form), []).append((ds, de, float(val)))
        for (accn, filed, form), pts in by_acc.items():
            # Deduplicate (start, end) inside same filing
            uniq = {(s, e): v for s, e, v in pts}
            entries = []
            for (s, e), v in uniq.items():
                dur = (e - s).days
                if (70 <= dur <= 110) or (330 <= dur <= 400):
                    entries.append((e, dur, v))
            if len(entries) < 2:
                continue
            # Prefer annual period inside 10-K, quarterly (70..110d) inside 10-Q
            pref = [x for x in entries if (x[1] >= 330 if form == "10-K" else x[1] <= 110)]
            cand_pool = pref if pref else entries
            cand_pool.sort(key=lambda x: x[0])
            cur_end, cur_dur, cur_val = cand_pool[-1]
            # Find comparative prior-year period inside the SAME filing (identical split basis)
            comp = [
                (abs((cur_end - e).days - 365), e, v)
                for e, dur, v in entries
                if 330 <= (cur_end - e).days <= 400 and abs(cur_dur - dur) <= 20
            ]
            if not comp:
                continue
            comp.sort(key=lambda x: x[0])
            _, _prev_end, prev_val = comp[0]
            raw_nby = (1.0 - cur_val / prev_val) * 100.0
            nby_pct = round(max(-25.0, min(25.0, raw_nby)), 3)
            nby_events.append(
                {
                    "filed_date": filed,
                    "period_end": cur_end.isoformat(),
                    "form": form,
                    "net_buyback_pct": nby_pct,
                    "shares_latest": cur_val,
                    "shares_prior": prev_val,
                }
            )
        if nby_events:
            break
    nby_events.sort(key=lambda x: (x["filed_date"], x["period_end"]))

    # 2. Point-in-time Total Assets (instant, 10-K or 10-Q)
    assets_events: List[tuple[str, str, float]] = []
    for r in ((gaap.get("Assets") or {}).get("units") or {}).get("USD") or []:
        if str(r.get("form") or "") not in ("10-K", "10-Q"):
            continue
        filed = str(r.get("filed") or "")[:10]
        end_s = str(r.get("end") or "")[:10]
        v = r.get("val")
        if len(filed) == 10 and len(end_s) == 10 and isinstance(v, (int, float)) and v > 0:
            assets_events.append((filed, end_s, float(v)))
    assets_events.sort(key=lambda x: (x[0], x[1]))

    def _annual_usd_map(tags: Sequence[str]) -> Dict[tuple[str, str], tuple[str, float]]:
        out: Dict[tuple[str, str], tuple[str, float]] = {}
        for tag in tags:
            for r in ((gaap.get(tag) or {}).get("units") or {}).get("USD") or []:
                if str(r.get("form") or "") != "10-K":
                    continue
                accn = str(r.get("accn") or "")
                filed = str(r.get("filed") or "")[:10]
                ds = _parse_iso_date(r.get("start"))
                de = _parse_iso_date(r.get("end"))
                v = r.get("val")
                if not (accn and len(filed) == 10 and ds and de and isinstance(v, (int, float))):
                    continue
                dur = (de - ds).days
                if not (330 <= dur <= 400):
                    continue
                key = (accn, de.isoformat())
                if key not in out:
                    out[key] = (filed, float(v))
        return out

    # 3. Point-in-time Annual Gross Profit (`10-K`)
    gp_map = _annual_usd_map(("GrossProfit",))
    if len(gp_map) < 3:
        rev_map = _annual_usd_map(
            (
                "Revenues",
                "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet",
                "SalesRevenueGoodsNet",
            )
        )
        cogs_map = _annual_usd_map(
            ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsAndServiceExcludingDepreciationDepletionAndAmortization")
        )
        for key, (filed, rev) in rev_map.items():
            if key not in gp_map and key in cogs_map:
                gp_map[key] = (filed, rev - cogs_map[key][1])
    gp_events = sorted((filed, end_s, val) for (_accn, end_s), (filed, val) in gp_map.items())

    # 4. Point-in-time Annual Free Cash Flow (`10-K`: OCF - |CapEx|)
    ocf_map = _annual_usd_map(("NetCashProvidedByUsedInOperatingActivities",))
    capex_map = _annual_usd_map(
        (
            "PaymentsToAcquirePropertyPlantAndEquipment",
            "PaymentsToAcquireProductiveAssets",
        )
    )
    fcf_events: List[tuple[str, str, float]] = []
    for key, (filed, ocf) in ocf_map.items():
        capex = abs(capex_map[key][1]) if key in capex_map else 0.0
        fcf_events.append((filed, key[1], ocf - capex))
    fcf_events.sort(key=lambda x: (x[0], x[1]))

    return {
        "ticker": ticker.upper(),
        "entity_name": entity,
        "nby_events": nby_events,
        "assets_events": assets_events,
        "gp_events": gp_events,
        "fcf_events": fcf_events,
    }


def xbrl_metrics_as_of(
    pit_series: Dict[str, Any],
    as_of_date: str,
) -> Optional[Dict[str, Any]]:
    """Evaluate strictly point-in-time (`filed_date <= as_of_date`) Quality & Buyback metrics."""
    as_of_d = _parse_iso_date(as_of_date)
    if as_of_d is None:
        return None
    cutoff = as_of_d.isoformat()

    # Latest NBY filed <= cutoff whose period_end is within 400 days of as_of_date
    nby_rec = None
    for ev in pit_series.get("nby_events") or ():
        if ev["filed_date"] <= cutoff:
            pe = _parse_iso_date(ev["period_end"])
            if pe and 0 <= (as_of_d - pe).days <= 400:
                if nby_rec is None or (ev["period_end"], ev["filed_date"]) >= (nby_rec["period_end"], nby_rec["filed_date"]):
                    nby_rec = ev

    def _latest_ev(events: Sequence[tuple[str, str, float]], max_age_days: int) -> Optional[tuple[str, str, float]]:
        best = None
        for filed, end_s, val in events:
            if filed <= cutoff:
                pe = _parse_iso_date(end_s)
                if pe and 0 <= (as_of_d - pe).days <= max_age_days:
                    if best is None or (end_s, filed) >= (best[1], best[0]):
                        best = (filed, end_s, val)
        return best

    assets_ev = _latest_ev(pit_series.get("assets_events") or (), 400)
    gp_ev = _latest_ev(pit_series.get("gp_events") or (), 520)
    fcf_ev = _latest_ev(pit_series.get("fcf_events") or (), 520)

    assets = assets_ev[2] if assets_ev else None
    gpa_pct = round((gp_ev[2] / assets) * 100.0, 2) if (gp_ev is not None and assets and assets > 0) else None
    fcf_pct = round((fcf_ev[2] / assets) * 100.0, 2) if (fcf_ev is not None and assets and assets > 0) else None
    nby_pct = float(nby_rec["net_buyback_pct"]) if nby_rec is not None else None

    if nby_pct is None and gpa_pct is None and fcf_pct is None:
        return None
    filed_dates = [x for x in (nby_rec["filed_date"] if nby_rec else None, assets_ev[0] if assets_ev else None, gp_ev[0] if gp_ev else None) if x]
    period_ends = [x for x in (nby_rec["period_end"] if nby_rec else None, assets_ev[1] if assets_ev else None) if x]
    return {
        "ticker": pit_series["ticker"],
        "entity_name": pit_series["entity_name"],
        "as_of_date": cutoff,
        "filed_date": max(filed_dates) if filed_dates else cutoff,
        "period_end": max(period_ends) if period_ends else cutoff,
        "net_buyback_pct": nby_pct,
        "gross_profit_assets_pct": gpa_pct,
        "fcf_assets_pct": fcf_pct,
        "shares_latest": nby_rec["shares_latest"] if nby_rec else None,
        "shares_prior": nby_rec["shares_prior"] if nby_rec else None,
    }


def compute_xbrl_quality_metrics(
    ticker: str,
    facts_payload: Dict[str, Any],
    *,
    fetched_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Compute Point-in-Time (`filed` date) Net Share Buyback Yield, Novy-Marx GP/A & FCF/Assets.

    Prefers split-safe same-filing (`accn`) comparative diluted share counts when available,
    falling back cleanly to cover-page `EntityCommonStockSharesOutstanding`.
    """
    now_ts = fetched_at or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    facts = (facts_payload or {}).get("facts") or {}
    dei = facts.get("dei") or {}
    gaap = facts.get("us-gaap") or {}

    pit = extract_xbrl_pit_series(ticker, facts_payload)
    nby_events = pit["nby_events"]

    latest_d = ""
    latest_filed = ""
    latest_sh: Optional[float] = None
    prior_sh: Optional[float] = None
    net_buyback_pct: Optional[float] = None

    if nby_events:
        best = max(nby_events, key=lambda x: (x["period_end"], x["filed_date"]))
        latest_d = best["period_end"]
        latest_filed = best["filed_date"]
        latest_sh = best["shares_latest"]
        prior_sh = best["shares_prior"]
        net_buyback_pct = round(float(best["net_buyback_pct"]), 2)
    else:
        share_pts: List[tuple[str, str, float]] = []
        for ns, tag in ((dei, "EntityCommonStockSharesOutstanding"), (gaap, "CommonStockSharesOutstanding")):
            units = (ns.get(tag) or {}).get("units") or {}
            for row in units.get("shares") or []:
                d = str(row.get("end") or "")[:10]
                filed = str(row.get("filed") or d)[:10]
                v = row.get("val")
                if d and isinstance(v, (int, float)) and v > 0:
                    share_pts.append((d, filed, float(v)))
            if share_pts:
                break
        if len(share_pts) < 2:
            return None
        by_d: Dict[str, tuple[str, float]] = {}
        for d, filed, v in share_pts:
            by_d[d] = (filed, v)
        ordered = sorted(by_d.items())
        latest_d, (latest_filed, latest_sh) = ordered[-1]
        try:
            d_last = dt.date.fromisoformat(latest_d)
        except ValueError:
            return None
        prior_cands = []
        for d, (_f, v) in ordered[:-1]:
            try:
                gap = (d_last - dt.date.fromisoformat(d)).days
            except ValueError:
                continue
            if 260 <= gap <= 470:
                prior_cands.append((abs(gap - 365), d, v))
        if not prior_cands:
            return None
        prior_cands.sort(key=lambda x: x[0])
        _gap, _prior_d, prior_sh = prior_cands[0]
        net_buyback_pct = round((1.0 - latest_sh / prior_sh) * 100.0, 2)

    def _latest_usd(tags: Sequence[str], *, annual_only: bool = False) -> Optional[float]:
        for tag in tags:
            units = (gaap.get(tag) or {}).get("units") or {}
            rows = []
            for r in units.get("USD") or []:
                form = str(r.get("form") or "")
                if annual_only and form != "10-K":
                    continue
                if not annual_only and form not in ("10-K", "10-Q"):
                    continue
                end_d = str(r.get("end") or "")[:10]
                v = r.get("val")
                if end_d and isinstance(v, (int, float)):
                    rows.append((end_d, float(v)))
            if rows:
                rows.sort(key=lambda x: x[0])
                return rows[-1][1]
        return None

    # Prefer duration-verified annual 10-K GP & FCF from `extract_xbrl_pit_series`
    # (prevents mixing quarterly COGS with annual revenue), falling back to legacy `_latest_usd`
    # for minimal/synthetic XBRL payloads without `start` dates.
    assets = pit["assets_events"][-1][2] if pit["assets_events"] else _latest_usd(("Assets",))
    gp: Optional[float] = None
    if pit["gp_events"]:
        gp = max(pit["gp_events"], key=lambda x: (x[1], x[0]))[2]
    else:
        gp = _latest_usd(("GrossProfit",), annual_only=True) or _latest_usd(("GrossProfit",))
        if gp is None:
            rev = _latest_usd(("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet"), annual_only=True)
            cogs = _latest_usd(("CostOfRevenue", "CostOfGoodsAndServicesSold"), annual_only=True)
            if rev is not None and cogs is not None:
                gp = rev - cogs

    fcf_val: Optional[float] = None
    if pit["fcf_events"]:
        fcf_val = max(pit["fcf_events"], key=lambda x: (x[1], x[0]))[2]
    else:
        ocf = _latest_usd(("NetCashProvidedByUsedInOperatingActivities",), annual_only=True)
        capex = _latest_usd(("PaymentsToAcquirePropertyPlantAndEquipment",), annual_only=True) or 0.0
        if ocf is not None:
            fcf_val = ocf - abs(capex)

    gpa_pct = round((gp / assets) * 100.0, 2) if (gp is not None and assets and assets > 0) else None
    fcf_pct = round((fcf_val / assets) * 100.0, 2) if (fcf_val is not None and assets and assets > 0) else None
    entity = str(facts_payload.get("entityName") or ticker.upper())

    return {
        "ticker": ticker.upper(),
        "filed_date": latest_filed,
        "period_end": latest_d,
        "entity_name": entity[:80],
        "net_buyback_pct": net_buyback_pct,
        "gross_profit_assets_pct": gpa_pct,
        "fcf_assets_pct": fcf_pct,
        "shares_latest": latest_sh,
        "shares_prior": prior_sh,
        "source": "SEC EDGAR XBRL companyfacts",
        "fetched_at": now_ts,
    }


def save_fundamentals_rows(rows: Sequence[Dict[str, Any]], db_path: str = DEFAULT_FUNDAMENTALS_DB) -> int:
    if not rows:
        return 0
    full = ensure_fundamentals_db(db_path)
    con = sqlite3.connect(full)
    try:
        con.executemany(
            """
            INSERT OR REPLACE INTO sec_fundamentals
            (ticker, filed_date, period_end, entity_name, net_buyback_pct,
             gross_profit_assets_pct, fcf_assets_pct, shares_latest, shares_prior, source, fetched_at)
            VALUES (:ticker, :filed_date, :period_end, :entity_name, :net_buyback_pct,
                    :gross_profit_assets_pct, :fcf_assets_pct, :shares_latest, :shares_prior, :source, :fetched_at)
            """,
            rows,
        )
        con.commit()
        return len(rows)
    finally:
        con.close()


def refresh_edgar_fundamentals(
    *,
    tickers: Optional[Sequence[str]] = None,
    zip_path: Optional[str] = None,
    limit: int = 60,
    db_path: str = DEFAULT_FUNDAMENTALS_DB,
    getter: Optional[Callable[[str], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Populate `data/sec_fundamentals.db` from a local SEC `companyfacts.zip` or per-ticker XBRL API."""
    rows: List[Dict[str, Any]] = []
    errors: List[str] = []

    if zip_path and os.path.exists(zip_path):
        import zipfile

        cik_to_tk = _reverse_cik_map()
        want = {t.upper() for t in tickers} if tickers else None
        with zipfile.ZipFile(zip_path, "r") as zf:
            for name in zf.namelist():
                if not name.upper().startswith("CIK") or not name.endswith(".json"):
                    continue
                try:
                    cik_i = int(name[3:-5])
                except ValueError:
                    continue
                tk = cik_to_tk.get(cik_i)
                if not tk or (want is not None and tk not in want):
                    continue
                try:
                    payload = json.loads(zf.read(name).decode("utf-8", errors="replace"))
                    rec = compute_xbrl_quality_metrics(tk, payload)
                    if rec is not None:
                        rows.append(rec)
                except Exception:
                    continue
    else:
        from src.insider.edgar import cik_for

        fetch = getter or _edgar_get_json
        if tickers is None:
            from src.equity_playbooks.universe import LIQUID_MEGA_LARGE_CAPS

            target_list = list(LIQUID_MEGA_LARGE_CAPS[: max(1, int(limit))])
        else:
            target_list = [str(t).upper().strip() for t in tickers if t][: max(1, int(limit))]
        for tk in target_list:
            cik = cik_for(tk)
            if not cik:
                continue
            try:
                facts = fetch(XBRL_FACTS_URL.format(cik=int(cik)))
                rec = compute_xbrl_quality_metrics(tk, facts)
                if rec is not None:
                    rows.append(rec)
            except Exception as exc:
                errors.append(f"{tk}: {str(exc)[:60]}")

    saved = save_fundamentals_rows(rows, db_path=db_path)
    return {"saved": saved, "errors": errors[:10], "db_path": db_path}


def load_quality_buyback_leaders(
    *,
    db_path: str = DEFAULT_FUNDAMENTALS_DB,
    min_buyback_pct: float = 0.5,
    limit: int = 30,
) -> List[Dict[str, Any]]:
    """Rank cached SEC XBRL issuers by Net Share Buyback Yield + Novy-Marx Gross Profitability (GP/A)."""
    full = repo_path(db_path)
    if not os.path.exists(full):
        return []
    con = sqlite3.connect(f"file:{full}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        rows = con.execute(
            """
            SELECT f.* FROM sec_fundamentals f
            INNER JOIN (SELECT ticker, MAX(period_end) AS mx FROM sec_fundamentals GROUP BY ticker) m
            ON f.ticker = m.ticker AND f.period_end = m.mx
            WHERE COALESCE(f.net_buyback_pct, -999) >= ?
            """,
            (float(min_buyback_pct),),
        ).fetchall()
        out = [dict(r) for r in rows]
    except sqlite3.Error:
        return []
    finally:
        con.close()

    # Composite display rank: net buyback yield + 0.15 * GP/A (transparent, descriptive WATCHLIST)
    out.sort(
        key=lambda r: -(
            float(r.get("net_buyback_pct") or 0.0)
            + 0.15 * float(r.get("gross_profit_assets_pct") or 0.0)
            + 0.10 * float(r.get("fcf_assets_pct") or 0.0)
        )
    )
    return out[:limit]


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
    cik_to_tk = _reverse_cik_map() if getter is None else {}

    # 1. Form 10-12B Spinoffs (last 90 days)
    try:
        q = urllib.parse.urlencode(
            {"forms": "10-12B", "dateRange": "custom", "startdt": start_90, "enddt": end_d}
        )
        payload = fetch(f"{EFTS_SEARCH_URL}?{q}")
        all_ev.extend(
            parse_efts_hits(
                payload, event_type="SPINOFF_FORM10", source_label="SEC EFTS Form 10-12B", cik_to_ticker=cik_to_tk
            )[:15]
        )
    except Exception as exc:
        errors.append(f"form10: {exc}")

    # 2. Schedule 13D Activist Filings (last 21 days)
    try:
        q = urllib.parse.urlencode(
            {"forms": "SC 13D", "dateRange": "custom", "startdt": start_21, "enddt": end_d}
        )
        payload = fetch(f"{EFTS_SEARCH_URL}?{q}")
        all_ev.extend(
            parse_efts_hits(
                payload, event_type="ACTIVIST_13D", source_label="SEC EFTS Schedule 13D", cik_to_ticker=cik_to_tk
            )[:15]
        )
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
        all_ev.extend(
            parse_efts_hits(
                payload, event_type="GUIDANCE_8K", source_label="SEC EFTS Form 8-K Guidance", cik_to_ticker=cik_to_tk
            )[:15]
        )
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
        all_ev.extend(
            parse_efts_hits(
                payload, event_type="SUPPLY_CHAIN_10K", source_label="SEC EFTS 10-K Supply Chain", cik_to_ticker=cik_to_tk
            )[:10]
        )
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
