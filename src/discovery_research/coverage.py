"""Read-only, per-symbol coverage. Missing observations are never neutral signals."""
from __future__ import annotations

import json
import math
import sqlite3
from contextlib import closing
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from src.breakout.refresh import completed_session
from src.data_quality import holiday_calendar_covers, is_trading_day
from src.desk_kit import shell


def _readonly(path: str) -> sqlite3.Connection:
    return sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)


def price_coverage(path: str, symbols: list[str], start: str,
                   now: Optional[datetime] = None) -> dict:
    expected = completed_session(now)
    report: dict = {"source": path, "expected_session": expected, "start": start,
                    "calendar_status": "known" if expected else "unknown holiday year",
                    "reference": "Exact SPY dates plus maintained-year US sessions; no nearest-date matching or forward fill",
                    "symbols": [], "status": "unknown"}
    try:
        with closing(_readonly(path)) as con:
            tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            columns = {r[1] for r in con.execute("PRAGMA table_info(ohlcv)")}
            if not {"ticker", "date", "close"}.issubset(columns):
                raise ValueError("Missing price schema")
            rows = con.execute("SELECT ticker,date,close FROM ohlcv WHERE date>=? ORDER BY date", (start,)).fetchall()
            selected = set(symbols) | {"SPY"}
            by_symbol: dict[str, dict] = {}
            invalid: dict[str, int] = {}
            for sym, day, value in rows:
                if sym not in selected:
                    continue
                try:
                    good = value is not None and math.isfinite(float(value)) and float(value) > 0
                except (TypeError, ValueError):
                    good = False
                if good:
                    by_symbol.setdefault(sym, {})[day] = float(value)
                else:
                    invalid[sym] = invalid.get(sym, 0) + 1
            spy_dates = set(by_symbol.get("SPY", {}))
            expected_dates = set(spy_dates)
            if expected:
                day = date.fromisoformat(start)
                while day.isoformat() <= expected:
                    if holiday_calendar_covers(day.year) and is_trading_day(day):
                        expected_dates.add(day.isoformat())
                    day += timedelta(days=1)
            report["reference_last"] = max(spy_dates) if spy_dates else None
            for sym in dict.fromkeys(symbols):
                prices = by_symbol.get(sym, {})
                dates = sorted(prices)
                first, last = (dates[0], dates[-1]) if dates else (None, None)
                missing = sorted(d for d in expected_dates if first and d >= first and d not in prices)
                unexpected = sorted(set(dates) - spy_dates) if spy_dates else []
                jumps = [{"date": b, "return": prices[b] / prices[a] - 1}
                         for a, b in zip(dates, dates[1:]) if abs(prices[b] / prices[a] - 1) > .45]
                state = (con.execute("SELECT source,adjustment,last_success,last_full FROM price_state WHERE ticker=?", (sym,)).fetchone()
                         if "price_state" in tables else None)
                attempt = (con.execute("SELECT fetched_at,accepted,error FROM price_fetches WHERE ticker=? ORDER BY id DESC LIMIT 1", (sym,)).fetchone()
                           if "price_fetches" in tables else None)
                evidence_rows = (con.execute("SELECT COUNT(*) FROM ohlcv_evidence WHERE ticker=? AND date>=?", (sym, start)).fetchone()[0]
                                 if "ohlcv_evidence" in tables else 0)
                if not dates:
                    status = "missing"
                elif expected is None or not spy_dates:
                    status = "unknown"
                elif last is not None and last < expected:
                    status = "stale"
                elif last is not None and last > expected:
                    status = "unsettled"
                elif missing or unexpected or invalid.get(sym) or jumps:
                    status = "quality_review"
                elif not state or evidence_rows != len(prices):
                    status = "unverified"
                else:
                    status = "current"
                report["symbols"].append({"symbol": sym, "first": first, "last": last, "rows": len(prices),
                    "status": status, "invalid_prices": invalid.get(sym, 0), "missing_spy_dates": missing,
                    "dates_outside_spy": unexpected, "large_moves": jumps,
                    "source": state[0] if state else None, "adjustment": state[1] if state else None,
                    "evidence_rows": evidence_rows, "last_success": state[2] if state else None,
                    "last_full_refresh": state[3] if state else None,
                    "last_attempt": {"at": attempt[0], "accepted": bool(attempt[1]), "error": attempt[2]} if attempt else None})
        report["status"] = "current" if report["symbols"] and all(r["status"] == "current" for r in report["symbols"]) else "incomplete"
    except (OSError, sqlite3.Error, ValueError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def event_coverage(path: str, symbols: list[str], *, analysts: bool = False) -> dict:
    table, clock = ("analyst_actions", "first_seen") if analysts else ("news_archive", "archived_at")
    report: dict = {"source": path, "availability": clock, "status": "unknown", "symbols": []}
    try:
        with closing(_readonly(path)) as con:
            for sym in symbols:
                count, first, last, missing = con.execute(
                    f"SELECT COUNT(*),MIN({clock}),MAX({clock}),SUM({clock} IS NULL OR {clock}='') "
                    f"FROM {table} WHERE symbol=?", (sym,)).fetchone()
                report["symbols"].append({"symbol": sym, "rows": count, "first_seen": first,
                    "last_seen": last, "missing_availability": missing or 0,
                    "status": "observed" if count and not missing else "unknown"})
            report["status"] = "observations only; absence does not prove a successful empty fetch"
            if analysts:
                count, first, last = con.execute(
                    "SELECT COUNT(*),MIN(archived_at),MAX(archived_at) FROM estimate_snapshots").fetchone()
                report["estimate_snapshots"] = {"rows": count, "first_seen": first, "last_seen": last}
    except (OSError, sqlite3.Error) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    return report


def snapshot(prices: str, news: str, analysts: str, symbols: list[str], start: str,
             now: Optional[datetime] = None) -> dict:
    return {"schema": 1, "generated_at": (now or datetime.now(timezone.utc)).isoformat(),
            "prices": price_coverage(prices, symbols, start, now),
            "news": event_coverage(news, symbols), "analysts": event_coverage(analysts, symbols, analysts=True),
            "interpretation": "Coverage and collection evidence only. No new hypothesis evaluation or trading changes."}


def price_table(report: dict) -> str:
    esc = shell.esc
    out = '<p>Expected completed session: <strong>{}</strong>. {}. Calendar: {}.</p>'.format(
        esc(report.get("expected_session") or "unknown"), esc(report.get("reference", "")), esc(report.get("calendar_status", "unknown")))
    if report.get("error"):
        out += '<p>{}</p>'.format(esc(report["error"]))
    out += '<div class="table-scroll"><table><thead><tr><th>Symbol</th><th>State</th><th>First / last</th><th>Gaps / invalid / large moves</th><th>Source evidence</th></tr></thead><tbody>'
    for row in report.get("symbols", []):
        attempt = row.get("last_attempt") or {}
        out += '<tr><td>{}</td><td>{}</td><td>{} / {}</td><td>{} / {} / {}</td><td>{}</td></tr>'.format(
            esc(row["symbol"]), esc(row["status"]), esc(row["first"] or "unknown"), esc(row["last"] or "unknown"),
            len(row["missing_spy_dates"]), row["invalid_prices"], len(row["large_moves"]),
            esc(attempt.get("error") or (row.get("source") if row["evidence_rows"] == row["rows"] and row["rows"] else None) or "Legacy source / adjustment unverified"))
    out += '</tbody></table></div><p>Gaps count exact SPY sessions plus maintained-year US sessions from each symbol’s first observation, including missing trailing sessions. '
    out += 'Earlier listing/history coverage remains unknown. Large moves are flagged, never repaired by guessing a split.</p>'
    return out


def write_report(data: dict, output_dir: str) -> Path:
    esc = shell.esc
    body = '<div class="grid">' + shell.card("Price coverage and freshness", price_table(data["prices"]), span=12) + '</div>'
    flagged = [(sym, move) for sym, info in (data.get("collection") or {}).get("symbols", {}).items()
               for move in info.get("large_moves", [])]
    if flagged:
        content = '<p>Large moves anywhere in the full archive, including before the research window. '
        content += 'These remain observations to review; collection never infers a split from their size.</p><ul>'
        content += ''.join('<li>{}: {} ({:+.2%})</li>'.format(esc(sym), esc(move["date"]), move["return"]) for sym, move in flagged)
        body += '<div class="grid">' + shell.card("Full-history review flags", content + '</ul>', span=12) + '</div>'
    for name in ("news", "analysts"):
        channel = data[name]
        content = '<p>{}. Availability: {}. {}</p>'.format(esc(channel["status"]), esc(channel["availability"]), esc(channel.get("error", "")))
        content += '<table><thead><tr><th>Symbol</th><th>Observed rows</th><th>First / last seen</th><th>State</th></tr></thead><tbody>'
        for row in channel["symbols"]:
            content += '<tr><td>{}</td><td>{}</td><td>{} / {}</td><td>{}</td></tr>'.format(
                esc(row["symbol"]), row["rows"], esc(row["first_seen"] or "unknown"), esc(row["last_seen"] or "unknown"), esc(row["status"]))
        content += '</tbody></table>'
        body += '<div class="grid">' + shell.card(name.title() + " collection evidence", content, span=12) + '</div>'
    actions = ('<p>Run <code>python -m src.discovery_research --update-prices --no-open</code> to retry price collection. '
               'The existing startup/launchd maintenance also collects after 18:00 ET and retries incomplete runs. '
               'A source or adjustment revision triggers a full refresh, retaining previous values and compressed fetch evidence in the same database.</p>'
               '<p>For quality_review, inspect missing_spy_dates, dates_outside_spy and large_moves in coverage.json and the stored action evidence. '
               'For missing news/analysts, collect forward; historical publication/grade dates cannot establish first availability. '
               'A stale last event is not proof that its collector failed or that no new events occurred.</p>'
               '<p><a href="latest.html">Saved hypothesis findings</a> retain their own dated input window. Refreshing coverage does not rerun them.</p>')
    body += '<div class="grid">' + shell.card("Follow-up actions", actions, span=12) + '</div>'
    masthead = shell.masthead("DISCOVERY DATA COVERAGE", "", meta_html=esc(data["generated_at"]), where="research")
    html = shell.page("Discovery data coverage", masthead, body,
                      extra_css='.table-scroll{overflow-x:auto}td{white-space:normal;overflow-wrap:anywhere}p{line-height:1.65}')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    stamp = data["generated_at"][:19].replace("-", "").replace(":", "").replace("T", "_")
    for base in ("coverage_" + stamp, "coverage"):
        (output / (base + ".json")).write_text(encoded)
        (output / (base + ".html")).write_text(html)
    return output / "coverage.html"
