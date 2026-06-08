"""Automation health / staleness checker.

The scheduled jobs (auto-log, exit-enforcer, weekly checkpoint) run via cron.
When cron silently stops — as it did ~2026-05-20 when it lost Full Disk Access
— the system keeps looking fine while the cohort quietly stops growing and
positions stop closing. This module surfaces that at startup so a silent death
can't go unnoticed for days.

Freshness is inferred from artifacts the jobs already produce, so it works
immediately (no new heartbeat plumbing required) and degrades gracefully:
a missing artifact is reported as "no activity recorded", never an exception.
"""
from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional


@dataclass
class Check:
    label: str
    last: Optional[datetime]      # last known activity, or None if never seen
    max_age_days: float           # warn once activity is older than this
    hint: str = ""                # remediation shown when stale/missing


def stale_warnings(checks: List[Check], now: datetime) -> List[str]:
    """Pure: turn a list of Checks into human warning strings. A Check with
    last=None warns as 'no activity recorded'; otherwise it warns only when the
    activity is strictly older than max_age_days."""
    out: List[str] = []
    for c in checks:
        if c.last is None:
            msg = f"{c.label}: no activity recorded."
            if c.hint:
                msg += f" {c.hint}"
            out.append(msg)
            continue
        age_days = (now - c.last).total_seconds() / 86400.0
        if age_days > c.max_age_days:
            msg = (f"{c.label}: last ran {age_days:.0f}d ago "
                   f"(expected within {c.max_age_days:.0f}d).")
            if c.hint:
                msg += f" {c.hint}"
            out.append(msg)
    return out


def _file_mtime(path: str) -> Optional[datetime]:
    try:
        return datetime.fromtimestamp(os.path.getmtime(path))
    except OSError:
        return None


def _last_db_trade_date(db_path: str) -> Optional[datetime]:
    """Most recent entry date in the trades table — the truest signal that
    logging is happening at all (cron or manual)."""
    if not os.path.exists(db_path):
        return None
    try:
        conn = sqlite3.connect(db_path)
        try:
            row = conn.execute("SELECT MAX(date) FROM trades").fetchone()
        finally:
            conn.close()
    except sqlite3.Error:
        return None
    if not row or not row[0]:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(row[0])[:19] if " " in str(row[0]) else str(row[0])[:10], fmt)
        except ValueError:
            continue
    return None


def _last_tsv_date(path: str) -> Optional[datetime]:
    """Date in the first column of the last data row of a TSV (e.g.
    reports/checkpoint_history.tsv: 'date\\tweeks\\tn\\t...')."""
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            lines = [ln for ln in f.read().splitlines() if ln.strip()]
    except OSError:
        return None
    if len(lines) < 2:
        return None
    first_col = lines[-1].split("\t", 1)[0].strip()
    try:
        return datetime.strptime(first_col[:10], "%Y-%m-%d")
    except ValueError:
        return None


def collect_checks(root: str = ".", db_path: str = "paper_trades.db",
                   now: Optional[datetime] = None) -> List[Check]:
    """Build the automation checks from existing artifacts. Never raises."""
    now = now or datetime.now()

    def _p(*parts: str) -> str:
        return os.path.join(root, *parts)

    # Auto-log: freshest of the DB's last trade date and the cron log mtime.
    autolog_last = _last_db_trade_date(db_path)
    log_mtime = _file_mtime(_p("logs", "auto_log_equity.log"))
    if log_mtime and (autolog_last is None or log_mtime > autolog_last):
        autolog_last = log_mtime

    return [
        Check("auto-log", autolog_last, max_age_days=4,
              hint="run the screener on a weekday during market hours so "
                   "startup auto-log fires (cron retired 2026-06-07)."),
        Check("exit-enforcer", _file_mtime(_p("logs", "enforce_exits.log")),
              max_age_days=4,
              hint="open positions may not be closing — run the screener so "
                   "startup exit-enforcement fires."),
        Check("weekly checkpoint",
              _last_tsv_date(_p("reports", "checkpoint_history.tsv")),
              max_age_days=9,
              hint="gate decision is not refreshing — run the screener "
                   "(checkpoint refreshes at startup once ≥7 days old)."),
    ]


def automation_health_warnings(root: str = ".", db_path: str = "paper_trades.db",
                               now: Optional[datetime] = None) -> List[str]:
    """Top-level convenience: collected checks → warning strings. Failure-safe."""
    try:
        return stale_warnings(collect_checks(root=root, db_path=db_path, now=now), now or datetime.now())
    except Exception:
        return []
