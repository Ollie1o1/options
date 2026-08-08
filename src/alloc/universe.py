"""The backtest universe and its coverage audit.

The audit answers one question before any backtesting happens: is each stratum
still viable? BROAD is the control group — if it is mostly empty, an "it
generalises" conclusion would be unsupported, so this is allowed to veto.

Known hazards it checks for, all observed in the real data:
  * symbols absent from the dataset entirely (QQQ, IWM, GLD and TLT are absent
    from DoltHub; a strategy naming them silently produces zero trades)
  * dates where NO symbol reports data — a source outage, which must never be
    read as "no opportunity that day"
  * tickers that terminate mid-sample through corporate actions (FB -> META
    2022-06-03, PBCT acquired 2022-04-01, WLTW renamed 2022-01-07). Their
    terminal date is reported so the engine can close open positions rather
    than silently dropping them.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from typing import Any, Dict, List, Optional

from src.dolt_options import READ_TIMEOUT_S

MIN_USABLE_DAYS = 100


def load_universe(path: str) -> Dict[str, List[str]]:
    """Read the stratum -> symbols mapping written by the universe builder."""
    with open(path) as f:
        return json.load(f)["strata"]


def symbol_stratum(universe: Dict[str, List[str]]) -> Dict[str, str]:
    """Invert the mapping: symbol -> stratum name."""
    return {s: name for name, syms in universe.items() for s in syms}


def all_symbols(universe: Dict[str, List[str]]) -> List[str]:
    return [s for syms in universe.values() for s in syms]


def audit_coverage(db_path: str, universe: Dict[str, List[str]],
                   min_usable_days: int = MIN_USABLE_DAYS) -> Dict[str, Any]:
    """Per-symbol and per-stratum coverage.

    Never raises on a missing symbol or a missing database — an audit that
    crashes tells you nothing, and its whole job is to report bad news.

    But it must not INVENT bad news either. A cache it cannot READ is a
    different fact from a cache that is empty, and conflating them was observed
    live: while a backfill held the write lock this returned every symbol as
    ABSENT, and the CLI printed "Universe is not viable — refusing to backtest
    on it", which reads as "your data is gone" rather than "the database is
    busy". On a read failure the states are `UNKNOWN`, `read_error` carries the
    reason, and `viable` is still False — declining to backtest is right; only
    the reason was wrong.
    """
    in_universe = symbol_stratum(universe)
    rows: List[tuple] = []
    read_error: Optional[str] = None
    try:
        conn = sqlite3.connect(db_path, timeout=READ_TIMEOUT_S)
        try:
            rows = conn.execute(
                "SELECT symbol, date, n_rows FROM dolt_fetched").fetchall()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        rows = []
        read_error = str(exc)

    if read_error is not None:
        detail = [{"stratum": stratum, "symbol": sym, "fetched_days": 0,
                   "nonempty_days": 0, "contracts": 0, "first": None,
                   "last": None, "state": "UNKNOWN"}
                  for stratum, syms in universe.items() for sym in sorted(syms)]
        summary = {stratum: {"total": len(syms), "usable": 0, "sparse": 0,
                             "absent": 0, "unknown": len(syms)}
                   for stratum, syms in universe.items()}
        return {"summary": summary, "detail": detail, "dead_dates": [],
                "thin_dates": [], "viable": False, "read_error": read_error}

    per_sym: Dict[str, List[tuple]] = defaultdict(list)
    date_total: Dict[str, int] = defaultdict(int)
    date_nonempty: Dict[str, int] = defaultdict(int)
    for sym, date, n in rows:
        if sym not in in_universe:
            continue
        per_sym[sym].append((date, n))
        date_total[date] += 1
        if n > 0:
            date_nonempty[date] += 1

    detail: List[Dict[str, Any]] = []
    summary: Dict[str, Dict[str, int]] = {}
    for stratum, syms in universe.items():
        usable = sparse = absent = 0
        for sym in sorted(syms):
            recs = per_sym.get(sym, [])
            nonempty = sum(1 for _, n in recs if n > 0)
            if nonempty == 0:
                state = "ABSENT"
                absent += 1
            elif nonempty < min_usable_days:
                state = "SPARSE"
                sparse += 1
            else:
                state = "ok"
                usable += 1
            dates = [d for d, n in recs if n > 0]
            detail.append({
                "stratum": stratum,
                "symbol": sym,
                "fetched_days": len(recs),
                "nonempty_days": nonempty,
                "contracts": sum(n for _, n in recs),
                "first": min(dates) if dates else None,
                "last": max(dates) if dates else None,
                "state": state,
            })
        summary[stratum] = {"total": len(syms), "usable": usable,
                            "sparse": sparse, "absent": absent}

    dead = sorted(d for d, t in date_total.items()
                  if t > 0 and date_nonempty[d] == 0)
    thin = sorted(d for d, t in date_total.items()
                  if t > 0 and 0 < date_nonempty[d] < 0.25 * t)
    viable = bool(summary) and all(v["usable"] >= 1 for v in summary.values())

    return {"summary": summary, "detail": detail, "dead_dates": dead,
            "thin_dates": thin, "viable": viable, "read_error": None}


def usable_symbols(audit: Dict[str, Any]) -> List[str]:
    return [d["symbol"] for d in audit["detail"] if d["state"] == "ok"]


def usable_dates(audit: Dict[str, Any], candidates: List[str]) -> List[str]:
    """Candidate dates that actually carry data.

    The DoltHub dataset does not cover every trading day — 2022-01-04 is a real
    Tuesday on which upstream holds zero rows even for AAPL. A date with no data
    is MISSING, and must never be scored as "no opportunity that day", which
    would quietly bias any always-on benchmark upward.

    Of the 232 Fridays sampled for this universe, 225 are usable and 7 are not.
    """
    dead = set(audit.get("dead_dates", ()))
    return [d for d in candidates if d not in dead]


def terminal_dates(audit: Dict[str, Any]) -> Dict[str, str]:
    """symbol -> last date it has data.

    A position still open on that date cannot be marked afterwards, so the
    engine closes it there rather than dropping it. Dropping would delete
    exactly the acquisitions and delistings, which is how a backtest becomes
    survivorship-biased despite unbiased data.
    """
    return {d["symbol"]: d["last"] for d in audit["detail"] if d["last"]}
