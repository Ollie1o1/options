"""Capture one archived session for the study and build both arms on identical quotes.

Source: `data/chain_archive.db`, written by the scheduled archive job (weekday
afternoons). Each (session, symbol) gets exactly one observation row, including the
empty ones, with a status that separates:

  captured          quotes usable; the selector ran (it may still have chosen nothing)
  missing_chain     no archived rows, and the archive did not record a provider error
  provider_failure  no archived rows, and the archive recorded an error or a dead feed
  unusable_quotes   rows exist but no expiration in the window had usable two-sided puts
  missing_spot      rows exist but carry no underlying price
  stale             the rows were observed on a different day than the session
  off_window        observed outside the agreed time window, or with no observation time

Only `missing_chain` and `provider_failure` are retried: the archive can still fill them.
Every other status, and every entry built from a captured observation, is frozen.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import sqlite3
import statistics
import uuid
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Sequence

from src.paths import repo_path
from src.selector_study import baseline as bl
from src.selector_study.db import load_settings, settings_hash
from src.structure.chain_data import ChainContract, ChainSlice, load_archived_chain_slice

STATUSES = ("captured", "missing_chain", "provider_failure", "unusable_quotes",
            "missing_spot", "stale", "off_window")
RETRYABLE = frozenset({"missing_chain", "provider_failure"})
_ET = "America/New_York"


def _utcnow() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_ts(raw: Any) -> Optional[dt.datetime]:
    if not raw:
        return None
    try:
        t = dt.datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)


def _to_et(t: dt.datetime) -> dt.datetime:
    from zoneinfo import ZoneInfo
    return t.astimezone(ZoneInfo(_ET))


def archive_sessions(chain_db: str) -> List[str]:
    full = repo_path(chain_db)
    if not os.path.exists(full):
        return []
    with sqlite3.connect(f"file:{full}?mode=ro", uri=True) as con:
        return [str(r[0]) for r in con.execute(
            "SELECT DISTINCT snap_date FROM chain_snapshots ORDER BY snap_date")]


def _archive_facts(chain_db: str, session: str, symbol: str) -> Dict[str, Any]:
    full = repo_path(chain_db)
    facts: Dict[str, Any] = {"rows": 0, "spot": None, "first": None, "last": None,
                             "source": None, "attempt": None}
    if not os.path.exists(full):
        return facts
    with sqlite3.connect(f"file:{full}?mode=ro", uri=True) as con:
        rows = con.execute(
            "SELECT spot, snapshot_ts, source FROM chain_snapshots WHERE symbol=? AND snap_date=?",
            (symbol, session)).fetchall()
        tables = {r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "archive_attempts" in tables:
            att = con.execute("SELECT status, detail FROM archive_attempts WHERE snap_date=? AND symbol=?",
                              (session, symbol)).fetchone()
            facts["attempt"] = (str(att[0]), att[1]) if att else None
    facts["rows"] = len(rows)
    spots = [float(r[0]) for r in rows if r[0] is not None and float(r[0]) > 0]
    facts["spot"] = statistics.median(spots) if spots else None
    stamps = sorted(str(r[1]) for r in rows if r[1])
    facts["first"], facts["last"] = (stamps[0], stamps[-1]) if stamps else (None, None)
    facts["source"] = ",".join(sorted({str(r[2]) for r in rows if r[2]})) or None
    return facts


def _time_status(session: str, first: Optional[str], last: Optional[str],
                 window: Sequence[str]) -> Optional[tuple[str, str]]:
    """None when the observation time is on the session and inside the window."""
    t0, t1 = _parse_ts(first), _parse_ts(last)
    if t0 is None or t1 is None:
        return "off_window", "no observation time recorded"
    e0, e1 = _to_et(t0), _to_et(t1)
    if e0.date().isoformat() != session or e1.date().isoformat() != session:
        return "stale", f"observed {e0:%Y-%m-%d %H:%M}..{e1:%Y-%m-%d %H:%M} ET for session {session}"
    lo, hi = (dt.time.fromisoformat(x) for x in window)
    if e0.time() < lo or e1.time() > hi:
        return "off_window", f"observed {e0:%H:%M}..{e1:%H:%M} ET, window {window[0]}-{window[1]}"
    return None


def slice_from_json(raw: Dict[str, Any]) -> ChainSlice:
    return ChainSlice(**{
        **raw,
        "calls": [ChainContract(**q) for q in raw.get("calls", [])],
        "puts": [ChainContract(**q) for q in raw.get("puts", [])],
    })


def _slice_json(sl: ChainSlice) -> Dict[str, Any]:
    raw = asdict(sl)
    raw["rnd_density"] = None
    return raw


def _entry_record(*, pair_id: str, arm: str, session: str, symbol: str, sl: ChainSlice,
                  short: ChainContract, long: ChainContract, frozen: Dict[str, Any]) -> Dict[str, Any]:
    v = bl.Vertical(short, long)
    return {
        "entry_id": f"{pair_id}|{arm}|{short.strike:g}/{long.strike:g}",
        "pair_id": pair_id, "arm": arm, "session": session, "symbol": symbol,
        "structure": "bull_put", "expiration": sl.expiration, "dte": sl.dte,
        "short_strike": short.strike, "long_strike": long.strike, "width": v.width,
        "short_delta": abs(short.delta) if short.delta is not None else None,
        "credit_mid": v.credit_mid, "credit_cross": v.credit_cross, "rel_spread": v.rel_spread,
        "frozen_json": json.dumps({**frozen, "short": asdict(short), "long": asdict(long)},
                                  sort_keys=True, allow_nan=False),
        "created_at": _utcnow(),
    }


def _insert(conn: sqlite3.Connection, table: str, rec: Dict[str, Any]) -> bool:
    cols = list(rec)
    cur = conn.execute(f"INSERT OR IGNORE INTO {table} ({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                       [rec[c] for c in cols])
    return cur.rowcount > 0


def build_arms(conn: sqlite3.Connection, *, session: str, symbol: str, sl: ChainSlice, spot: float,
               earnings_days: Optional[int], observed_at: Optional[str], source: Optional[str],
               settings: Dict[str, Any]) -> Dict[str, int]:
    """Run the generic selector on the slice, then its matched baseline. Returns counts."""
    from src.tui.options_engine import _build_candidates_for_slice

    max_fric = float(settings["max_friction_pct"])
    rows = [r.to_dict() for r in _build_candidates_for_slice(
        sl, spot, rv20=None, earnings_days=earnings_days, strategy_filter="bull_put",
        max_friction_pct=max_fric) if r.strategy_key == "bull_put"]
    counts = {"selector": len(rows), "pairs_new": 0, "entries_new": 0}
    frozen_common = {"spot": spot, "observed_at": observed_at, "source": source,
                     "quote_time": None, "earnings_days": earnings_days,
                     "settings_hash": settings_hash(settings),
                     "selector": "src.tui.options_engine._build_candidates_for_slice"}
    for row in rows:
        pair_id = f"{session}|{symbol}|bull_put|{row['expiration']}"
        s_leg = next(lg for lg in row["legs"] if lg["side"] == "SELL")
        l_leg = next(lg for lg in row["legs"] if lg["side"] == "BUY")
        s_c, l_c = bl.find_leg(sl, float(s_leg["strike"])), bl.find_leg(sl, float(l_leg["strike"]))
        if s_c is None or l_c is None:
            continue
        sel = _entry_record(pair_id=pair_id, arm="selector", session=session, symbol=symbol, sl=sl,
                            short=s_c, long=l_c,
                            frozen={**frozen_common, "flags": list(row.get("risk_flags") or ())})
        best, status, n_eligible = bl.match(sl, row, rules=settings["baseline"], max_friction_pct=max_fric)
        base = None
        if best is not None:
            base = _entry_record(pair_id=pair_id, arm="baseline", session=session, symbol=symbol, sl=sl,
                                 short=best.short, long=best.long,
                                 frozen={**frozen_common, "eligible_verticals": n_eligible})
        counts["entries_new"] += int(_insert(conn, "entries", sel))
        if base is not None:
            counts["entries_new"] += int(_insert(conn, "entries", base))
        counts["pairs_new"] += int(_insert(conn, "pairs", {
            "pair_id": pair_id, "session": session, "symbol": symbol, "expiration": sl.expiration,
            "status": "no_match" if best is None else status,
            "reason": status.split(":", 1)[1] if status.startswith("no_match:") else None,
            "eligible_baseline": n_eligible, "selector_entry": sel["entry_id"],
            "baseline_entry": base["entry_id"] if base else None,
        }))
    return counts


def capture_session(conn: sqlite3.Connection, session: str, *, chain_db: str = "data/chain_archive.db",
                    earnings: Optional[Dict[str, int]] = None,
                    symbols: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Capture `session` for every agreed symbol (or the agreed subset in `symbols`).

    Idempotent; returns the run summary. A study whose `capture_window_et` is None reads an
    end-of-day source with no observation time, so the time check does not apply."""
    settings = load_settings(conn)
    window = settings["capture_window_et"]
    dte = settings["dte_window"]
    agreed = list(dict.fromkeys(s.strip().upper() for s in settings["symbols"] if s.strip()))
    if symbols is not None:
        wanted = {s.upper() for s in symbols}
        agreed = [s for s in agreed if s in wanted]
    frozen = {str(sym) for sym, st in conn.execute(
        "SELECT symbol, status FROM observations WHERE session=?", (session,)) if st not in RETRYABLE}
    if all(sym in frozen for sym in agreed):
        return {"session": session, "skipped": True, "already": len(agreed), "run_status": "complete"}
    run_id = f"{session}@{_utcnow()}#{uuid.uuid4().hex[:6]}"
    conn.execute("INSERT INTO capture_runs(run_id, session, started_at, source, status) VALUES (?,?,?,?,?)",
                 (run_id, session, _utcnow(), "chain_archive", "running"))
    conn.commit()
    counts: Dict[str, Any] = {"run_id": run_id, "session": session, "already": 0,
                              "status": {}, "selector": 0, "pairs_new": 0, "entries_new": 0}
    for symbol in agreed:
        prior = conn.execute("SELECT status FROM observations WHERE session=? AND symbol=?",
                             (session, symbol)).fetchone()
        if prior is not None and prior[0] not in RETRYABLE:
            counts["already"] += 1
            counts["status"][prior[0]] = counts["status"].get(prior[0], 0) + 1
            continue
        facts = _archive_facts(chain_db, session, symbol)
        obs: Dict[str, Any] = {"session": session, "symbol": symbol, "source": facts["source"],
                               "observed_first": facts["first"], "observed_at": facts["last"],
                               "quote_time": None, "spot": facts["spot"], "detail": None}
        sl: Optional[ChainSlice] = None
        if facts["rows"] == 0:
            att = facts["attempt"]
            if att and att[0] in ("error", "feed_down"):
                obs.update(status="provider_failure", detail=att[1] or att[0])
            else:
                obs.update(status="missing_chain",
                           detail=(att[1] if att else "no archive rows and no archive attempt recorded"))
        elif facts["spot"] is None:
            obs.update(status="missing_spot", detail=f"{facts['rows']} rows without an underlying price")
        else:
            timing = None if window is None else _time_status(session, facts["first"], facts["last"], window)
            if timing is not None:
                obs.update(status=timing[0], detail=timing[1])
            else:
                sl = load_archived_chain_slice(symbol, target_dte=int(dte["target"]), min_dte=int(dte["min"]),
                                               max_dte=int(dte["max"]), spot=facts["spot"], chain_db=chain_db,
                                               build_rnd=False, snap_date=session)
                two_sided = sum(c.bid > 0 and c.ask >= c.bid for c in sl.puts) if sl else 0
                if sl is None or two_sided == 0:
                    obs.update(status="unusable_quotes",
                               detail=f"no {dte['min']}-{dte['max']} DTE expiration with two-sided puts")
                    sl = None
                else:
                    sl.spot = facts["spot"]
                    obs.update(status="captured", expiration=sl.expiration, dte=sl.dte, two_sided=two_sided)
        earn = (earnings or {}).get(symbol)
        obs["inputs_json"] = json.dumps({"spot": facts["spot"], "earnings_days": earn,
                                         "settings_hash": settings_hash(settings),
                                         "rows": facts["rows"]}, sort_keys=True)
        if sl is not None:
            obs["slice_json"] = json.dumps(_slice_json(sl), sort_keys=True, allow_nan=False)
            built = build_arms(conn, session=session, symbol=symbol, sl=sl, spot=float(facts["spot"]),
                               earnings_days=earn, observed_at=facts["last"], source=facts["source"],
                               settings=settings)
            obs["selector_candidates"] = built["selector"]
            for k in ("selector", "pairs_new", "entries_new"):
                counts[k] += built[k]
        cols = ["status", "detail", "source", "observed_first", "observed_at", "quote_time", "spot",
                "expiration", "dte", "two_sided", "selector_candidates", "slice_json", "inputs_json"]
        vals = [obs.get(c) for c in cols]
        conn.execute(
            f"INSERT INTO observations (session, symbol, {','.join(cols)}, attempts, first_run, last_run) "
            f"VALUES (?, ?, {','.join('?' * len(cols))}, 1, ?, ?) "
            f"ON CONFLICT(session, symbol) DO UPDATE SET "
            + ", ".join(f"{c}=excluded.{c}" for c in cols)
            + ", attempts = observations.attempts + 1, last_run = excluded.last_run",
            [session, symbol, *vals, run_id, run_id])
        conn.commit()
        counts["status"][obs["status"]] = counts["status"].get(obs["status"], 0) + 1
    incomplete = sum(counts["status"].get(s, 0) for s in RETRYABLE)
    status = "partial" if incomplete else "complete"
    conn.execute("UPDATE capture_runs SET finished_at=?, status=?, counts_json=? WHERE run_id=?",
                 (_utcnow(), status, json.dumps(counts, sort_keys=True), run_id))
    conn.commit()
    counts["run_status"] = status
    return counts
