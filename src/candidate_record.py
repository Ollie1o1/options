"""Every candidate a scan considered, not just the ones it took.

The sort key that selects real entries is `ev_per_contract`
(`candidate_verdict.py:230`), persisted as `trades.entry_ev_net`. That column
is populated on 26 rows, of which two are closed. The ledger cannot settle
whether the key ranks, because the ledger holds only what the ranker chose,
does not record rank position, and every row in it is a top-5 pick.

So this module records the population the ledger is missing: every pre-gate
row, its refusal reason, its rank position, and whether it was taken. It
observes only — no gate, ranker, or entry decision reads anything written here.

See docs/CANDIDATE_RECORD_SPEC.md.
"""
from __future__ import annotations

import contextvars
import functools
import json
import logging
import sqlite3
import traceback
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = "data/candidates.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidates (
  scan_id TEXT NOT NULL, ts TEXT NOT NULL, board TEXT NOT NULL,
  contract_key TEXT NOT NULL,
  symbol TEXT, strategy_name TEXT, expiration TEXT, strike REAL, opt_type TEXT,
  bid REAL, ask REAL, premium REAL, theta REAL, delta REAL,
  ev_net REAL, ev_gross REAL, ev_cost REAL, ev_noise REAL,
  quality_score REAL, round_trip_pct REAL,
  rank_pos INTEGER, refused_by TEXT, gate_passed INTEGER,
  gating_failed INTEGER NOT NULL DEFAULT 0,
  auto_logged INTEGER NOT NULL DEFAULT 0,
  entry_id INTEGER, features_json TEXT,
  PRIMARY KEY (scan_id, board, contract_key)
);
CREATE INDEX IF NOT EXISTS idx_cand_ts ON candidates(ts);
CREATE INDEX IF NOT EXISTS idx_cand_strategy ON candidates(strategy_name, ts);
CREATE TABLE IF NOT EXISTS recorder_errors (
  ts TEXT NOT NULL, scan_id TEXT, board TEXT, where_ TEXT, traceback TEXT
);
"""

# Option types a scan row may carry. `type` is overloaded — candidate_verdict
# reads `strategy_name or type`, so it sometimes holds a STRATEGY name. Only
# values in this map are accepted as an option type.
_OPT_TYPES = {"call": "call", "c": "call", "put": "put", "p": "put"}

# Leg strikes per structure, in a fixed order so the key is stable.
_LEG_STRIKES = {
    "Iron Condor": ("short_put_strike", "long_put_strike",
                    "short_call_strike", "long_call_strike"),
    "Bull Put": ("short_strike", "long_strike"),
    "Bear Call": ("short_strike", "long_strike"),
}


def connect(db_path: str = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open the candidate database, creating the schema when absent."""
    import os
    parent = os.path.dirname(db_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def _num(value: Any) -> Optional[float]:
    """A finite float, or None. Never raises. NaN is not a number either."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _strategy_of(row: Dict[str, Any]) -> str:
    name = row.get("strategy_name")
    if isinstance(name, str) and name.strip():
        return name.strip()
    raw = row.get("type")
    if isinstance(raw, str) and raw.strip().lower() not in _OPT_TYPES:
        return raw.strip()
    return ""


def _opt_type_of(row: Dict[str, Any]) -> str:
    for key in ("opt_type", "option_type", "type"):
        raw = row.get(key)
        if isinstance(raw, str) and raw.strip().lower() in _OPT_TYPES:
            return _OPT_TYPES[raw.strip().lower()]
    return ""


def _symbol_of(row: Dict[str, Any]) -> str:
    for key in ("symbol", "ticker", "Symbol", "Ticker"):
        val = row.get(key)
        if val:
            return str(val).strip().upper()
    return ""


def contract_key(row: Dict[str, Any]) -> str:
    """Deterministic identity for one candidate.

    Single leg: ``SYMBOL|expiration|opt_type|strike``.
    Structure:  ``SYMBOL|expiration|Strategy|s1/s2[/s3/s4]``.

    Stable across scans, and distinguishes two structures differing in exactly
    one leg. Sub-project 2 uses this to find the contract in chain_archive.db,
    so its shape is load-bearing beyond this module.
    """
    sym = _symbol_of(row)
    exp = str(row.get("expiration") or "").strip()
    strategy = _strategy_of(row)

    legs = _LEG_STRIKES.get(strategy)
    if legs:
        strikes = [_num(row.get(name)) for name in legs]
        # Only take the structure branch when at least one leg is actually
        # quoted. A structure row carrying no leg strikes would otherwise key
        # as "SYM|exp|Strategy|/" for EVERY such row, and the primary key would
        # silently overwrite one candidate with the next.
        if any(s is not None for s in strikes):
            return f"{sym}|{exp}|{strategy}|" + "/".join(
                "" if s is None else format(s, "g") for s in strikes)

    # Field 3 is the discriminator: the option type when the row names one,
    # otherwise the strategy. A strategy name reaching this slot is fine here
    # and is NOT the same thing as it reaching the `opt_type` column, which
    # `_opt_type_of` refuses.
    strike = _num(row.get("strike"))
    return (f"{sym}|{exp}|{_opt_type_of(row) or strategy}|"
            f"{'' if strike is None else format(strike, 'g')}")


# ── Observability ────────────────────────────────────────────────────────────
# A recorder that must never raise is one keystroke away from a recorder that
# never writes. `update_shadow_marks` returned cleanly under a bare
# `except: pass` and produced no data for four months before anyone noticed.
# So every failure lands in a counter, a WARNING, and a row.
STATS: Dict[str, int] = {"recorded": 0, "errors": 0, "autolog_only": 0}

_SCAN_ID: contextvars.ContextVar = contextvars.ContextVar(
    "candidate_scan_id", default=None)


def reset_stats() -> None:
    """Zero the counters. For tests and for a fresh scheduler run."""
    for key in STATS:
        STATS[key] = 0


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


_ERROR_SCHEMA = """
CREATE TABLE IF NOT EXISTS recorder_errors (
  ts TEXT NOT NULL, scan_id TEXT, board TEXT, where_ TEXT, traceback TEXT
);
"""


def _record_error(where: str, tb: str, db_path: str) -> None:
    """Persist a recorder failure. Best effort — it must not raise either.

    Deliberately does NOT go through `connect`. That runs the full schema
    script, including indexes over `candidates`, so a damaged `candidates`
    table takes the error path down with it — the recorder would then be
    unable to report the one failure most worth reporting. This touches only
    the table it writes to.

    When the database file itself is what broke, no write can succeed. The
    counter and the WARNING are the surviving signal in that case, which is
    why `health_lines` reads the counter and not only this table.
    """
    try:
        conn = sqlite3.connect(db_path)
        try:
            conn.executescript(_ERROR_SCHEMA)
            conn.execute(
                "INSERT INTO recorder_errors (ts, scan_id, board, where_, traceback)"
                " VALUES (?,?,?,?,?)",
                (_now(), _SCAN_ID.get(), None, where, tb))
            conn.commit()
        finally:
            conn.close()
    except Exception:
        log.warning("candidate recorder could not persist its own error",
                    exc_info=True)


def _safe(default):
    """Never raise into a scan; never fail silently.

    A broken recorder must not be able to stop a scan or change a pick. It
    must also never look like a recorder that had nothing to write.
    """
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, db_path: str = DEFAULT_DB_PATH, **kwargs):
            try:
                return fn(*args, db_path=db_path, **kwargs)
            except Exception:
                STATS["errors"] += 1
                log.warning("candidate recorder failed in %s",
                            fn.__name__, exc_info=True)
                _record_error(fn.__name__, traceback.format_exc(), db_path)
                return default
        return wrapper
    return deco


# Column order for every write. Kept in one place so the INSERT and the
# payload builder cannot drift apart.
_COLUMNS = ("scan_id", "ts", "board", "contract_key", "symbol", "strategy_name",
            "expiration", "strike", "opt_type", "bid", "ask", "premium",
            "theta", "delta", "ev_net", "ev_gross", "ev_cost", "ev_noise",
            "quality_score", "round_trip_pct", "rank_pos", "refused_by",
            "gate_passed", "gating_failed", "auto_logged", "entry_id",
            "features_json")


@_safe(default=0)
def record_board_rows(rows: List[Dict[str, Any]], *,
                      db_path: str = DEFAULT_DB_PATH) -> int:
    """Insert prepared candidate rows. Returns the number written."""
    if not rows:
        return 0
    sql = (f"INSERT OR REPLACE INTO candidates ({','.join(_COLUMNS)}) "
           f"VALUES ({','.join('?' * len(_COLUMNS))})")
    with connect(db_path) as conn:
        conn.executemany(sql, [tuple(r.get(c) for c in _COLUMNS) for r in rows])
        conn.commit()
    STATS["recorded"] += len(rows)
    return len(rows)


# ── Flattening a scan row into the schema ────────────────────────────────────
# Fixed columns are read off the row by these source keys, first match wins.
# Anything not named here goes to features_json, so a new scorer is never a
# migration.
_FIELD_SOURCES = {
    "symbol": ("symbol", "ticker"),
    "expiration": ("expiration",),
    "strike": ("strike",),
    "bid": ("bid",), "ask": ("ask",), "premium": ("premium",),
    "theta": ("theta",), "delta": ("delta",),
    "ev_net": ("ev_per_contract", "ev_net"),
    "ev_gross": ("ev_gross_per_contract", "ev_gross"),
    "ev_cost": ("ev_cost_per_contract", "ev_cost"),
    "ev_noise": ("ev_noise", "entry_ev_noise"),
    "quality_score": ("quality_score",),
    # `rank_by_verdict` writes Verdict.round_trip_pct into a column named
    # `friction_pct`. Read it, store it under the name that describes it.
    "round_trip_pct": ("round_trip_pct", "friction_pct"),
}

_TEXT_FIELDS = {"symbol", "expiration"}

# Consumed into fixed columns or handled separately; never duplicated into the
# features blob. `friction_pct` is here so the misleading name does not survive
# in the tail after being read into round_trip_pct.
_BLOB_EXCLUDE = (
    {src for sources in _FIELD_SOURCES.values() for src in sources}
    | {"refused_by", "verdict", "type", "ticker", "strategy_name",
       "opt_type", "option_type"}
)


def scan(mode: str):
    """Open one scan_id for the duration of a scan.

    One id spans every board that scan produces — it is opened around the
    scan, not around a board. That is what joins a gate record to the
    auto-log record for the same candidate.
    """
    import contextlib
    import uuid

    @contextlib.contextmanager
    def _ctx():
        scan_id = f"{_now()}|{mode}|{uuid.uuid4().hex[:8]}"
        token = _SCAN_ID.set(scan_id)
        try:
            yield scan_id
        finally:
            _SCAN_ID.reset(token)

    return _ctx()


def current_scan_id() -> str:
    """The active scan_id, or a standalone one when no scan is open.

    An unparented recording is still worth keeping — dropping rows because a
    caller forgot the context manager would be the silent-zero failure again.
    The `orphan` marker makes those rows findable rather than invisible.
    """
    import uuid
    return _SCAN_ID.get() or f"{_now()}|orphan|{uuid.uuid4().hex[:8]}"


def _is_jsonable(value: Any) -> bool:
    if value is None or isinstance(value, (bool, int, str)):
        return True
    if isinstance(value, float):
        return value == value and value not in (float("inf"), float("-inf"))
    return False


def row_payload(row: Dict[str, Any], *, board: str, scan_id: str,
                **over: Any) -> Dict[str, Any]:
    """One scan row flattened into the candidates schema."""
    out: Dict[str, Any] = {
        "scan_id": scan_id, "ts": _now(), "board": board,
        "contract_key": contract_key(row),
        "strategy_name": _strategy_of(row) or None,
        "opt_type": _opt_type_of(row) or None,
        "gating_failed": 0, "auto_logged": 0,
    }
    for field, sources in _FIELD_SOURCES.items():
        value = None
        for src in sources:
            if row.get(src) is not None:
                value = row.get(src)
                break
        if field in _TEXT_FIELDS:
            out[field] = str(value).strip() if value is not None else None
        else:
            out[field] = _num(value)

    tail = {k: v for k, v in row.items()
            if k not in _BLOB_EXCLUDE and _is_jsonable(v)}
    out["features_json"] = json.dumps(tail, sort_keys=True) if tail else None
    out.update(over)
    return out


def record_board(result: Any, *, board: str,
                 db_path: str = DEFAULT_DB_PATH) -> int:
    """Record a BoardResult: everything it kept and everything it refused.

    The refused rows are the point. A table of survivors only would be exactly
    as useless as the ledger this exists to supplement.
    """
    scan_id = current_scan_id()
    gating_failed = 1 if getattr(result, "gating_failed", False) else 0
    payloads: List[Dict[str, Any]] = []

    for frame, passed in ((getattr(result, "kept", None), 1),
                          (getattr(result, "refused", None), 0)):
        if frame is None or len(frame) == 0:
            continue
        for row in frame.to_dict("records"):
            payloads.append(row_payload(
                row, board=board, scan_id=scan_id,
                gate_passed=passed,
                refused_by=(None if passed else row.get("refused_by")),
                gating_failed=gating_failed))

    return record_board_rows(payloads, db_path=db_path)


# ── The entry path ───────────────────────────────────────────────────────────
# These upsert rather than insert. The auto-log frame is derived independently
# of the gated board (`options_screener.py` picks `_log_src` from picks /
# credit spreads / condors) and is deliberately NOT gated — G5 must not freeze
# its own training set. So a row arriving here may or may not have a gate
# record, and both cases have to work.

@_safe(default=0)
def mark_ranked(rows: List[Dict[str, Any]], *, board: str,
                db_path: str = DEFAULT_DB_PATH) -> int:
    """Write rank position across a ranked frame, 1-based, in frame order.

    Rows with no gate record are inserted AND counted. That count is the
    board/auto-log divergence — the same structural split that produced the
    "cleared the gates showed ungated rows" defect, measured rather than
    assumed absent.
    """
    if not rows:
        return 0
    scan_id = current_scan_id()
    keys = [contract_key(r) for r in rows]

    with connect(db_path) as conn:
        known = {k for (k,) in conn.execute(
            "SELECT contract_key FROM candidates WHERE scan_id=? AND board=?",
            (scan_id, board))}
        conn.executemany(
            "UPDATE candidates SET rank_pos=? "
            "WHERE scan_id=? AND board=? AND contract_key=?",
            [(i, scan_id, board, k) for i, k in enumerate(keys, start=1)
             if k in known])
        conn.commit()

    fresh = [(r, i) for i, (r, k) in enumerate(zip(rows, keys), start=1)
             if k not in known]
    if fresh:
        STATS["autolog_only"] += len(fresh)
        log.warning("%d auto-log rows on board %r had no gate record",
                    len(fresh), board)
        record_board_rows(
            [row_payload(r, board=board, scan_id=scan_id, rank_pos=i)
             for r, i in fresh], db_path=db_path)
    return len(rows)


@_safe(default=0)
def mark_refused(rows: List[Dict[str, Any]], reason: str, *, board: str,
                 db_path: str = DEFAULT_DB_PATH) -> int:
    """Record why a ranked candidate never reached the top-N cut.

    The auto-log allowlist and the per-scan budget cap both filter BEFORE the
    cut, so without this a candidate that was never eligible looks identical
    to one that competed and lost.
    """
    if not rows:
        return 0
    scan_id = current_scan_id()
    with connect(db_path) as conn:
        conn.executemany(
            "UPDATE candidates SET refused_by=?, gate_passed=0 "
            "WHERE scan_id=? AND board=? AND contract_key=?",
            [(reason, scan_id, board, contract_key(r)) for r in rows])
        conn.commit()
    return len(rows)


@_safe(default=None)
def mark_logged(row: Dict[str, Any], *, board: str, entry_id: Optional[int],
                db_path: str = DEFAULT_DB_PATH) -> None:
    """Flag one candidate as actually entered, with its ledger entry_id."""
    scan_id = current_scan_id()
    key = contract_key(row)
    with connect(db_path) as conn:
        cur = conn.execute(
            "UPDATE candidates SET auto_logged=1, entry_id=? "
            "WHERE scan_id=? AND board=? AND contract_key=?",
            (entry_id, scan_id, board, key))
        matched = cur.rowcount
        conn.commit()

    if matched == 0:
        # A trade was entered from a candidate nothing recorded. Keep it —
        # losing the taken row would be the worst possible gap in this table.
        record_board_rows([row_payload(row, board=board, scan_id=scan_id,
                                       auto_logged=1, entry_id=entry_id)],
                          db_path=db_path)
