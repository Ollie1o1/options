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
import json
import logging
import sqlite3
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
