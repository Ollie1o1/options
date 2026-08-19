"""Outcomes for candidates that were never traded.

`data/candidates.db` records every pre-gate candidate and why it was refused,
but a refusal with no recorded consequence is the same hole the ledger has —
the ledger just hides it better, by containing only what was taken. This module
opens a simulated position per recorded candidate, marks it daily from the same
quote call that marks real shadowed trades, and resolves it against the exit
rules read from config.

It observes. Nothing in the scan path reads these tables, and `paper_trades.db`
is never opened for writing.

See docs/CANDIDATE_FORWARD_MARKS_SPEC.md.
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Any, Dict, List, Optional, Tuple

from . import candidate_record as cr

log = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS candidate_marks (
  contract_key TEXT NOT NULL,
  mark_date    TEXT NOT NULL,
  bid REAL, ask REAL, mid REAL,
  source       TEXT NOT NULL,
  PRIMARY KEY (contract_key, mark_date)
);
CREATE TABLE IF NOT EXISTS candidate_positions (
  scan_id      TEXT NOT NULL,
  board        TEXT NOT NULL,
  contract_key TEXT NOT NULL,
  family       TEXT,
  entry_date   TEXT NOT NULL,
  entry_price  REAL,
  status       TEXT NOT NULL,
  exit_date    TEXT,
  exit_price   REAL,
  exit_reason  TEXT,
  pnl_pct      REAL,
  PRIMARY KEY (scan_id, board, contract_key)
);
CREATE INDEX IF NOT EXISTS idx_pos_status ON candidate_positions(status);
CREATE INDEX IF NOT EXISTS idx_pos_contract ON candidate_positions(contract_key);
"""

# Exit-rule family per strategy — the keys of config.json -> exit_rules.
_FAMILY_BY_STRATEGY = {
    "Long Call": "long_option", "Long Put": "long_option",
    "Bull Put": "spread", "Bear Call": "spread", "Iron Condor": "spread",
    "Short Put": "short_premium", "Short Call": "short_premium",
}

OPEN = "OPEN"
CLOSED = "CLOSED"
UNMARKABLE = "UNMARKABLE"      # nothing to price it from; no entry price
UNSUPPORTED = "UNSUPPORTED"    # a family whose exits this version cannot decide


def connect(db_path: Optional[str] = None) -> sqlite3.Connection:
    """Open the candidate database with this module's tables present.

    Goes through `candidate_record.connect` so the recorder's tables, its
    column migrations and the path resolution stay in one place; each module
    owns and applies its own schema on top of that.
    """
    conn = cr.connect(db_path)
    conn.executescript(_SCHEMA)
    conn.commit()
    return conn


def family_for(mode: Optional[str], opt_type: Optional[str],
               strategy_name: Optional[str] = None) -> Optional[str]:
    """The `config.exit_rules` family for one candidate, or None.

    A recorded `strategy_name` wins: structure boards label their rows, and a
    Bull Put is a Bull Put regardless of which mode produced it. Single legs
    carry no strategy — only an option type — so theirs is derived from the
    mode by the same `strategy_label_for_mode` the auto-log path uses, rather
    than a second copy of that mapping.

    Returns None when there is nothing to derive from. **A None family is not a
    default**; it means this candidate cannot be simulated, and the caller must
    not invent one for it.
    """
    if strategy_name:
        return _FAMILY_BY_STRATEGY.get(str(strategy_name).strip())
    if not mode or not opt_type:
        return None
    from .trade_analysis import strategy_label_for_mode
    try:
        label = strategy_label_for_mode(str(mode), opt_type)
    except Exception:
        log.debug("strategy labelling failed", exc_info=True)
        return None
    return _FAMILY_BY_STRATEGY.get(label)
