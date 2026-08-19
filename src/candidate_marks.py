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


# ── Pricing a would-be entry ─────────────────────────────────────────────────
# Leg quote columns per structure, in the same fixed order as the strikes in
# `candidate_record._LEG_STRIKES`, with the side each leg is traded on.
_LEG_QUOTES = {
    "Iron Condor": (("short_put", "sell"), ("long_put", "buy"),
                    ("short_call", "sell"), ("long_call", "buy")),
    "Bull Put": (("short", "sell"), ("long", "buy")),
    "Bear Call": (("short", "sell"), ("long", "buy")),
}


def _blob(row: Dict[str, Any]) -> Dict[str, Any]:
    """The recorded feature tail, or an empty dict."""
    import json
    raw = row.get("features_json")
    if not raw:
        return {}
    try:
        out = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return out if isinstance(out, dict) else {}


def _quote(bid: Any, ask: Any) -> Optional[Tuple[float, float]]:
    b, a = cr._num(bid), cr._num(ask)
    if b is None or a is None or a <= 0 or b < 0 or a < b:
        return None
    return b, a


def legs_for(row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """Quoted legs for one recorded candidate, or None if it cannot be priced.

    Structure legs live in `features_json`. The recorder promotes only a single
    `bid`/`ask` pair to fixed columns, and `_BLOB_EXCLUDE` never excluded the
    per-leg names, so `short_bid` and friends survive in the blob verbatim.

    Refuses the whole structure when any leg is unquoted, matching
    `candidate_verdict._legs_of`: a spread priced from one real quote and one
    guess is not a price.
    """
    strategy = (row.get("strategy_name") or "").strip()
    spec = _LEG_QUOTES.get(strategy)
    if spec:
        blob = _blob(row)
        legs = []
        for prefix, side in spec:
            q = _quote(blob.get(f"{prefix}_bid"), blob.get(f"{prefix}_ask"))
            if q is None:
                return None
            legs.append({"bid": q[0], "ask": q[1], "side": side})
        return legs

    q = _quote(row.get("bid"), row.get("ask"))
    if q is None:
        return None
    side = "sell" if strategy.startswith("Short") else "buy"
    return [{"bid": q[0], "ask": q[1], "side": side}]


def entry_price_for(row: Dict[str, Any]) -> Optional[float]:
    """What this candidate would have been entered at, signed.

    Positive is a net credit received, negative a net debit paid — the sign
    convention `execution_truth.FillResult.price` already uses.

    Priced at the `limit` policy because that is what real entries record
    (`trades.fill_policy = 'limit'`). A candidate priced on any other policy is
    not comparable to the book, and comparability is the only reason to mark it
    forward at all.
    """
    from . import execution_truth as et

    legs = legs_for(row)
    if legs is None:
        return None
    fill = et.structure_fill(legs, "limit")
    if fill is None:
        return None
    price = cr._num(fill.price)
    return price if price else None


def pnl_pct(entry_signed: Optional[float],
            mark_abs: Optional[float]) -> Optional[float]:
    """Return on the premium, from a signed entry and a positive mark.

    Derived from the SIGN rather than from the family, so a debit spread is
    handled correctly without anyone remembering to add it to a table:

        debit  (entry < 0): paid |e|, now worth m      -> (m - |e|) / |e|
        credit (entry > 0): kept |e|, costs m to close -> (|e| - m) / |e|
    """
    e, m = cr._num(entry_signed), cr._num(mark_abs)
    if e is None or m is None or e == 0:
        return None
    base = abs(e)
    raw = (m - base) / base
    return raw if e < 0 else -raw
