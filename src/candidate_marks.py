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


# ── The simulated book ───────────────────────────────────────────────────────

@cr._safe(default=0)
def open_positions(*, db_path: Optional[str] = None,
                   today: Optional[str] = None) -> int:
    """Give every recorded candidate without one a simulated position.

    One per (scan_id, board, contract_key) — one per decision instance,
    matching the primary key of `candidates` itself. The same contract recorded
    on three scans is three positions at three entry prices sharing one stream
    of marks. The correlation that creates is handled at analysis time by
    clustering on `contract_key`: a cluster-robust error can always be computed
    later, whereas observations discarded here cannot be recovered.
    """
    from datetime import datetime
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")

    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT c.* FROM candidates c "
            "LEFT JOIN candidate_positions p "
            "  ON p.scan_id = c.scan_id AND p.board = c.board "
            " AND p.contract_key = c.contract_key "
            "WHERE p.contract_key IS NULL").fetchall()

        made = 0
        for raw in rows:
            row = dict(raw)
            family = family_for(row.get("mode"), row.get("opt_type"),
                                row.get("strategy_name"))
            if family is None:
                # Nothing to derive exit rules from, so nothing to simulate.
                # Deliberately not an UNMARKABLE row: every pre-`mode` row in
                # the database would become one, burying the rows that mean
                # something under thousands of inert placeholders.
                continue

            if family == "short_premium":
                status, price, reason = UNSUPPORTED, None, "needs_spot_and_delta"
            else:
                price = entry_price_for(row)
                status = OPEN if price is not None else UNMARKABLE
                reason = None if price is not None else "no_two_sided_quote"

            conn.execute(
                "INSERT OR REPLACE INTO candidate_positions "
                "(scan_id, board, contract_key, family, entry_date, entry_price,"
                " status, exit_reason) VALUES (?,?,?,?,?,?,?,?)",
                (row["scan_id"], row["board"], row["contract_key"], family,
                 today, price, status, reason))
            made += 1
        conn.commit()
    return made


def _default_fetch(ticker: str, expiration: str):
    """Live bid/ask per (strike, type), from the call that marks the real book.

    Using `PaperManager._fetch_chain_quotes` rather than a second fetcher is
    the whole point: a candidate marked on a different basis than a real trade
    is not comparable to one, and comparability is the only reason to mark it.
    """
    from .paper_manager import PaperManager
    return PaperManager()._fetch_chain_quotes(ticker, expiration)


@cr._safe(default=0)
def mark_open(*, db_path: Optional[str] = None, today: Optional[str] = None,
              fetch=None) -> int:
    """Quote every open position and record the mark. Returns rows written.

    One chain request per (symbol, expiration), shared by every position on
    that pair — the same shape `PaperManager.update_shadow_marks` uses. A
    failure on one pair must not cost the others their marks: a missing day is
    missing forever.
    """
    from datetime import datetime
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")
    if fetch is None:
        fetch = _default_fetch

    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT DISTINCT c.symbol, c.expiration, c.strike, c.opt_type,"
            "       c.contract_key "
            "FROM candidate_positions p JOIN candidates c "
            "  ON c.scan_id = p.scan_id AND c.board = p.board "
            " AND c.contract_key = p.contract_key "
            "WHERE p.status = ?", (OPEN,))]

        chains: Dict[Tuple[str, str], dict] = {}
        written = 0
        for row in rows:
            symbol, expiration = row.get("symbol"), row.get("expiration")
            strike, opt_type = cr._num(row.get("strike")), row.get("opt_type")
            if not symbol or not expiration or strike is None or not opt_type:
                continue
            key = (symbol, expiration)
            if key not in chains:
                try:
                    chains[key] = fetch(symbol, expiration) or {}
                except Exception:
                    log.warning("candidate mark fetch failed for %s %s",
                                symbol, expiration, exc_info=True)
                    cr.STATS["errors"] += 1
                    chains[key] = {}
            q = _quote(*chains[key].get((round(strike, 4),
                                         str(opt_type).lower()), (None, None)))
            if q is None:
                continue
            conn.execute(
                "INSERT OR REPLACE INTO candidate_marks "
                "(contract_key, mark_date, bid, ask, mid, source) "
                "VALUES (?,?,?,?,?,?)",
                (row["contract_key"], today, q[0], q[1], (q[0] + q[1]) / 2.0,
                 "live_quote"))
            written += 1
        conn.commit()
    return written


# ── Exits ────────────────────────────────────────────────────────────────────

def exit_rules(cfg_path: str = "config.json") -> Dict[str, Any]:
    """The book's exit rules, READ rather than restated.

    `enforce_exits` reads the same block. Copying these numbers into this
    module would let them drift, and a counterfactual resolved under different
    rules than the book's is not a counterfactual — it is a different
    experiment wearing the same label.
    """
    import json
    try:
        with open(cfg_path) as fh:
            return (json.load(fh) or {}).get("exit_rules") or {}
    except Exception:
        log.warning("exit rules unreadable at %s", cfg_path, exc_info=True)
        return {}


def _days_between(a: str, b: str) -> Optional[int]:
    """Calendar days from a to b, or None if either date is unusable."""
    from datetime import date
    try:
        ya, ma, da = (int(x) for x in str(a).split("-"))
        yb, mb, db = (int(x) for x in str(b).split("-"))
        return (date(yb, mb, db) - date(ya, ma, da)).days
    except Exception:
        return None


@cr._safe(default=0)
def resolve(*, db_path: Optional[str] = None, today: Optional[str] = None,
            cfg_path: str = "config.json") -> int:
    """Close open positions whose latest mark triggers an exit.

    Precedence matches the book: expiry, then the time exit, then take-profit,
    then stop-loss. `min_days_held` suppresses everything except expiry, so a
    position cannot be opened and closed on the same mark.

    Only marks dated on or before `today` are considered — resolving day N
    against a mark from day N+1 would be lookahead, and lookahead in a
    counterfactual is how a study proves whatever it likes.
    """
    from datetime import datetime
    if today is None:
        today = datetime.now().strftime("%Y-%m-%d")

    rules = exit_rules(cfg_path)
    min_days = int(rules.get("min_days_held") or 0)
    time_exit_dte = int(rules.get("time_exit_dte") or 0)

    with connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        rows = [dict(r) for r in conn.execute(
            "SELECT p.rowid AS rid, p.family, p.entry_date, p.entry_price,"
            "       c.expiration,"
            "       (SELECT m.mid FROM candidate_marks m "
            "         WHERE m.contract_key = p.contract_key "
            "           AND m.mark_date <= ? "
            "         ORDER BY m.mark_date DESC LIMIT 1) AS mark,"
            "       (SELECT m.mark_date FROM candidate_marks m "
            "         WHERE m.contract_key = p.contract_key "
            "           AND m.mark_date <= ? "
            "         ORDER BY m.mark_date DESC LIMIT 1) AS mark_date "
            "FROM candidate_positions p JOIN candidates c "
            "  ON c.scan_id = p.scan_id AND c.board = p.board "
            " AND c.contract_key = p.contract_key "
            "WHERE p.status = ?", (today, today, OPEN))]

        closed = 0
        for row in rows:
            mark = cr._num(row.get("mark"))
            if mark is None:
                continue
            pnl = pnl_pct(row.get("entry_price"), mark)
            if pnl is None:
                continue

            held = _days_between(row.get("entry_date") or "", today)
            dte = _days_between(today, row.get("expiration") or "")
            fam = rules.get(row.get("family")) or {}
            reason = None

            if dte is not None and dte <= 0:
                reason = "expired"
            elif held is not None and held < min_days:
                reason = None
            elif dte is not None and time_exit_dte and dte <= time_exit_dte:
                reason = "time_exit"
            elif (fam.get("take_profit") is not None
                    and pnl >= float(fam["take_profit"])):
                reason = "take_profit"
            elif (fam.get("stop_loss") is not None
                    and pnl <= float(fam["stop_loss"])):
                reason = "stop_loss"

            if reason is None:
                continue
            conn.execute(
                "UPDATE candidate_positions SET status=?, exit_date=?, "
                "exit_price=?, exit_reason=?, pnl_pct=? WHERE rowid=?",
                (CLOSED, row.get("mark_date") or today, mark, reason, pnl,
                 row["rid"]))
            closed += 1
        conn.commit()
    return closed
