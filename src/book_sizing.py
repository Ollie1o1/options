"""How many contracts the paper book buys, and when it refuses to buy any.

Why this exists: every one of the 1,011 rows in ``paper_trades.db`` carried
``quantity = 1.0`` — a schema-migration default that nothing ever wrote — so
position size was the option's *premium*, a function of share price and implied
volatility with no relation to the pick. Measured 2026-08-19 over 889 closed
trades the book made **+$8,198 at PF 1.061 as sized** and **-$2,266 at PF
0.997** equal-weighted (CI [0.844, 1.178]). The headline profit was an artifact
of which trades happened to be large. Until size is a decision, no P&L figure
this system reports means what its label says.

The rule is fractional risk, not Kelly: half-Kelly needs a win probability and
this system's ranker is disproven (OOS IC -0.12; rho -0.030, p 0.38 over 880
closed). A rule that needs no forecast is the honest choice while no forecast
is trustworthy.

Two properties this module is built around:

* **Risk comes from `capital_at_risk`, never from entry premium.** A Bull Put
  *receives* its credit and risks ``width - credit``; it never pays
  ``entry x 100``. ``src/execution/sizing.py`` looks like this module's job and
  is not — it is long-call mirror-mode code whose formulas
  (``(entry - stop) * 100``, ``entry * 100``) cannot price the one strategy the
  book actually auto-logs. It is deliberately untouched and unused here.
* **`size` is pure.** No I/O, no config file, no clock. This project has
  already shipped one defect because a decision hid behind a default argument
  where no test could see it; the arithmetic worth testing exhaustively is
  therefore separated from the two queries that feed it.

Design record: docs/BOOK_SIZING_SPEC.md.
"""
from __future__ import annotations

import logging
import math
import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from .capital_risk import capital_at_risk_for_row

logger = logging.getLogger(__name__)

# Sizing off means one contract — the behaviour every historical row carries.
# A config that never opted in has not chosen to size.
_DEFAULTS: Dict[str, Any] = {
    "enabled": False,
    "opening_balance": 50_000.0,
    "equity_basis_date": "2026-08-05",
    "sizing_start_date": None,
    "max_risk_pct": 0.02,
    "max_open_risk_pct": 0.10,
}

#: Every value ``SizingDecision.reason`` can take. Refusals are diagnostic:
#: "the cap on simultaneous exposure bound" and "this one position is too big
#: for the account" are different facts and a quiet book must be able to say
#: which one it hit.
REASONS = ("risk_capped", "concurrent_capped", "below_one_contract",
           "unbounded_risk", "disabled", "no_equity")


@dataclass(frozen=True)
class SizingDecision:
    """The whole outcome of sizing one position.

    ``contracts == 0`` means REFUSE. A position too big to size at the account's
    risk fraction is one the account cannot afford, and rounding it up to a
    single contract would place a bet larger than the rule permits — sizing is a
    gate as well as a scale.
    """
    contracts: int
    reason: str
    risk_per_contract: Optional[float]
    equity: float


def _num(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def load_sizing_config(config: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Normalise ``config["position_sizing"]``, falling back per key.

    A malformed value falls back to its default rather than raising: this is
    read on the path that logs trades, and a typo in one field must not be able
    to take the ledger down. It cannot silently *loosen* anything either — the
    defaults are the conservative end of every knob.
    """
    block: Mapping[str, Any] = {}
    if isinstance(config, Mapping):
        raw = config.get("position_sizing")
        if isinstance(raw, Mapping):
            block = raw

    out = dict(_DEFAULTS)
    out["enabled"] = bool(block.get("enabled", _DEFAULTS["enabled"]))
    for key in ("opening_balance", "max_risk_pct", "max_open_risk_pct"):
        value = _num(block.get(key))
        if value is not None and value > 0:
            out[key] = value
    for key in ("equity_basis_date", "sizing_start_date"):
        value = block.get(key, _DEFAULTS[key])
        out[key] = str(value)[:10] if value else None
    return out


def size(risk_per_contract: Optional[float], equity: float, open_risk: float,
         cfg: Mapping[str, Any]) -> SizingDecision:
    """Contracts to trade, or 0 to refuse. Pure — no database, no clock.

    ``risk_per_contract`` must come from ``capital_at_risk`` at quantity 1.
    ``None`` there means the loss cannot be bounded (a naked call), which is
    unsizable rather than free.
    """
    equity_f = _num(equity) or 0.0
    risk = _num(risk_per_contract)

    if not cfg.get("enabled"):
        return SizingDecision(1, "disabled", risk, equity_f)
    if risk is None or risk <= 0:
        return SizingDecision(0, "unbounded_risk", risk, equity_f)
    if equity_f <= 0:
        return SizingDecision(0, "no_equity", risk, equity_f)

    max_risk_pct = _num(cfg.get("max_risk_pct")) or _DEFAULTS["max_risk_pct"]
    max_open_pct = (_num(cfg.get("max_open_risk_pct"))
                    or _DEFAULTS["max_open_risk_pct"])

    by_trade = int(math.floor(equity_f * max_risk_pct / risk))
    headroom = equity_f * max_open_pct - (_num(open_risk) or 0.0)
    by_book = int(math.floor(headroom / risk)) if headroom > 0 else 0

    contracts = min(by_trade, by_book)
    # Which cap bound is the diagnostic half of this answer: a book that has
    # gone quiet needs to distinguish "already fully deployed" from "these
    # trades are too big for the account".
    capped_by_book = by_book < by_trade
    if contracts < 1:
        reason = "concurrent_capped" if capped_by_book else "below_one_contract"
        return SizingDecision(0, reason, risk, equity_f)
    return SizingDecision(contracts,
                          "concurrent_capped" if capped_by_book else "risk_capped",
                          risk, equity_f)


def book_equity(conn: sqlite3.Connection, cfg: Mapping[str, Any]) -> float:
    """Opening balance plus realised P&L of trades ENTERED on or after the basis date.

    Entry-dated, not exit-dated, for two reasons: it is the split the 2026-08-05
    book restart uses everywhere else in this system, and it is the population
    the $50,000 opening balance was calibrated against (-$9,890 realised over 25
    closed trades -> $40,110 equity -> an $802 per-trade budget). A trade entered
    under the old book is not this book's money, whenever it happened to close.

    Only CLOSED trades count. An open position's mark is not realised, and
    sizing off a number that moves every minute would make the contract count
    depend on when the scan ran.
    """
    basis = cfg.get("equity_basis_date")
    opening = _num(cfg.get("opening_balance")) or _DEFAULTS["opening_balance"]
    sql = ("SELECT COALESCE(SUM(pnl_usd), 0.0) FROM trades "
           "WHERE status = 'CLOSED' AND pnl_usd IS NOT NULL")
    params: tuple = ()
    if basis:
        sql += " AND date >= ?"
        params = (basis,)
    row = conn.execute(sql, params).fetchone()
    return float(opening) + float((row[0] if row and row[0] is not None else 0.0))


def open_risk(conn: sqlite3.Connection, cfg: Mapping[str, Any]) -> float:
    """Dollars at risk across positions opened in the SIZED era.

    The 122 positions open when sizing shipped are grandfathered out: they were
    opened unsized, they carry $176,323 of risk against a $4,011 ceiling — 117x
    over — and holding an unsized book against a sized-era cap would refuse
    every new trade for months. ``sizing_start_date`` is the boundary; ``None``
    means the sized era has not started, so nothing counts yet.

    A row whose stored ``capital_at_risk`` is NULL is RECOMPUTED, never summed
    as zero: in this schema NULL means "not recorded", and treating it as no
    exposure would let the cap admit trades it should refuse. A row that still
    cannot be bounded is logged and skipped — one unbounded legacy position
    must not deadlock the whole book.
    """
    start = cfg.get("sizing_start_date")
    if not start:
        return 0.0
    cols = ("entry_id, strategy_name, entry_price, strike, max_loss_usd, "
            "spread_width, net_credit, quantity, ticker, capital_at_risk")
    total = 0.0
    for row in conn.execute(
            f"SELECT {cols} FROM trades WHERE status = 'OPEN' AND date >= ?",
            (start,)):
        record = dict(zip([c.strip() for c in cols.split(",")], row))
        stored = _num(record.get("capital_at_risk"))
        if stored is None:
            stored = _num(capital_at_risk_for_row(record))
        if stored is None:
            logger.warning(
                "open_risk: entry_id %s (%s on %s) has unbounded risk and is "
                "excluded from the concurrent cap — the cap understates by "
                "however much this position can lose",
                record.get("entry_id"), record.get("strategy_name"),
                record.get("ticker"))
            continue
        total += stored
    return total
