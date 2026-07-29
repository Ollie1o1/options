"""What a round trip actually costs, measured rather than assumed.

The ledger charged a flat $0.05 per share per leg to every structure. Joined
against the archived quotes of the exact contracts it logged, on the dates it
logged them (n=172 contract-days), that constant is wrong in both directions
and by a lot:

    Bull Put     $0.162 real vs $0.050 charged   3.2x undercharged
    Short Put    $0.100                          2.0x
    Long Call    $0.073                          1.5x
    Iron Condor  $0.055                          ~right
    Bear Call    $0.025                          2x OVERcharged

That asymmetry lands exactly where the backlog's execution conclusions live:
prefer-bull-put-at-small-size compares Bull Put's toll against Bear Call's, and
the flat model flatters the first while penalising the second.

The table is empirical — a join, not a model — so it improves on its own as
data/chain_archive.db accumulates. Where a structure has too few observations
to support a constant, the caller's default is used rather than a number
invented from three quotes.
"""
from __future__ import annotations

import sqlite3
from statistics import median
from typing import Any, Dict, Optional

DEFAULT_ARCHIVE = "data/chain_archive.db"
DEFAULT_LEDGER = "paper_trades.db"

# Below this many matched contract-days a bucket cannot set a cost constant.
MIN_OBSERVATIONS = 10


def _key(strategy: str) -> str:
    return (strategy or "").strip().lower()


def measure_half_spreads(archive_db: str = DEFAULT_ARCHIVE,
                         ledger_db: str = DEFAULT_LEDGER) -> Dict[str, Dict[str, Any]]:
    """Median quoted half-spread per structure, in dollars per share.

    Joins each logged contract to its archived quote on the trade's own entry
    date. Only two-sided quotes count: a zero bid or a crossed quote is missing
    data, and averaging it in would understate the real cost of crossing.
    """
    conn = sqlite3.connect(archive_db)
    try:
        conn.execute("ATTACH DATABASE ? AS pt", (ledger_db,))
        rows = conn.execute(
            """
            SELECT tr.strategy_name, (cs.ask - cs.bid) / 2.0
            FROM pt.trades tr
            JOIN chain_snapshots cs
              ON cs.symbol = tr.ticker
             AND cs.strike = tr.strike
             AND cs.expiration = tr.expiration
             AND substr(cs.type, 1, 1) = substr(tr.type, 1, 1)
             AND cs.snap_date = substr(tr.date, 1, 10)
            WHERE cs.bid > 0 AND cs.ask > cs.bid
            """
        ).fetchall()
    finally:
        conn.close()

    buckets: Dict[str, list] = {}
    for strategy, half in rows:
        if strategy is None or half is None:
            continue
        buckets.setdefault(strategy, []).append(float(half))

    return {
        strategy: {"n": len(vals), "median_half_spread": round(median(vals), 4)}
        for strategy, vals in buckets.items()
    }


def half_spread_for(strategy: str, table: Dict[str, Dict[str, Any]],
                    default: float) -> float:
    """Measured half-spread for a structure, or the caller's default."""
    wanted = _key(strategy)
    for name, cell in table.items():
        if _key(name) != wanted:
            continue
        if int(cell.get("n", 0)) < MIN_OBSERVATIONS:
            return default
        value = cell.get("median_half_spread")
        return float(value) if value is not None else default
    return default


def round_trip_friction(n_legs: int, half_spread: float,
                        commission_per_contract: float,
                        round_trip: bool = True) -> float:
    """Total friction per share for a position, in dollars.

    Every leg crosses the spread once on the way in and once on the way out,
    and pays a commission each time. ``round_trip=False`` prices the
    hold-to-expiry case: an option that expires worthless is never sold, so it
    pays the opening side only.
    """
    sides = 2 if round_trip else 1
    spread_cost = sides * float(half_spread) * int(n_legs)
    commissions = sides * float(commission_per_contract) * int(n_legs) / 100.0
    return spread_cost + commissions


class CostModel:
    """Friction for a structure, as dollars per share and as a fraction of credit.

    Fraction-of-credit is the unit the ledger's exit path works in, so this is
    directly comparable to the flat model it replaces.
    """

    def __init__(self, table: Optional[Dict[str, Dict[str, Any]]] = None,
                 default_half_spread: float = 0.05,
                 commission_per_contract: float = 0.65):
        self.table = table or {}
        self.default_half_spread = float(default_half_spread)
        self.commission_per_contract = float(commission_per_contract)

    def half_spread(self, strategy: str) -> float:
        return half_spread_for(strategy, self.table, self.default_half_spread)

    def friction(self, strategy: str, n_legs: int, round_trip: bool = True) -> float:
        return round_trip_friction(
            n_legs=n_legs,
            half_spread=self.half_spread(strategy),
            commission_per_contract=self.commission_per_contract,
            round_trip=round_trip,
        )

    def friction_fraction(self, strategy: str, entry_credit: float, n_legs: int,
                          round_trip: bool = True) -> float:
        """Friction as a fraction of the credit taken. 0.0 when there is no credit
        to measure against — a missing credit is not a free trade, it is a row
        the caller should skip."""
        credit = float(entry_credit or 0)
        if credit <= 0:
            return 0.0
        return self.friction(strategy, n_legs, round_trip) / credit


def reprice_pnl_pct(pnl_pct: float, strategy: str, entry_credit: float, n_legs: int,
                    old_model: CostModel, new_model: CostModel,
                    round_trip: bool = True) -> float:
    """A closed trade's return under a different cost model.

    Stored pnl_pct is already net of whatever friction was charged at close, so
    the old friction is added back before the new one is taken off. Everything
    else about the trade — entry, exit, timing — is untouched: this isolates the
    cost assumption and nothing else.
    """
    old = old_model.friction_fraction(strategy, entry_credit, n_legs, round_trip)
    new = new_model.friction_fraction(strategy, entry_credit, n_legs, round_trip)
    return float(pnl_pct) + old - new


def load_measured_model(archive_db: str = DEFAULT_ARCHIVE,
                        ledger_db: str = DEFAULT_LEDGER,
                        default_half_spread: float = 0.05,
                        commission_per_contract: float = 0.65) -> CostModel:
    """CostModel backed by the archive, falling back to the flat constant if the
    archive is absent or unreadable."""
    try:
        table = measure_half_spreads(archive_db, ledger_db)
    except sqlite3.Error:
        table = {}
    return CostModel(table, default_half_spread, commission_per_contract)
