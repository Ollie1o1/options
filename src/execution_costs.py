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
from typing import Any, Dict, Optional, Tuple

from src.spread_surface import SpreadSurface

DEFAULT_ARCHIVE = "data/chain_archive.db"
DEFAULT_LEDGER = "paper_trades.db"

# Below this many matched contract-days a bucket cannot set a cost constant.
MIN_OBSERVATIONS = 10

# The commission used ONLY when no config is reachable. Every normal path reads
# `config.paper_trading.commission_per_contract`, which is 0.0 — this operator's
# broker (Wealthsimple) charges nothing per equity or ETF option contract.
#
# This fallback is deliberately NOT 0.0. It fires when a helper is constructed
# directly without a commission threaded through — typically a standalone
# backtest — and in that situation a US-retail rate overstates cost, while 0.0
# would understate it. Overstating is the safe direction: it can make a
# strategy look worse than it is, never better.
#
# One number, one place. When the broker changes, change it here. Note that a
# single scalar cannot express index options, which DO carry a per-contract fee
# (SPX/SPXW $1.00, VIX $1.00, RUT $0.50, XSP $0.25) — see the
# `index-options-are-not-free` entry before any SPX work.
FALLBACK_COMMISSION_PER_CONTRACT = 0.65


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


def is_measured(strategy: str, table: Dict[str, Dict[str, Any]]) -> bool:
    """Whether `half_spread_for` would return this table's own measured
    value for `strategy` rather than falling back to the caller's default.

    Same matching and MIN_OBSERVATIONS floor as `half_spread_for`, kept in
    lockstep with it here rather than duplicated at each call site — a
    caller that wants to label a number as "measured" must ask this, not
    assume a table entry means a measurement: a strategy with fewer than
    MIN_OBSERVATIONS matched quotes has an entry in the table but its
    value is never used.
    """
    wanted = _key(strategy)
    for name, cell in table.items():
        if _key(name) != wanted:
            continue
        return int(cell.get("n", 0)) >= MIN_OBSERVATIONS
    return False


def fx_cost(cash_usd: float, rate: float, round_trip: bool = True) -> float:
    """Currency conversion cost on money moved, in dollars.

    A CAD account buying US-listed options converts on the way in and back on
    the way out. Wealthsimple's spread is 1.5% below US$10k of conversions,
    stepping to 1.0% / 0.5% / 0% at $25k / $100k thresholds, or 0% outright with
    a USD account at $10/month. Unlike a contract fee this scales with the size
    of the position, so it hits long premium hardest and does not shrink by
    trading fewer legs.
    """
    sides = 2 if round_trip else 1
    return abs(float(cash_usd)) * float(rate) * sides


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
                 commission_per_contract: float = FALLBACK_COMMISSION_PER_CONTRACT,
                 fx_rate: float = 0.0,
                 multiplier: float = 100.0,
                 surface: Optional[SpreadSurface] = None):
        self.table = table or {}
        self.default_half_spread = float(default_half_spread)
        self.commission_per_contract = float(commission_per_contract)
        self.fx_rate = float(fx_rate)
        self.multiplier = float(multiplier)
        self.surface = surface

    def half_spread(self, strategy: str, *, mid: Optional[float] = None,
                    abs_delta: Optional[float] = None,
                    dte: Optional[float] = None,
                    open_interest: Optional[float] = None) -> float:
        """Half-spread in dollars per share.

        With full contract context and a fitted surface, the contract prices
        itself: a strategy name is not a contract, and the per-strategy
        constant charges a Bull Put's cheap long wing the same as its short
        leg. Without full context — or without a surface — this is exactly the
        previous per-strategy lookup, so existing callers are unaffected.

        Partial context does NOT half-guess a cell. A mid with no delta cannot
        locate a bucket, and inventing the missing dimension is how a number
        that describes something other than its label gets used.
        """
        fallback = half_spread_for(strategy, self.table, self.default_half_spread)
        if (self.surface is not None and mid is not None and mid > 0
                and abs_delta is not None and dte is not None):
            return self.surface.half_spread(
                mid, abs_delta=abs_delta, dte=dte,
                open_interest=open_interest, default=fallback / float(mid))
        return fallback

    def half_spread_with_provenance(
            self, strategy: str, *, n_legs: int,
            mid: Optional[float] = None,
            abs_delta: Optional[float] = None,
            dte: Optional[float] = None,
            open_interest: Optional[float] = None) -> Tuple[float, str]:
        """Half-spread in dollars per share, with where the number came from.

        A cost that cannot be told apart from a measurement is how an invented
        number gets quoted as fact, so the provenance travels with the value.

        MULTI-LEG STRUCTURES REFUSE SURFACE PRICING. The surface returns a
        RELATIVE half-spread and multiplies by the mid it is handed. For a
        spread the stored price is a NET CREDIT, not a leg mid — a $1.40 credit
        on a $30-wide Bull Put is not the mid of anything — so multiplying by it
        yields a number that is not a spread cost at all. Pricing a spread needs
        per-leg mids, which the ledger does not record. Until it does, spreads
        keep the per-strategy constant and say so.
        """
        fallback = half_spread_for(strategy, self.table, self.default_half_spread)
        if n_legs != 1:
            return fallback, "unpriceable_multi_leg"
        if (self.surface is None or mid is None or float(mid) <= 0
                or abs_delta is None or dte is None):
            return fallback, "strategy_table"
        rel, prov = self.surface.relative(
            abs_delta=abs_delta, dte=dte, open_interest=open_interest,
            default=fallback / float(mid))
        return rel * float(mid), prov

    def friction(self, strategy: str, n_legs: int, entry_credit: float = 0.0,
                 round_trip: bool = True, *,
                 mid: Optional[float] = None,
                 abs_delta: Optional[float] = None,
                 dte: Optional[float] = None,
                 open_interest: Optional[float] = None) -> float:
        """Friction per share: spread and commissions, plus currency conversion
        on the premium actually moved.

        With full single-leg contract context and a fitted surface, the contract
        prices itself. Without it — which is every existing caller — this is
        exactly the previous per-strategy lookup, so nothing already wired
        changes behaviour.
        """
        half, _prov = self.half_spread_with_provenance(
            strategy, n_legs=n_legs, mid=mid, abs_delta=abs_delta, dte=dte,
            open_interest=open_interest)
        cost = round_trip_friction(
            n_legs=n_legs,
            half_spread=half,
            commission_per_contract=self.commission_per_contract,
            round_trip=round_trip,
        )
        if self.fx_rate and entry_credit:
            cash = abs(float(entry_credit)) * self.multiplier
            cost += fx_cost(cash, self.fx_rate, round_trip) / self.multiplier
        return cost

    def friction_fraction(self, strategy: str, entry_credit: float, n_legs: int,
                          round_trip: bool = True, *,
                          mid: Optional[float] = None,
                          abs_delta: Optional[float] = None,
                          dte: Optional[float] = None,
                          open_interest: Optional[float] = None) -> float:
        """Friction as a fraction of the credit taken. 0.0 when there is no credit
        to measure against — a missing credit is not a free trade, it is a row
        the caller should skip."""
        credit = float(entry_credit or 0)
        if credit <= 0:
            return 0.0
        return self.friction(strategy, n_legs, credit, round_trip,
                             mid=mid, abs_delta=abs_delta, dte=dte,
                             open_interest=open_interest) / credit


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
                        commission_per_contract: float = 0.0,
                        fx_rate: float = 0.0) -> CostModel:
    """CostModel backed by the archive, falling back to the flat constant if the
    archive is absent or unreadable."""
    try:
        table = measure_half_spreads(archive_db, ledger_db)
    except sqlite3.Error:
        table = {}
    return CostModel(table, default_half_spread, commission_per_contract, fx_rate)
