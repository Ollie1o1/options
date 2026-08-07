"""What a setup costs to trade, before it is asked whether it works.

The allocation backtest settled the ordering of problems for this desk: the
binding constraint on short premium is not *when* to trade, it is the spread
paid to get in. A Bull Put whose round trip costs 68% of its credit needs a win
rate no signal in this repo has ever produced; a Bear Call at 23% does not. That
difference is invisible on a board that only shows the hypothesis, which is why
every setup carries a friction figure beside it.

The number is MEASURED, in this order:

  1. the setup's own `cost_profile`, if a backtest has landed one on it;
  2. the ledger's live quotes — median |cross - mid| per share, per structure,
     over trades that recorded both, requiring MIN_OBSERVATIONS to set a constant;
  3. the recorded 2026-08-06 derivation, kept only as a fallback and labelled.

Step 3 exists so the board still renders on a machine with no ledger. It is
deliberately the last resort: a constant that cannot move when reality does is
how the flat $0.05/share assumption survived long enough to invert a conclusion
(docs/EXECUTION_TRUTH.md, and the cost-wall re-derivation of 2026-08-06).

Structures the ledger has never traded report UNMEASURED, never zero. A missing
number is not a free trade.
"""
from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from statistics import median
from typing import Any, Dict, Optional
from src.paths import repo_path

DEFAULT_LEDGER = "paper_trades.db"
DEFAULT_CONFIG = "config.json"

# Below this many matched trades a bucket cannot set a cost constant. Same line
# the archive-backed cost model draws (src/execution_costs.py).
MIN_OBSERVATIONS = 10

# Fallback only. Median crossing cost per share and median position price per
# share, over the 194 ledger trades that recorded both a mid and a crossed fill,
# re-derived 2026-08-06 by exactly the query `measure_from_ledger` runs — no
# quote filtering. Round trip is 2x per_share, so bull_put reads 68% of credit
# and bear_call 23%, the figures in docs/SCORER_IMPROVEMENTS.md §5.
#
# An earlier derivation published 53% for bull_put because it dropped the 4
# trades whose crossed price was not a tradeable credit. Those are real quotes
# on real chains; excluding them measures the friction of the trades you would
# have wanted, not the friction the chain offered. The unfiltered figure is the
# one this desk shows, so the fallback and the live measurement agree by method.
_RECORDED_SOURCE = "recorded 2026-08-06 (n=194 ledger fills)"

RECORDED: Dict[str, Dict[str, Any]] = {
    "bull_put":    {"per_share": 0.350, "credit": 1.025, "n": 30,
                    "source": _RECORDED_SOURCE},
    "bear_call":   {"per_share": 0.050, "credit": 0.44, "n": 41,
                    "source": _RECORDED_SOURCE},
    "iron_condor": {"per_share": 0.175, "credit": 9.645, "n": 59,
                    "source": _RECORDED_SOURCE},
    "short_put":   {"per_share": 0.100, "credit": 6.345, "n": 25,
                    "source": _RECORDED_SOURCE},
    "long_call":   {"per_share": 0.100, "credit": 8.3875, "n": 34,
                    "source": _RECORDED_SOURCE},
}

_DEFAULT_CEILING = 0.25


def _key(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "_")


@dataclass(frozen=True)
class FrictionProfile:
    """Round-trip cost of a structure, as dollars and as a share of its credit."""

    structure: str
    per_share: Optional[float]
    credit: Optional[float]
    round_trip: bool
    n: int
    source: str

    @property
    def measured(self) -> bool:
        return (self.per_share is not None and self.credit is not None
                and self.credit > 0 and self.n > 0)

    @property
    def sides(self) -> int:
        """Two crossings when the position is closed, one when it expires."""
        return 2 if self.round_trip else 1

    @property
    def cost_usd(self) -> Optional[float]:
        """Friction per contract, in dollars."""
        per_share = self.per_share
        if per_share is None:
            return None
        return self.sides * float(per_share) * 100.0

    @property
    def credit_usd(self) -> Optional[float]:
        credit = self.credit
        if credit is None:
            return None
        return abs(float(credit)) * 100.0

    @property
    def pct_of_credit(self) -> Optional[float]:
        cost, credit = self.cost_usd, self.credit_usd
        if cost is None or credit is None or credit <= 0 or self.n <= 0:
            return None
        return cost / credit

    def over_ceiling(self, limit: Optional[float] = None) -> bool:
        """True when the round trip eats more of the credit than config allows."""
        pct = self.pct_of_credit
        if pct is None:
            return False
        return pct > (ceiling() if limit is None else float(limit))


def ceiling(config_path: str = DEFAULT_CONFIG) -> float:
    """`auto_log.max_friction_to_credit` — the same ceiling that refuses trades."""
    try:
        with open(repo_path(config_path)) as f:
            cfg = json.load(f)
        value = (cfg.get("auto_log") or {}).get("max_friction_to_credit")
    except (OSError, ValueError, AttributeError):
        return _DEFAULT_CEILING
    if value is None or value in ("", 0, False):
        return _DEFAULT_CEILING
    try:
        return float(value)
    except (TypeError, ValueError):
        return _DEFAULT_CEILING


def measure_from_ledger(db_path: str = DEFAULT_LEDGER) -> Dict[str, Dict[str, Any]]:
    """Median crossing cost per structure, from trades that logged both prices.

    `entry_price_mid` is what the screener quoted; `entry_price_cross` is what
    crossing the spread actually paid. Their gap is the toll for one side, and
    it is the only friction figure in this repo that was neither assumed nor
    modelled.
    """
    if not os.path.exists(db_path):
        return {}
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            """
            SELECT strategy_name, entry_price_mid, entry_price_cross
            FROM trades
            WHERE entry_price_mid IS NOT NULL AND entry_price_cross IS NOT NULL
            """
        ).fetchall()
    except sqlite3.Error:
        return {}
    finally:
        conn.close()

    buckets: Dict[str, list] = {}
    for name, mid, cross in rows:
        if name is None or mid is None or cross is None:
            continue
        buckets.setdefault(_key(str(name)), []).append(
            (abs(float(cross) - float(mid)), abs(float(mid))))

    out: Dict[str, Dict[str, Any]] = {}
    for structure, vals in buckets.items():
        out[structure] = {
            "per_share": round(median([v[0] for v in vals]), 4),
            "credit": round(median([v[1] for v in vals]), 4),
            "n": len(vals),
            "source": f"ledger n={len(vals)}",
        }
    return out


def load_table(db_path: str = DEFAULT_LEDGER) -> Dict[str, Dict[str, Any]]:
    """Measured table over the recorded fallback, thin buckets discarded."""
    table = dict(RECORDED)
    for structure, cell in measure_from_ledger(db_path).items():
        if int(cell.get("n", 0)) >= MIN_OBSERVATIONS:
            table[structure] = cell
    return table


def profile_for(record: Any,
                table: Optional[Dict[str, Dict[str, Any]]] = None) -> FrictionProfile:
    """Friction for one setup: its own measured profile, else its structure's."""
    spec = record.spec
    structure = _key(spec.structure)
    round_trip = not bool(spec.exit.get("hold_to_expiry"))

    own: Dict[str, Any] = getattr(record, "cost_profile", None) or {}
    if own.get("unmeasured"):
        # An explicit refusal to quote the structure-wide figure at this setup.
        # A bull put on SPY does not pay what a bull put on a $30 single name
        # pays, and printing the single-name number here would be a wrong
        # answer wearing the authority of a measurement.
        return FrictionProfile(structure, None, None, round_trip, 0,
                               str(own.get("why") or "unmeasured"))
    cell: Optional[Dict[str, Any]] = (
        own if own.get("per_share") is not None else None)
    if cell is None:
        cell = (load_table() if table is None else table).get(structure)
    if not cell:
        return FrictionProfile(structure, None, None, round_trip, 0, "unmeasured")

    return FrictionProfile(
        structure=structure,
        per_share=cell.get("per_share"),
        credit=cell.get("credit"),
        round_trip=round_trip,
        n=int(cell.get("n", 0) or 0),
        source=str(cell.get("source", "unmeasured")),
    )


def format_cell(profile: FrictionProfile) -> str:
    """Round trip as a share of credit — the unit the ceiling is written in."""
    pct = profile.pct_of_credit
    if pct is None:
        return "—"
    return f"{pct:.0%}"


def style_for(profile: FrictionProfile) -> str:
    """Semantic style name for the friction cell. Never a raw colour."""
    pct = profile.pct_of_credit
    if pct is None:
        return "muted"
    limit = ceiling()
    if pct > limit:
        return "bad"
    if pct > limit / 2.0:
        return "warn"
    return "good"


def describe(profile: FrictionProfile) -> str:
    """One line for the detail view, including where the number came from."""
    cost, credit = profile.cost_usd, profile.credit_usd
    if not profile.measured or cost is None or credit is None:
        if profile.source and profile.source != "unmeasured":
            return f"unmeasured — {profile.source}"
        return f"unmeasured — no matched quotes for {profile.structure}"
    trip = "round trip" if profile.round_trip else "one side (held to expiry)"
    verdict = " — OVER the ceiling" if profile.over_ceiling() else ""
    return (f"${cost:.2f} {trip} against ${credit:.0f} "
            f"credit = {format_cell(profile)} of credit "
            f"(ceiling {ceiling():.0%}){verdict}  ·  {profile.source}")
