"""What the measured surface does to the closed book, in four disjoint buckets.

Coverage genuinely differs across the ledger, so this never averages a weak
bucket into a stronger one.

Tier 1 trades join a two-sided archived quote on their own entry date, so they
carry real open interest and price on the full 3D surface. They price off the
archived quote's OWN mid — never off `trades.entry_price` — because for a
multi-leg structure `entry_price` is the NET CREDIT across legs (see
paper_manager.log_spread / log_condor), not any single option's mid.

Tier 2 trades are SINGLE-LEG trades (no `net_credit`, so `entry_price` really
is a leg mid) with entry_delta and a computable DTE but no archived quote —
the ledger has no open_interest column — so they price on the OI-collapsed
marginal. They are a LOWER BOUND on cost: the surface is fit on 15 liquid
symbols while the ledger spans 91 tickers, and extrapolating liquid-name
spreads onto the illiquid tail understates friction. Understating is the
direction that flatters a book whose measured PF is 1.044.

`no_leg_mid` trades are MULTI-LEG structures (`net_credit IS NOT NULL`) with no
archived quote. Their `entry_price` is a net credit, not a leg mid, and no leg
mid exists anywhere else in the ledger for them. There is no correct way to
price these, so they are counted and left unpriced rather than multiplying a
leg-calibrated relative half-spread by a net credit and reporting a number
that is wrong by several times.

`net_credit IS NOT NULL` is the multi-leg test — structural, not the strategy
name. This repo shipped a defect where every Bear Call was labelled "Bull
Put" for months, so a name is not a structure.

This module renders. It does not decide anything.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.spread_surface import (DEFAULT_ARCHIVE, REFIT_COMMAND, SpreadSurface,
                                load_surface)

DEFAULT_LEDGER = "paper_trades.db"


@dataclass(frozen=True)
class TierRow:
    entry_id: int
    strategy: str
    tier: int
    old_friction: float
    new_friction: float
    provenance: str


@dataclass(frozen=True)
class UnpricedRow:
    """A closed trade counted but deliberately NOT priced: a multi-leg
    structure whose `entry_price` is a net credit, with no archived quote to
    supply a real leg mid."""
    entry_id: int
    strategy: str


_TIER1_SQL = """
    SELECT tr.entry_id, tr.strategy_name,
           (cs.bid + cs.ask) / 2.0,
           abs(cs.delta), julianday(tr.expiration) - julianday(tr.date),
           cs.open_interest
    FROM trades tr
    JOIN ar.chain_snapshots cs
      ON cs.symbol = tr.ticker AND cs.strike = tr.strike
     AND cs.expiration = tr.expiration
     AND substr(cs.type, 1, 1) = substr(tr.type, 1, 1)
     AND cs.snap_date = substr(tr.date, 1, 10)
    WHERE tr.status != 'OPEN' AND cs.bid > 0 AND cs.ask > cs.bid
    GROUP BY tr.entry_id
"""

# The exclusion clause is built rather than formatted with a bare "NULL":
# `entry_id NOT IN (NULL)` evaluates to NULL for every row, which is falsy, so
# an empty tier 1 would silently empty the rest of the buckets too.
_REST_SQL = """
    SELECT tr.entry_id, tr.strategy_name, tr.entry_price, tr.net_credit,
           tr.entry_delta, julianday(tr.expiration) - julianday(tr.date)
    FROM trades tr
    WHERE tr.status != 'OPEN'{exclusion}
"""


def classify_tiers(ledger_db: str = DEFAULT_LEDGER,
                   archive_db: str = DEFAULT_ARCHIVE,
                   surface: Optional[SpreadSurface] = None,
                   old_half_spread: float = 0.05) -> Dict[str, List[Any]]:
    surface = surface if surface is not None else load_surface()
    con = sqlite3.connect(ledger_db)
    try:
        con.execute("ATTACH DATABASE ? AS ar", (archive_db,))
        tier1_raw = con.execute(_TIER1_SQL).fetchall()
        seen = [r[0] for r in tier1_raw]
        if seen:
            exclusion = (" AND tr.entry_id NOT IN ("
                         + ",".join("?" for _ in seen) + ")")
        else:
            exclusion = ""
        rest = con.execute(_REST_SQL.format(exclusion=exclusion),
                           seen).fetchall()
    finally:
        con.close()

    tier1: List[TierRow] = []
    for eid, strat, mid, ad, dte, oi in tier1_raw:
        # `mid` here is the archived quote's own (bid+ask)/2 — the true mid of
        # the leg that matched, NOT tr.entry_price. For a multi-leg structure
        # entry_price is a net credit; using it here would price friction off
        # the wrong number even though a real leg mid was available.
        if not mid or mid <= 0:
            continue
        rel, prov = surface.relative(abs_delta=ad, dte=dte, open_interest=oi,
                                     default=old_half_spread / float(mid))
        tier1.append(TierRow(eid, strat or "?", 1, old_half_spread,
                             rel * float(mid), prov))

    tier2: List[TierRow] = []
    no_leg_mid: List[UnpricedRow] = []
    uncovered: List[int] = []
    for eid, strat, entry_price, net_credit, ad, dte in rest:
        if ad is None or dte is None:
            uncovered.append(eid)
            continue
        if net_credit is not None:
            # Multi-leg structure (paper_manager.log_spread/log_condor stamp
            # net_credit; single legs never do) with no archived quote. Its
            # entry_price is a net credit, not a leg mid, and there is no leg
            # mid anywhere else in the ledger for it — refuse to price it.
            no_leg_mid.append(UnpricedRow(eid, strat or "?"))
            continue
        mid = entry_price
        if not mid or mid <= 0:
            uncovered.append(eid)
            continue
        # Single-leg trade: entry_price genuinely is this option's mid. No
        # open interest in the ledger: collapse that dimension by asking for
        # the cell the delta/DTE pair lands in, letting the surface fall back.
        rel, prov = surface.relative(abs_delta=ad, dte=dte, open_interest=None,
                                     default=old_half_spread / float(mid))
        tier2.append(TierRow(eid, strat or "?", 2, old_half_spread,
                             rel * float(mid), prov))

    return {"tier1": tier1, "tier2": tier2, "no_leg_mid": no_leg_mid,
            "uncovered": uncovered}


def _by_strategy(rows: List[TierRow]) -> Dict[str, List[TierRow]]:
    out: Dict[str, List[TierRow]] = {}
    for r in rows:
        out.setdefault(r.strategy, []).append(r)
    return out


def _count_block(title: str, note: str, rows: List[UnpricedRow]) -> List[str]:
    """Render a bucket that is counted but never priced."""
    lines = [f"  {title}  (n={len(rows)})", f"    {note}", ""]
    if not rows:
        lines.append("    no trades in this bucket")
        return lines + [""]
    counts: Dict[str, int] = {}
    for r in rows:
        counts[r.strategy] = counts.get(r.strategy, 0) + 1
    lines.append(f"    {'strategy':<14}{'n':>5}")
    for strat, n in sorted(counts.items()):
        lines.append(f"    {strat:<14}{n:>5}")
    return lines + [""]


def _tier_block(title: str, note: str, rows: List[TierRow]) -> List[str]:
    lines = [f"  {title}  (n={len(rows)})", f"    {note}", ""]
    if not rows:
        lines.append("    no trades in this tier")
        return lines + [""]
    lines.append(f"    {'strategy':<14}{'n':>5}{'charged':>10}"
                 f"{'measured':>10}{'change':>10}")
    for strat, rs in sorted(_by_strategy(rows).items()):
        old = sum(r.old_friction for r in rs) / len(rs)
        new = sum(r.new_friction for r in rs) / len(rs)
        lines.append(f"    {strat:<14}{len(rs):>5}{old:>10.3f}"
                     f"{new:>10.3f}{new - old:>+10.3f}")
    return lines + [""]


def render_report(tiers: Dict[str, List[Any]], stamp: Dict[str, Any]) -> str:
    """Render the reprice report. Dollars per share of half-spread."""
    lines = ["", "  MEASURED SPREAD SURFACE — REPRICE REPORT", ""]
    fit = stamp.get("fit_date", "unknown")
    lines.append(f"    surface fit {fit}; refit with "
                 f"{stamp.get('refit_command', REFIT_COMMAND)}")
    lines.append("")
    lines += _tier_block(
        "Tier 1 — archived quote, full surface",
        "real open interest; this is the trustworthy number",
        tiers["tier1"])
    lines += _tier_block(
        "Tier 2 — single-leg, no archived quote, open interest collapsed",
        "a LOWER BOUND on cost: fit on 15 liquid symbols, applied to a "
        "91-ticker book",
        tiers["tier2"])
    lines += _count_block(
        "No leg mid — multi-leg net credit, no archived quote",
        "entry_price here is a spread's NET CREDIT across legs, not a leg "
        "mid, and no leg mid exists anywhere else in the ledger; friction is "
        "NOT computed for these",
        tiers["no_leg_mid"])
    lines.append(f"  uncovered: {len(tiers['uncovered'])} closed trades "
                 f"lack both a quote and an entry delta")
    lines.append("")
    total = (len(tiers["tier1"]) + len(tiers["tier2"])
             + len(tiers["no_leg_mid"]) + len(tiers["uncovered"]))
    if total:
        unpriced_pct = 100.0 * len(tiers["no_leg_mid"]) / total
        lines.append(
            f"  {unpriced_pct:.0f}% of the closed book "
            f"({len(tiers['no_leg_mid'])} of {total} trades) is unpriced "
            f"(no_leg_mid); no book-wide total is offered here because "
            f"summing tier 1 and tier 2 alone would silently undercount it.")
        lines.append("")
    return "\n".join(lines)
