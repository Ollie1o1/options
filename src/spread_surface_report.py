"""What the measured surface does to the closed book, in two disjoint tiers.

Coverage genuinely differs across the ledger, so this never averages the weak
tier into the strong one.

Tier 1 trades join a two-sided archived quote on their own entry date, so they
carry real open interest and price on the full 3D surface.

Tier 2 trades have entry_delta and a computable DTE but no archived quote — the
ledger has no open_interest column — so they price on the OI-collapsed
marginal. They are a LOWER BOUND on cost: the surface is fit on 15 liquid
symbols while the ledger spans 91 tickers, and extrapolating liquid-name
spreads onto the illiquid tail understates friction. Understating is the
direction that flatters a book whose measured PF is 1.044.

This module renders. It does not decide anything.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.spread_surface import DEFAULT_ARCHIVE, SpreadSurface, load_surface

DEFAULT_LEDGER = "paper_trades.db"


@dataclass(frozen=True)
class TierRow:
    entry_id: int
    strategy: str
    tier: int
    old_friction: float
    new_friction: float
    provenance: str


_TIER1_SQL = """
    SELECT tr.entry_id, tr.strategy_name, tr.entry_price,
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
# an empty tier 1 would silently empty tier 2 as well.
_REST_SQL = """
    SELECT tr.entry_id, tr.strategy_name, tr.entry_price, tr.entry_delta,
           julianday(tr.expiration) - julianday(tr.date)
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
        if not mid or mid <= 0:
            continue
        rel, prov = surface.relative(abs_delta=ad, dte=dte, open_interest=oi,
                                     default=old_half_spread / float(mid))
        tier1.append(TierRow(eid, strat or "?", 1, old_half_spread,
                             rel * float(mid), prov))

    tier2: List[TierRow] = []
    uncovered: List[int] = []
    for eid, strat, mid, ad, dte in rest:
        if ad is None or dte is None or not mid or mid <= 0:
            uncovered.append(eid)
            continue
        # No open interest in the ledger: collapse that dimension by asking for
        # the cell the delta/DTE pair lands in, letting the surface fall back.
        rel, prov = surface.relative(abs_delta=ad, dte=dte, open_interest=None,
                                     default=old_half_spread / float(mid))
        tier2.append(TierRow(eid, strat or "?", 2, old_half_spread,
                             rel * float(mid), prov))

    return {"tier1": tier1, "tier2": tier2, "uncovered": uncovered}


def _by_strategy(rows: List[TierRow]) -> Dict[str, List[TierRow]]:
    out: Dict[str, List[TierRow]] = {}
    for r in rows:
        out.setdefault(r.strategy, []).append(r)
    return out


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
                 f"{stamp.get('refit_command', 'python -m src.spread_surface --fit')}")
    lines.append("")
    lines += _tier_block(
        "Tier 1 — archived quote, full surface",
        "real open interest; this is the trustworthy number",
        tiers["tier1"])
    lines += _tier_block(
        "Tier 2 — no archived quote, open interest collapsed",
        "a LOWER BOUND on cost: fit on 15 liquid symbols, applied to a "
        "91-ticker book",
        tiers["tier2"])
    lines.append(f"  uncovered: {len(tiers['uncovered'])} closed trades "
                 f"lack both a quote and an entry delta")
    lines.append("")
    return "\n".join(lines)
