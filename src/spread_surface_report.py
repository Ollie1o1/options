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
the ledger has no open_interest column, so open interest is genuinely UNKNOWN
for these trades, not merely missing from one lookup. Picking either extreme
would assert knowledge we don't have, so two numbers are reported instead of
one: a CENTRAL estimate (the OI-collapsed marginal — `SpreadSurface.
oi_collapsed_relative`, median relative half-spread across every OI bucket
for the delta/DTE cell) and a CONSERVATIVE bound (the illiquid-bucket-0 pin —
`SpreadSurface.relative(..., open_interest=None)`, which `cell_key` resolves
to the most illiquid bucket). The true cost lies somewhere between the two.
Both are also fit on 15 liquid symbols while the ledger spans 91 tickers, and
extrapolating liquid-name spreads onto the illiquid tail understates
friction in the same direction for both numbers — the direction that
flatters a book whose measured PF is 1.044.

`relative(..., open_interest=None)` and `oi_collapsed_relative` are NOT
interchangeable despite both nominally meaning "I don't know the open
interest": `cell_key` maps a missing OI to bucket 0, so `relative()` returns
that exact cell's own value (provenance "cell") whenever bucket 0 is
populated, which it always is in the real surface — the OI-collapsed rung of
its fallback ladder is dead code for a full surface. `oi_collapsed_relative`
starts at that rung directly. Using `relative(..., open_interest=None)` alone
and calling the result a lower bound was Tier 2's original defect: bucket 0
is the highest-cost bucket, so that number is a conservative UPPER bound on
open-interest uncertainty, not a lower one.

`no_leg_mid` trades are MULTI-LEG structures (`net_credit IS NOT NULL`) with no
archived quote. Their `entry_price` is a net credit, not a leg mid, and no leg
mid exists anywhere else in the ledger for them. There is no correct way to
price these, so they are counted and left unpriced rather than multiplying a
leg-calibrated relative half-spread by a net credit and reporting a number
that is wrong by several times.

`net_credit IS NOT NULL` is the multi-leg test — structural, not the strategy
name. This repo shipped a defect where every Bear Call was labelled "Bull
Put" for months, so a name is not a structure.

The "baseline" column is what `execution_costs.load_measured_model()` charges
TODAY — the same measured per-strategy table `short_premium_gate.py` reads
for its live verdicts and `paper_manager.py` falls back to when it has no
per-leg quote — not the flat $0.05 `CostModel` default this branch's
`classify_tiers` used to hardcode. Comparing the surface against that dead
constant instead of the live model can flip the SIGN of "change" for a
strategy — a surface that looks more expensive than a flat $0.05 can be
cheaper than what is actually charged today, or vice versa. The numbers move
with every refit and every `execution_costs` measurement; do not hardcode a
comparison here.

`execution_costs.load_measured_model` is itself a measurement with its own
floor: below `execution_costs.MIN_OBSERVATIONS` matched quotes for a
strategy, `half_spread_for` falls back to ITS OWN $0.05 default — a table
entry existing is not the same as that entry being used (see
`execution_costs.is_measured`). Such a row's baseline is that default, not a
measurement, and is marked `*` in the rendered table with a footnote — the
note text never claims a "*" row is measured when it is not.

Every row also carries its own lookup provenance (`TierRow.provenance` /
`conservative_provenance`): "cell" means a real measured cell, anything else
("oi_collapsed", "dte_collapsed", "global", "caller_default") is a fallback
rung. A fallback that renders identically to a measurement is how an invented
number gets quoted as fact — see `spread_surface.SpreadSurface.relative`'s
own docstring. `render_report` refuses outright when the surface has zero
cells (unfitted): every row would otherwise be a `caller_default` fallback
indistinguishable on the page from a real measurement.

This module renders. It does not decide anything.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from src.execution_costs import CostModel, is_measured, load_measured_model
from src.spread_surface import (DEFAULT_ARCHIVE, REFIT_COMMAND, SpreadSurface,
                                load_surface)

DEFAULT_LEDGER = "paper_trades.db"

# All dollar figures in this module are $/share of half-spread.
UNITS = "$/share (half-spread)"


@dataclass(frozen=True)
class TierRow:
    entry_id: int
    strategy: str
    tier: int
    baseline_friction: float
    new_friction: float
    provenance: str
    # Tier 2 only: open interest is unknown for these trades (the ledger has
    # no open_interest column), so a single number would assert knowledge we
    # don't have. `new_friction`/`provenance` above carry the CENTRAL
    # estimate (the OI-collapsed marginal); these two carry the CONSERVATIVE
    # bound (the illiquid-bucket-0 pin). Tier 1 rows leave both None — they
    # have real open interest and need only one number.
    conservative_friction: Optional[float] = None
    conservative_provenance: Optional[str] = None
    # Whether `baseline_friction` is a real execution_costs measurement for
    # this strategy (n >= execution_costs.MIN_OBSERVATIONS) or that module's
    # OWN $0.05 fallback below its floor. A strategy can have a table entry
    # and still not be measured — see execution_costs.is_measured. Defaults
    # True so existing positional TierRow(...) calls in tests are unaffected.
    baseline_measured: bool = True


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
    WHERE (tr.status IS NULL OR tr.status != 'OPEN')
      AND cs.bid > 0 AND cs.ask > cs.bid
      AND cs.delta IS NOT NULL
      AND tr.expiration IS NOT NULL AND tr.date IS NOT NULL
    GROUP BY tr.entry_id
"""

# The exclusion clause is built rather than formatted with a bare "NULL":
# `entry_id NOT IN (NULL)` evaluates to NULL for every row, which is falsy, so
# an empty tier 1 would silently empty the rest of the buckets too.
#
# `tr.status IS NULL OR tr.status != 'OPEN'` (both here and in _TIER1_SQL,
# not the bare `!= 'OPEN'`) is deliberate: SQL's `NULL != 'OPEN'` evaluates to
# NULL, which is falsy, so a NULL-status row would be silently dropped from
# every bucket, quietly breaking the report's exhaustiveness claim. None
# exist in today's ledger; a NULL status is treated as "not stated OPEN" —
# i.e. eligible for classification like any other closed row — rather than
# hidden.
_REST_SQL = """
    SELECT tr.entry_id, tr.strategy_name, tr.entry_price, tr.net_credit,
           tr.entry_delta, julianday(tr.expiration) - julianday(tr.date)
    FROM trades tr
    WHERE (tr.status IS NULL OR tr.status != 'OPEN'){exclusion}
"""


def classify_tiers(ledger_db: str = DEFAULT_LEDGER,
                   archive_db: str = DEFAULT_ARCHIVE,
                   surface: Optional[SpreadSurface] = None,
                   cost_model: Optional[CostModel] = None) -> Dict[str, List[Any]]:
    """Classify the closed book into pricing tiers.

    `cost_model` supplies the "baseline" figure each row is compared against
    — what the system actually charges TODAY (default:
    `execution_costs.load_measured_model()`, the same measured per-strategy
    table the live gate and ledger use). That model is itself measured with
    a floor: a strategy below its MIN_OBSERVATIONS still gets a baseline
    from `half_spread`, but it is that model's own $0.05 default, not a
    measurement — `TierRow.baseline_measured` (via `execution_costs.
    is_measured`) records which one a given row got. Tests MUST inject an
    explicit `CostModel` — the default reads data/chain_archive.db and
    paper_trades.db.
    """
    surface = surface if surface is not None else load_surface()
    if cost_model is None:
        cost_model = load_measured_model(archive_db, ledger_db,
                                         commission_per_contract=0.0,
                                         fx_rate=0.0)
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
        strategy = strat or "?"
        # Context-free call (no mid/delta/dte/oi): CostModel.half_spread falls
        # straight to its per-strategy measured lookup, never the surface —
        # this is what the LIVE system charges today, not the surface being
        # evaluated here.
        baseline = cost_model.half_spread(strategy)
        baseline_measured = is_measured(strategy, cost_model.table)
        rel, prov = surface.relative(abs_delta=ad, dte=dte, open_interest=oi,
                                     default=baseline / float(mid))
        tier1.append(TierRow(eid, strategy, 1, baseline, rel * float(mid),
                             prov, baseline_measured=baseline_measured))

    tier2: List[TierRow] = []
    no_leg_mid: List[UnpricedRow] = []
    uncovered: List[int] = []
    for eid, strat, entry_price, net_credit, ad, dte in rest:
        strategy = strat or "?"
        if ad is None or dte is None:
            uncovered.append(eid)
            continue
        if net_credit is not None:
            # Multi-leg structure (paper_manager.log_spread/log_condor stamp
            # net_credit; single legs never do) with no archived quote. Its
            # entry_price is a net credit, not a leg mid, and there is no leg
            # mid anywhere else in the ledger for it — refuse to price it.
            no_leg_mid.append(UnpricedRow(eid, strategy))
            continue
        mid = entry_price
        if not mid or mid <= 0:
            uncovered.append(eid)
            continue
        # Single-leg trade: entry_price genuinely is this option's mid. No
        # open interest in the ledger, and it is unknown for the trade
        # itself — not merely omitted from this lookup — so report both
        # ends of the range rather than picking one. `oi_collapsed_relative`
        # gives the genuine marginal (central); `relative(...,
        # open_interest=None)` gives the illiquid-bucket-0 pin
        # (conservative). See the module docstring: these are NOT the same
        # number for a populated surface.
        baseline = cost_model.half_spread(strategy)
        baseline_measured = is_measured(strategy, cost_model.table)
        default = baseline / float(mid)
        central_rel, central_prov = surface.oi_collapsed_relative(
            abs_delta=ad, dte=dte, default=default)
        conservative_rel, conservative_prov = surface.relative(
            abs_delta=ad, dte=dte, open_interest=None, default=default)
        tier2.append(TierRow(
            eid, strategy, 2, baseline,
            central_rel * float(mid), central_prov,
            conservative_rel * float(mid), conservative_prov,
            baseline_measured=baseline_measured))

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


def _provenance_counts(rows: List[TierRow], attr: str) -> Dict[str, int]:
    """How many rows resolved off a real measured cell versus a fallback
    rung. A row's `provenance`/`conservative_provenance` is the only thing
    that distinguishes a measurement from an invented number once both are
    printed as a plain float — this is what makes that distinction visible."""
    counts: Dict[str, int] = {}
    for r in rows:
        key = getattr(r, attr)
        if key is None:
            continue
        counts[key] = counts.get(key, 0) + 1
    return counts


def _format_provenance(counts: Dict[str, int]) -> str:
    if not counts:
        return "n/a"
    return ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))


def _baseline_marker(rs: List[TierRow]) -> str:
    """" *" when this strategy's baseline is execution_costs.py's OWN $0.05
    default (fewer than its MIN_OBSERVATIONS matched quotes) rather than a
    measurement — the same distinction the surface side already carries as
    `provenance`, made visible here too rather than left implicit in a note
    that would otherwise say "measured" for a number that is not."""
    return "" if all(r.baseline_measured for r in rs) else "  *"


_BASELINE_FOOTNOTE = ("    * baseline is execution_costs.py's own $0.05 "
                     "default for this strategy (fewer than its "
                     "MIN_OBSERVATIONS matched quotes), not a measurement")


def _tier_block(title: str, note: str, rows: List[TierRow]) -> List[str]:
    lines = [f"  {title}  (n={len(rows)})", f"    {note}", ""]
    if not rows:
        lines.append("    no trades in this tier")
        return lines + [""]
    lines.append(f"    {'strategy':<14}{'n':>5}{'baseline':>10}"
                 f"{'surface':>10}{'change':>10}")
    any_unmeasured = False
    for strat, rs in sorted(_by_strategy(rows).items()):
        base = sum(r.baseline_friction for r in rs) / len(rs)
        new = sum(r.new_friction for r in rs) / len(rs)
        marker = _baseline_marker(rs)
        any_unmeasured = any_unmeasured or bool(marker)
        lines.append(f"    {strat:<14}{len(rs):>5}{base:>10.3f}"
                     f"{new:>10.3f}{new - base:>+10.3f}{marker}")
    if any_unmeasured:
        lines.append(_BASELINE_FOOTNOTE)
    lines.append(f"    provenance: "
                 f"{_format_provenance(_provenance_counts(rows, 'provenance'))}")
    return lines + [""]


def _tier2_block(title: str, note: str, rows: List[TierRow]) -> List[str]:
    """Tier 2 has two figures per row, not one: open interest is unknown for
    these trades, so a central estimate (OI-collapsed marginal) and a
    conservative bound (illiquid-bucket-0 pin) are both shown rather than
    picking one and calling it the answer."""
    lines = [f"  {title}  (n={len(rows)})", f"    {note}", ""]
    if not rows:
        lines.append("    no trades in this tier")
        return lines + [""]
    lines.append(f"    {'strategy':<14}{'n':>5}{'baseline':>10}"
                 f"{'central':>10}{'conserv.':>10}")
    any_unmeasured = False
    for strat, rs in sorted(_by_strategy(rows).items()):
        base = sum(r.baseline_friction for r in rs) / len(rs)
        central = sum(r.new_friction for r in rs) / len(rs)
        # classify_tiers always populates conservative_friction for a tier 2
        # row; a row that somehow lacks it must not silently render
        # conservative == central (an unreachable rung of the caller's
        # fallback ladder rendering as a real, matching value would be the
        # same defect this module refuses everywhere else). Raise instead.
        conservative_values: List[float] = []
        for r in rs:
            cf = r.conservative_friction
            if cf is None:
                raise ValueError(
                    f"tier 2 row entry_id={r.entry_id} ({strat}) has no "
                    f"conservative_friction; classify_tiers must always "
                    f"populate it for a tier 2 row")
            conservative_values.append(cf)
        conservative = sum(conservative_values) / len(rs)
        marker = _baseline_marker(rs)
        any_unmeasured = any_unmeasured or bool(marker)
        lines.append(f"    {strat:<14}{len(rs):>5}{base:>10.3f}"
                     f"{central:>10.3f}{conservative:>10.3f}{marker}")
    if any_unmeasured:
        lines.append(_BASELINE_FOOTNOTE)
    lines.append(
        f"    central provenance: "
        f"{_format_provenance(_provenance_counts(rows, 'provenance'))}")
    lines.append(
        f"    conservative provenance: "
        f"{_format_provenance(_provenance_counts(rows, 'conservative_provenance'))}")
    return lines + [""]


def render_report(tiers: Dict[str, List[Any]], surface: SpreadSurface) -> str:
    """Render the reprice report. Dollars per share of half-spread."""
    if not surface.cells:
        return "\n".join([
            "",
            "  MEASURED SPREAD SURFACE — REPRICE REPORT",
            "",
            "  REFUSING TO RENDER: the surface has 0 cells (unfitted).",
            "  Every figure below would be a caller-default fallback with",
            "  nothing to distinguish it on the page from a real",
            "  measurement. Fit the surface first, then re-run --report:",
            "",
            f"    {REFIT_COMMAND}",
            "",
        ])
    stamp = surface.stamp
    lines = ["", "  MEASURED SPREAD SURFACE — REPRICE REPORT", ""]
    fit = stamp.get("fit_date", "unknown")
    lines.append(f"    surface fit {fit}; refit with "
                 f"{stamp.get('refit_command', REFIT_COMMAND)}")
    lines.append(f"    all dollar figures below are {UNITS}")
    lines.append("")
    lines += _tier_block(
        "Tier 1 — archived quote, full surface",
        "real open interest; this is the trustworthy number. baseline is "
        "what execution_costs.py charges TODAY (load_measured_model): "
        "measured per strategy where enough quotes exist, else that "
        "module's own $0.05 default ('*' rows below)",
        tiers["tier1"])
    lines += _tier2_block(
        "Tier 2 — single-leg, no archived quote, open interest UNKNOWN",
        "open interest is unknown for these trades, not merely missing "
        "from one lookup: central collapses across OI buckets (the "
        "genuine marginal); conserv. assumes the most illiquid bucket "
        "(the worst case). True cost lies between the two. Both are fit "
        "on 15 liquid symbols, applied to a 91-ticker book. baseline is "
        "what execution_costs.py charges TODAY, same as tier 1 (measured "
        "where enough quotes exist, else its $0.05 default — '*' rows "
        "below)",
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
