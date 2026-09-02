"""scripts/reprice_single_leg_book.py

Reprice the single-leg closed book under the measured spread surface
(data/spread_surface.json, PR #87) instead of whatever friction was actually
charged at close, and report net profit factor with a bootstrapped 95% CI
clustered on entry day, beside the gross figure.

Reporting only — writes nothing, changes no exit behaviour.

MULTI-LEG TRADES ARE REFUSED, NOT PRICED. `entry_price` on a spread is a NET
CREDIT across legs, not a single leg's mid — multiplying a leg-calibrated
relative half-spread by a net credit produces a number that is not a spread
cost (src/spread_surface_report.py's module docstring makes the same refusal
for the $/share question; this file makes it independently for the PF
question). `net_credit IS NOT NULL` is the structural multi-leg test, not the
strategy name — this repo shipped a defect where every Bear Call was labelled
"Bull Put" for months.

WHY THIS DOES NOT USE execution_costs.reprice_pnl_pct
------------------------------------------------------
reprice_pnl_pct adds back an "old" friction fraction computed from a
CostModel and subtracts a "new" one — correct only when the historical
friction really was a strategy-level constant. scripts/cost_model_report.py
already excludes single-leg trades from its own reprice for exactly this
reason: their close path charges 30% of the LIVE quoted bid-ask width AT THE
MOMENT OF EXIT (paper_manager._get_spread_slippage), floored at $0.05/share
and capped at $0.50 — a per-trade, per-exit-time observation no CostModel can
reconstruct. An expired single-leg trade charged ZERO friction historically,
not one side (paper_manager.py's dte<=0 settlement branch never subtracts
anything).

Instead this recomputes the GROSS (pre-friction) return directly from
entry_price and exit_price — both stored on every closed row — using exactly
the formula that produced them (paper_manager._evaluate_short_single_leg_exit
/ _evaluate_long_single_leg_exit). That needs no cost-model assumption. Only
the NEW (surface-measured) friction is then subtracted, isolating the cost
assumption for real.

OPEN INTEREST: the ledger has no open_interest column. Where a two-sided
archived quote for the exact contract on its entry date exists
(data/chain_archive.db), its real open_interest prices the row on the
surface's exact cell. Where it doesn't, two figures are reported: a CENTRAL
estimate (SpreadSurface.oi_collapsed_relative, the OI-collapsed marginal) and
a CONSERVATIVE bound (SpreadSurface.relative(..., open_interest=None), which
resolves to the most illiquid bucket) — the true cost lies between the two.
"""
from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.execution_costs import round_trip_friction  # noqa: E402
from src.spread_surface import SpreadSurface  # noqa: E402

#: Bootstrap resamples, and the seed that makes the interval reproducible —
#: same convention as scripts/publish_track_record.py's _BOOTSTRAP_SEED.
BOOTSTRAP_N = 4_000
BOOTSTRAP_SEED = 20260902

#: Fallback relative half-spread (as a fraction of mid) when the surface has
#: no cell and no collapse rung applies. 0.05/mid at a typical few-dollar
#: single-leg premium lands near the pre-surface flat-dollar default; used
#: only so SpreadSurface.relative never has to raise for a genuinely empty
#: surface — the CLI already refuses outright before this is reachable in
#: practice (see main()).
_FALLBACK_HALF_SPREAD_DOLLARS = 0.05


def gross_pct(entry_price: float, exit_price: float, short: bool) -> float:
    """Pre-friction return, computed exactly as the code that produced the
    stored pnl_pct computed it (paper_manager.py:213 short, :389 long).
    Needs no cost-model assumption — entry_price and exit_price are both
    stored on every closed row."""
    if entry_price <= 0:
        return 0.0
    return ((entry_price - exit_price) / entry_price if short
            else (exit_price - entry_price) / entry_price)


def is_expired(exit_reason: Optional[str]) -> bool:
    """Whether a row settled at expiry (paper_manager.py's dte<=0 branch,
    reason='Expired (settled at intrinsic)') rather than being actively
    closed. Expiry pays the opening friction side only — an option that
    expires worthless (or is cash/share-settled at intrinsic) is never sold."""
    return bool(exit_reason) and exit_reason.startswith("Expired")


def new_friction_fraction(surface: SpreadSurface, mid: float, abs_delta: float,
                          dte: float, open_interest: Optional[float],
                          round_trip: bool, *, central: bool = False
                          ) -> Tuple[float, str]:
    """New round-trip friction as a fraction of `mid`, plus its lookup
    provenance ("cell" = a real measured cell; anything else is a fallback
    rung — see SpreadSurface.relative's own docstring).

    `central=True` (meaningful only when `open_interest` is None) uses
    `oi_collapsed_relative` — the genuine OI-unknown marginal — instead of
    `relative(..., open_interest=None)`, which resolves to the conservative
    illiquid-bucket-0 pin. The two are NOT interchangeable; using the wrong
    one and calling it "central" silently reports the conservative bound as
    if it were the best estimate."""
    default = _FALLBACK_HALF_SPREAD_DOLLARS / mid if mid > 0 else 0.0
    if open_interest is None and central:
        rel, prov = surface.oi_collapsed_relative(
            abs_delta=abs_delta, dte=dte, default=default)
    else:
        rel, prov = surface.relative(
            abs_delta=abs_delta, dte=dte, open_interest=open_interest,
            default=default)
    half_dollar = rel * mid
    friction_dollar = round_trip_friction(
        n_legs=1, half_spread=half_dollar, commission_per_contract=0.0,
        round_trip=round_trip)
    return (friction_dollar / mid if mid > 0 else 0.0), prov


def cluster_bootstrap_pf(rows: Sequence[dict], value_key: str,
                         cluster_key: str = "date",
                         n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED,
                         alpha: float = 0.05
                         ) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    """Point profit factor plus a percentile 95% CI, resampling whole entry
    days rather than rows.

    Multiple trades opened the same day are not independent draws — the same
    overcounting trap that inflated the pre-registered ranker test and the
    catalyst bootstrap (both counted rows where the unit was a cluster).
    src.prereg_ranker.cluster_bootstrap_ci uses the same resample-whole-
    clusters pattern for the rank IC statistic; this is that pattern applied
    to profit factor instead.

    Returns (point, lo, hi). `lo`/`hi` are None when fewer than 2 clusters
    exist, or when too many resamples had no losing trade for an interval to
    mean anything (profit_factor itself returns None there) — reported as
    "underpowered", never as a fabricated interval.
    """
    from scripts.publish_track_record import profit_factor

    values = [r[value_key] for r in rows]
    point = profit_factor(values)

    groups: Dict[str, List[float]] = {}
    for r in rows:
        groups.setdefault(r[cluster_key], []).append(r[value_key])
    keys = list(groups)
    if len(keys) < 2:
        return point, None, None

    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(n_boot):
        sample: List[float] = []
        for _ in keys:
            k = keys[rng.randrange(len(keys))]
            sample.extend(groups[k])
        pf = profit_factor(sample)
        if pf is not None:
            draws.append(pf)

    if len(draws) < n_boot // 2:
        return point, None, None
    draws.sort()
    lo = draws[int(alpha / 2 * len(draws))]
    hi = draws[int((1 - alpha / 2) * len(draws)) - 1]
    return point, lo, hi


_SINGLE_LEG_SQL = """
    SELECT tr.entry_id, tr.ticker, tr.strike, tr.expiration, tr.date,
           tr.type, tr.strategy_name, tr.entry_price, tr.exit_price,
           tr.exit_reason, tr.pnl_pct, tr.pnl_usd, tr.capital_at_risk,
           tr.quantity, tr.entry_delta,
           julianday(tr.expiration) - julianday(tr.date) AS dte
    FROM trades tr
    WHERE tr.status = 'CLOSED' AND tr.net_credit IS NULL
      AND tr.entry_price IS NOT NULL AND tr.entry_price > 0
      AND tr.exit_price IS NOT NULL
      AND tr.entry_delta IS NOT NULL
      AND tr.expiration IS NOT NULL{dup_filter}
"""

_OI_SQL = """
    SELECT open_interest FROM chain_snapshots
    WHERE symbol = ? AND strike = ? AND expiration = ?
      AND substr(type, 1, 1) = substr(?, 1, 1)
      AND snap_date = substr(?, 1, 10)
      AND bid > 0 AND ask > bid
    LIMIT 1
"""


def count_multi_leg_refused(ledger_db: str) -> int:
    """Closed multi-leg trades (net_credit IS NOT NULL — the structural
    test, not the strategy name), counted but never priced here: entry_price
    on a spread is a net credit, not a leg mid."""
    import sqlite3

    from src.ledger_filters import exclude_ruled_duplicates

    con = sqlite3.connect(ledger_db)
    try:
        dup_filter = exclude_ruled_duplicates(con)
        return con.execute(
            "SELECT COUNT(*) FROM trades WHERE status='CLOSED' "
            f"AND net_credit IS NOT NULL{dup_filter}"
        ).fetchone()[0]
    finally:
        con.close()


def fetch_single_leg_rows(ledger_db: str, archive_db: str) -> List[dict]:
    """Every closed single-leg row, with real open_interest joined in from
    the archive where a two-sided quote exists for that exact contract on
    its own entry date. `open_interest` is None (genuinely unknown, not
    zero) when no such quote exists."""
    import sqlite3

    from src.ledger_filters import exclude_ruled_duplicates

    con = sqlite3.connect(ledger_db)
    con.row_factory = sqlite3.Row
    try:
        dup_filter = exclude_ruled_duplicates(con)
        raw = con.execute(
            _SINGLE_LEG_SQL.format(dup_filter=dup_filter)).fetchall()
    finally:
        con.close()

    arch = sqlite3.connect(archive_db)
    try:
        out = []
        for r in raw:
            row = dict(r)
            oi = arch.execute(
                _OI_SQL, (row["ticker"], row["strike"], row["expiration"],
                         row["type"], row["date"])
            ).fetchone()
            row["open_interest"] = float(oi[0]) if oi and oi[0] is not None else None
            row["abs_delta"] = abs(float(row["entry_delta"]))
            out.append(row)
        return out
    finally:
        arch.close()


def dollar_scale_factor(entry_price: float, booked_pct: float,
                        booked_pnl_usd: Optional[float],
                        quantity: Optional[float]) -> float:
    """The multiplier*lots factor implicit in the ledger's own booked row:
    pnl_usd = entry_price * pnl_pct * factor (paper_manager._sanitize_close_
    values). Deriving it from the booked row rather than reimporting
    paper_manager's private _get_multiplier/_lots keeps this script decoupled
    from exit-path internals — and the basis audit already verified pnl_usd
    reconciles against pnl_pct * entry_price * quantity on the real ledger.

    Falls back to quantity * 100.0 (the standard equity-option multiplier)
    when booked_pct is ~0, where the division is undefined rather than merely
    small."""
    if abs(booked_pct) > 1e-9 and booked_pnl_usd is not None:
        return booked_pnl_usd / (entry_price * booked_pct)
    return float(quantity or 1.0) * 100.0


def reprice_row(row: dict, surface: SpreadSurface) -> dict:
    """Add gross/repriced return figures to one cohort row. Mutates a copy;
    the input row is unchanged."""
    from src.utils import is_short_position

    out = dict(row)
    short = is_short_position(row["strategy_name"] or "")
    out["short"] = short
    out["gross_pct"] = gross_pct(float(row["entry_price"]), float(row["exit_price"]), short)
    round_trip = not is_expired(row.get("exit_reason"))
    oi = row.get("open_interest")
    out["oi_known"] = oi is not None

    if oi is not None:
        frac, prov = new_friction_fraction(
            surface, mid=float(row["entry_price"]), abs_delta=row["abs_delta"],
            dte=float(row["dte"]), open_interest=oi, round_trip=round_trip)
        out["friction_fraction_central"] = frac
        out["friction_fraction_conservative"] = frac
        out["provenance_central"] = prov
        out["provenance_conservative"] = prov
    else:
        c_frac, c_prov = new_friction_fraction(
            surface, mid=float(row["entry_price"]), abs_delta=row["abs_delta"],
            dte=float(row["dte"]), open_interest=None, round_trip=round_trip,
            central=True)
        v_frac, v_prov = new_friction_fraction(
            surface, mid=float(row["entry_price"]), abs_delta=row["abs_delta"],
            dte=float(row["dte"]), open_interest=None, round_trip=round_trip,
            central=False)
        out["friction_fraction_central"] = c_frac
        out["friction_fraction_conservative"] = v_frac
        out["provenance_central"] = c_prov
        out["provenance_conservative"] = v_prov

    out["repriced_pct_central"] = out["gross_pct"] - out["friction_fraction_central"]
    out["repriced_pct_conservative"] = out["gross_pct"] - out["friction_fraction_conservative"]

    scale = dollar_scale_factor(float(row["entry_price"]), float(row["pnl_pct"]),
                                row.get("pnl_usd"), row.get("quantity"))
    out["dollar_scale"] = scale
    out["pnl_usd_repriced_central"] = (
        float(row["entry_price"]) * out["repriced_pct_central"] * scale)
    out["pnl_usd_repriced_conservative"] = (
        float(row["entry_price"]) * out["repriced_pct_conservative"] * scale)
    return out
