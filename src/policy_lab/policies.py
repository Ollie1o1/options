"""The preregistered knob grid.

Expanding this after seeing results invalidates the run: `grid_cardinality` is
fed straight to the deflated-Sharpe correction as `n_trials`, so a grid that
grows quietly makes every past verdict too generous.

Short and long premium sweep different knobs — the live -50% stop is a
long-premium rule (n=117, -$67,488) while short premium stops at 2.0x credit —
so they are swept and reported separately and never pooled.

`NOSTOP_GRID` is preregistered in
`docs/PREREG_POLICY_LAB_NOSTOP_20260928.md`. It is a frozen, 6-cell
confirmatory grid testing whether removing the stop from the Corpus A
recorder baseline (`ExitPolicy(0.50, 2.0, None, None)`, not itself a grid
member) produces a real edge on held-out Bear Call data. Expanding it after
seeing results invalidates the run and requires a new preregistration — its
cardinality (6, via `grid_cardinality`) is exactly what feeds the
family-wise correction, so a grid that grows quietly makes that correction
too generous, the same trap the module docstring above warns about for the
750-cell grids.
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple


@dataclass(frozen=True)
class ExitPolicy:
    """A rule for leaving a position. `None` means the knob is not armed.

    `take_profit_frac` and `stop_mult` are both fractions/multiples of the
    OPENING price of the structure, so they read the same way for a credit
    spread and a long option.
    """
    name: str
    take_profit_frac: Optional[float]
    stop_mult: Optional[float]
    time_exit_dte: Optional[int]
    max_hold_days: Optional[int]


NULL_POLICIES: Tuple[ExitPolicy, ...] = (
    ExitPolicy("null_hold_to_expiry", None, None, None, None),
    ExitPolicy("null_never_stop", 0.50, None, 21, None),
    ExitPolicy("null_fixed_7d", None, None, None, 7),
)


def _name(prefix: str, tp, sl, dte, hold) -> str:
    def part(label, v):
        return f"{label}{'none' if v is None else v}"
    return "_".join([prefix, part("tp", tp), part("sl", sl),
                     part("dte", dte), part("hold", hold)])


def _build(prefix: str, tps: Sequence, sls: Sequence, dtes: Sequence,
           holds: Sequence) -> Tuple[ExitPolicy, ...]:
    return tuple(
        ExitPolicy(_name(prefix, tp, sl, dte, hold), tp, sl, dte, hold)
        for tp, sl, dte, hold in itertools.product(tps, sls, dtes, holds)
    )


# Short premium: take profit as a fraction of credit, stop as a multiple of it.
SHORT_PREMIUM_GRID: Tuple[ExitPolicy, ...] = _build(
    "sp",
    tps=(0.25, 0.35, 0.50, 0.65, 0.75, None),
    sls=(1.0, 1.5, 2.0, 3.0, None),
    dtes=(7, 14, 21, 28, None),
    holds=(3, 7, 14, 30, None),
)

# Long premium: the live rule is a -50% debit stop, so the stop grid is the
# fraction of the debit lost rather than a multiple of a credit.
LONG_PREMIUM_GRID: Tuple[ExitPolicy, ...] = _build(
    "lp",
    tps=(0.50, 1.00, 2.00, None),
    sls=(0.25, 0.50, 0.75, None),
    dtes=(7, 14, 21, None),
    holds=(3, 7, 14, 30, None),
)

# Frozen 6-cell confirmatory grid — see the module docstring and
# docs/PREREG_POLICY_LAB_NOSTOP_20260928.md. All six cells hold
# `time_exit_dte=None` and `max_hold_days=None`: the 750-cell run showed DTE
# 21 and 28 are degenerate on this corpus (median entry DTE 17), and leaving
# both knobs unarmed keeps the grid honest and small. The baseline,
# `ExitPolicy(0.50, 2.0, None, None)` (== `CORPUS_A_RECORDER_SHORT` in
# report.py), is deliberately NOT a member: it is the comparison point, not
# a candidate.
NOSTOP_GRID: Tuple[ExitPolicy, ...] = (
    ExitPolicy("nostop_tp050_slnone", 0.50, None, None, None),
    ExitPolicy("nostop_tp050_sl3.0", 0.50, 3.0, None, None),
    ExitPolicy("nostop_tp050_sl1.5", 0.50, 1.5, None, None),
    ExitPolicy("nostop_tp065_slnone", 0.65, None, None, None),
    ExitPolicy("nostop_tp065_sl2.0", 0.65, 2.0, None, None),
    ExitPolicy("nostop_tp035_slnone", 0.35, None, None, None),
)

LIVE_BASELINE_SHORT = ExitPolicy(
    _name("sp", 0.50, 2.0, 21, None), 0.50, 2.0, 21, None)
LIVE_BASELINE_LONG = ExitPolicy(
    _name("lp", None, 0.50, 21, None), None, 0.50, 21, None)


def grid_cardinality(grid: Sequence[ExitPolicy]) -> int:
    """Number of trials in a grid — the `n_trials` a DSR must be deflated by."""
    return len(grid)
