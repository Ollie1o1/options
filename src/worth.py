"""How far a candidate clears its own bar — graded, and honest about what it is.

The boards refuse rather than rank (`src/pick_ranking.py`): no ordering of the
survivors beat `quality_score` at the top slot out of sample, so position on a
board carries no claim. That left a real gap. A card said `POSITIVE EV +88/ct`
and the table said `TAKE`, but neither said *by how much* — and "is this worth
buying, a little or a lot" is the question a reader actually has.

This grades that, on three margins, and it grades them the way the gates do:

    the grade is the WEAKEST of the three, never a blend.

Blending plausible-sounding components into one number is exactly what produced
a composite whose top quintile lost $10,173 across the book. A contract whose
edge clears its error bar 2.4x but hands 30% of its reward to the spread is not
STRONG; it is whatever its worst margin says it is.

WHAT THIS IS NOT
----------------
Not a profit forecast, and not a claim that a STRONG contract will beat a THIN
one. Net EV is Black-Scholes over a trailing vol estimate whose out-of-sample
information content in this repo measures ~0. What the primary axis reports is
how confident the model is in its own estimate: `net_ev / noise`, where `noise`
is what a one-sigma error in this contract's implied vol is worth to it. An
edge of +$9 on a contract whose vega moves $40 per IV point is a rounding error
with a sign, and that is worth saying out loud on the card.

It cannot yet be validated. `ev_per_contract` and its noise band were computed
at scan time and discarded — the ledger kept only `ev_score`, a within-scan
rank. Schema 21 persists all four, so from 2026-08-10 the question "did STRONG
beat THIN" accumulates an answer instead of an argument.

The two secondary margins are arithmetic and need no validation: friction is
measured off the quotes, and p* = 1 - credit/width is the win rate a credit
structure must beat to break even.

A NOTE ON THE BREAKEVEN MARGIN
------------------------------
p* against your own history looked like a strong per-contract signal on
2026-08-10: Spearman +0.246 (p=0.00001, n=302) against return on capital,
computed strictly causally, positive in all three time thirds. Demeaned WITHIN
structure it is +0.104 (p=0.072). It is a structure-selection signal — Bull Put
margin +0.106 and +22% return against Iron Condor -0.137 and -8.1% — not a
contract-selection one. So it is shown as context and can demote a grade on
arithmetic grounds, but it is never presented as predicting which spread wins.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Optional

# Edge measured in units of its own error bar. The 1.0 boundary is not chosen
# here — it is where `decide_verdict` already splits TAKE from MARGINAL, and
# this scale refines that split rather than inventing a second one.
STRONG_SIGMA = 2.0
CLEAR_SIGMA = 1.0

# Round-trip crossing cost as a share of reward. Measured 2026-08-04: 0.7-1.7%
# for a single leg, ~33% per crossing for a two-leg credit spread. A candidate
# handing over a tenth of its reward to trade cannot be called strong however
# large its modelled edge.
STRONG_MAX_FRICTION = 0.05
CLEAR_MAX_FRICTION = 0.10

GRADES = ("THIN", "CLEAR", "STRONG")
_PIPS = {"STRONG": "●●●○", "CLEAR": "●●○○", "THIN": "●○○○", "UNGRADED": "○○○○"}


@dataclass(frozen=True)
class Worth:
    """A grade and the three margins behind it."""
    grade: str                                  # STRONG / CLEAR / THIN / UNGRADED
    sigma: Optional[float] = None               # net EV in error bars
    friction: Optional[float] = None            # round trip / reward
    breakeven_margin: Optional[float] = None    # your win rate - p*
    limiting: str = ""                          # which margin set the grade

    @property
    def pips(self) -> str:
        return _PIPS.get(self.grade, _PIPS["UNGRADED"])

    def line(self) -> str:
        """One line for a card. Says the grade, then what drove it."""
        if self.grade == "UNGRADED":
            return f"{self.pips} UNGRADED  no trustworthy vol basis for an edge estimate"
        parts = []
        if self.sigma is not None:
            parts.append(f"edge {self.sigma:.1f}x its error bar")
        if self.friction is not None:
            parts.append(f"costs {self.friction:.0%} of reward")
        if self.breakeven_margin is not None:
            parts.append(f"breakeven margin {self.breakeven_margin:+.0%}")
        detail = " · ".join(parts)
        limit = f"  (limited by {self.limiting})" if self.limiting else ""
        return f"{self.pips} {self.grade:<6}  {detail}{limit}"


def _finite(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _grade_sigma(sigma: Optional[float]) -> Optional[str]:
    if sigma is None:
        return None
    if sigma >= STRONG_SIGMA:
        return "STRONG"
    if sigma >= CLEAR_SIGMA:
        return "CLEAR"
    return "THIN"


def _grade_friction(friction: Optional[float]) -> Optional[str]:
    if friction is None:
        return None
    if friction <= STRONG_MAX_FRICTION:
        return "STRONG"
    if friction <= CLEAR_MAX_FRICTION:
        return "CLEAR"
    return "THIN"


def _grade_breakeven(margin: Optional[float]) -> Optional[str]:
    """Arithmetic only: a structure that must beat a win rate you do not run is
    not a strong proposition, whatever its modelled edge says."""
    if margin is None:
        return None
    if margin >= 0.10:
        return "STRONG"
    if margin >= 0.0:
        return "CLEAR"
    return "THIN"


def assess(row: Dict[str, Any],
           *,
           historical_win_rate: Optional[float] = None,
           required_win_rate: Optional[float] = None) -> Worth:
    """Grade one candidate. Never raises — an ungradeable row says so.

    `row` is a scan row or a ledger row; it needs `ev_per_contract` and enough
    to compute a noise band. Friction and breakeven come from
    `candidate_verdict`, which prices the candidate at what it would fill for.
    """
    try:
        # Before anything is graded: were these bands ever measured at this
        # tenor? Past 250 DTE the median candidate's friction alone exceeds
        # CLEAR_MAX_FRICTION, so STRONG and CLEAR become unreachable alone and
        # every long-dated candidate reads THIN regardless of quality. THIN is
        # a verdict; the honest answer out there is that there is no verdict.
        from .cost_calibration import (OUT_OF_RANGE_REASON, entry_dte,
                                       in_calibration)
        if not in_calibration(entry_dte(row)):
            return Worth("UNGRADED", limiting=OUT_OF_RANGE_REASON)

        net = _finite(row.get("ev_per_contract"))
        noise = _finite(row.get("ev_noise"))
        if noise is None:
            from .tearsheet.collect import ev_noise as _noise_of
            noise = _finite(_noise_of(row))

        sigma = None
        if net is not None and noise is not None and noise > 0:
            sigma = net / noise
        elif net is not None and noise == 0:
            # No vega and no cost to scale by. The edge is real arithmetic but
            # its uncertainty is unknown, so it cannot carry a sigma grade.
            sigma = None

        friction = beh = None
        try:
            from . import candidate_verdict as cv
            v = cv.verdict_for(row, historical_win_rate=historical_win_rate)
            friction = v.round_trip_pct
            # The MANAGED required rate, not `1 - credit/width`. The latter is
            # the hold-to-expiry figure, ruled against 2026-08-13, and it is
            # only defined for rows carrying a spread_width — which is why
            # every single-leg board rendered `n/a` here. Falls back to the
            # per-contract number when no managed rate exists for the
            # strategy (fewer than 20 closed trades, or no losers to measure).
            req = required_win_rate
            if req is None:
                try:
                    req = cv.required_win_rates_from_ledger().get(
                        row.get("strategy_name"))
                except Exception:
                    req = None
            if req is None:
                req = v.breakeven
            if req is not None and historical_win_rate is not None:
                beh = historical_win_rate - req
        except Exception:
            pass

        candidates = {
            "edge vs its error bar": _grade_sigma(sigma),
            "trading cost": _grade_friction(friction),
            "breakeven margin": _grade_breakeven(beh),
        }
        live = {k: g for k, g in candidates.items() if g is not None}
        if not live:
            return Worth("UNGRADED", sigma, friction, beh)

        worst = min(live.values(), key=GRADES.index)
        # "Limited by" is only true when one margin is genuinely holding the
        # grade down. When every margin agrees, naming one of them invents a
        # constraint that is not there — a STRONG contract read as "STRONG,
        # limited by edge vs its error bar", which is a contradiction.
        held_back = [k for k, g in live.items() if g == worst]
        limiting = held_back[0] if len(held_back) < len(live) else ""
        return Worth(worst, sigma, friction, beh, limiting)
    except Exception:
        return Worth("UNGRADED")
