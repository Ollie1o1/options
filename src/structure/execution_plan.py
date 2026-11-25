"""Walk-Away Limit-Order Ladder & Execution Plan builder (`src/structure/execution_plan.py`).

Computes exact limit prices so the trader never chases a negative-EV or
high-friction fill across Wealthsimple or IBKR.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class WalkAwayLadder:
    structure: str
    is_credit: bool
    p_mid: float
    p_step25: float
    p_walk: float
    p_cross: float
    half_spread: float
    step1_label: str
    step2_label: str
    abort_label: str
    never_chase_warning: bool


def build_walk_away_ladder(
    *,
    structure: str,
    is_credit: bool,
    p_mid: float,
    p_cross: float,
    ev_view_gross_per_share: float,
    exit_half_spread_per_share: float = 0.0,
    max_friction_to_credit: float = 0.25,
) -> WalkAwayLadder:
    """Compute exact 3-step limit ladder and HARD ABORT walk-away price.

    - For **debit** structures (`p_cross >= p_mid`):
      `p_walk = min(p_cross, max(p_mid * 0.5, ev_view_gross_per_share - exit_half_spread_per_share))`
      Wait: if `ev_view_gross_per_share <= p_mid + exit_half_spread_per_share`, even mid is negative EV
      so `p_walk = ev_view_gross_per_share - exit_half_spread_per_share` (and `never_chase_warning = True`).
    - For **credit** structures (`p_cross <= p_mid`):
      Require round-trip concession <= `max_friction_to_credit` of credit received:
      `2 * (p_mid - p_walk) <= f * p_walk` => `p_walk >= p_mid / (1 + f/2)` (e.g., `p_mid / 1.125` for 25%).
      Also bounded below by `p_cross`.
    """
    half = abs(float(p_cross) - float(p_mid))
    if not is_credit:
        # Debit: buyer wants lower price; stepping toward cross increases limit
        p_25 = float(p_mid) + 0.25 * half
        breakeven_debit = float(ev_view_gross_per_share) - float(exit_half_spread_per_share)
        p_walk = min(float(p_cross), max(0.01, breakeven_debit))
        warn = p_walk < float(p_mid)
        step2_px = min(p_25, p_walk) if not warn else p_walk
        return WalkAwayLadder(
            structure=structure,
            is_credit=False,
            p_mid=round(float(p_mid), 2),
            p_step25=round(step2_px, 2),
            p_walk=round(p_walk, 2),
            p_cross=round(float(p_cross), 2),
            half_spread=round(half, 3),
            step1_label=f"Step 1 (0-3m): Post BUY LIMIT @ ${p_mid:.2f} (mid, 0% conceded)",
            step2_label=(
                f"Step 2 (3-6m): Step to ${step2_px:.2f} (<=25% half-spread)"
                if not warn
                else f"Step 2: DO NOT STEP UP — breakeven limit ${p_walk:.2f} is below mid"
            ),
            abort_label=f"HARD ABORT: Cancel if not filled by ${p_walk:.2f} (EV=0 walk-away)",
            never_chase_warning=warn,
        )

    # Credit: seller wants higher credit; stepping toward cross decreases limit
    p_25 = max(float(p_cross), float(p_mid) - 0.25 * half)
    friction_floor = float(p_mid) / (1.0 + 0.5 * float(max_friction_to_credit))
    p_walk = max(float(p_cross), friction_floor)
    warn = p_walk > float(p_mid) or p_cross <= 0
    step2_px = max(p_25, p_walk)
    return WalkAwayLadder(
        structure=structure,
        is_credit=True,
        p_mid=round(float(p_mid), 2),
        p_step25=round(step2_px, 2),
        p_walk=round(p_walk, 2),
        p_cross=round(float(p_cross), 2),
        half_spread=round(half, 3),
        step1_label=f"Step 1 (0-3m): Post SELL LIMIT @ ${p_mid:.2f} credit (mid, 0% conceded)",
        step2_label=f"Step 2 (3-6m): Step to ${step2_px:.2f} credit (<=25% half-spread)",
        abort_label=f"HARD ABORT: Cancel if not filled at/above ${p_walk:.2f} (<=25% friction ceiling)",
        never_chase_warning=warn,
    )


def format_ladder_lines(ladder: WalkAwayLadder) -> Tuple[str, str, str]:
    return (ladder.step1_label, ladder.step2_label, ladder.abort_label)
