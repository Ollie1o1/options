"""Entry-timing verdict for DISCOVER candidates: BUY NOW (already close to
a real support/fallback level) or WAIT for a specific price. Reuses
zones.py's own distance/sigma widening rule against a candidate's own
suggested_ladder — not a new weighted score. See
docs/superpowers/specs/2026-07-22-discover-entry-verdict-design.md.
"""
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .discover import CandidateRead, DeepRead

BUY_NOW = "BUY NOW"
WAIT = "WAIT"

# Same widening rule as zones.assess: "near" means within one day's typical
# move, or 2%, whichever is wider.
_NEAR_WIDTH_FLOOR = 0.02

# Same threshold as board.py's _EARNINGS_WARN_DAYS — a BUY NOW inside this
# window gets a caution, never silently upgraded to a clean signal.
_EARNINGS_CAUTION_DAYS = 14


@dataclass
class Verdict:
    """state is BUY_NOW or WAIT. target is the trigger price for WAIT,
    always None for BUY_NOW. caution is None unless a BUY_NOW is inside the
    earnings-caution window (see apply_caution) — WAIT verdicts never carry
    a caution, since "wait" is already the instruction."""
    state: str
    target: Optional[float] = None
    reason: str = ""
    caution: Optional[str] = None


def verdict_for(candidate: "CandidateRead") -> Verdict:
    """Fast-tier-only verdict: BUY NOW if spot is already within one day's
    typical move (or 2%, whichever is wider) of the candidate's own
    suggested_ladder second tranche (its nearest real support below spot,
    or the -10% synthetic fallback when no real support exists); WAIT for
    that level otherwise.

    Precondition: candidate.suggested_ladder has at least 2 tranches —
    guaranteed by suggest_ladder()'s own contract (always exactly 3).
    """
    target_level = candidate.suggested_ladder[1].level
    spot = candidate.spot
    distance_frac = (spot - target_level) / target_level

    daily_vol_frac = (candidate.ann_vol_pct / 100.0 / math.sqrt(252)
                      if candidate.ann_vol_pct else 0.0)
    width = max(daily_vol_frac, _NEAR_WIDTH_FLOOR)

    has_real_support = bool(candidate.supports) and candidate.supports[0]["level"] < spot
    label = candidate.supports[0]["label"] if has_real_support else "-10% fallback level"

    if distance_frac <= width:
        return Verdict(state=BUY_NOW,
                       reason=f"within {distance_frac * 100:.1f}% of {label}")
    return Verdict(state=WAIT, target=target_level,
                   reason=f"{distance_frac * 100:.1f}% below {label}")


def apply_caution(verdict: Verdict, deep: Optional["DeepRead"]) -> Verdict:
    """Downgrades a BUY_NOW to carry an earnings-proximity caution when
    deep-tier data shows earnings inside _EARNINGS_CAUTION_DAYS. WAIT
    verdicts and missing/incomplete deep data pass through unchanged."""
    if verdict.state != BUY_NOW or deep is None or deep.earnings_days is None:
        return verdict
    if deep.earnings_days <= _EARNINGS_CAUTION_DAYS:
        return Verdict(state=verdict.state, target=verdict.target,
                       reason=verdict.reason,
                       caution=f"earnings in {deep.earnings_days} days")
    return verdict
