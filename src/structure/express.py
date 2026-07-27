"""Route a view into the structures whose measured breakeven it can clear.

Five filters, each recording WHY it rejected something - the user learns the
logic instead of obeying a number.
"""
import json
from typing import Dict, List, Optional, Tuple

from .types import (BEARISH_STRUCTURES, BULLISH_STRUCTURES, CREDIT_STRUCTURES,
                    DEBIT_STRUCTURES, LEG_COUNT, NEUTRAL_STRUCTURES,
                    Expression, Rejection, View)
from .view import implied_hit

DEFAULT_COMMISSION = 0.65
DEFAULT_SLIPPAGE = 0.05


def load_costs(config_path: str = "config.json") -> Tuple[float, float]:
    """Read the ONE source of truth for execution costs."""
    try:
        with open(config_path) as f:
            pt = (json.load(f).get("paper_trading") or {})
    except (OSError, ValueError):
        return DEFAULT_COMMISSION, DEFAULT_SLIPPAGE
    return (float(pt.get("commission_per_contract", DEFAULT_COMMISSION)),
            float(pt.get("slippage_per_share", DEFAULT_SLIPPAGE)))


def round_trip_cost(legs: int, commission: float = DEFAULT_COMMISSION,
                    slippage: float = DEFAULT_SLIPPAGE) -> float:
    """Open + close, both legs, slippage per share x100 shares per contract."""
    return (slippage * 100.0 * legs * 2.0) + (commission * legs * 2.0)


def _allowed_for(direction: str):
    if direction == "BULLISH":
        return set(BULLISH_STRUCTURES)
    if direction == "BEARISH":
        return set(BEARISH_STRUCTURES)
    return set(NEUTRAL_STRUCTURES)


def express(view: View, table: Dict, capital_usd: float,
            candidates: Dict[str, dict], max_cost_drag_pct: float = 25.0,
            commission: Optional[float] = None,
            slippage: Optional[float] = None
            ) -> Tuple[List[Expression], List[Rejection]]:
    commission = DEFAULT_COMMISSION if commission is None else commission
    slippage = DEFAULT_SLIPPAGE if slippage is None else slippage

    out: List[Expression] = []
    rejections: List[Rejection] = []
    allowed = _allowed_for(view.direction)
    view_hit = implied_hit(view)

    for name, margin in sorted(table.items()):
        # 1. direction
        if name not in allowed:
            rejections.append(Rejection(name, "wrong direction for a {} view"
                                        .format(view.direction)))
            continue
        # 2. bench / evidence
        if margin.state == "BENCHED":
            rejections.append(Rejection(
                name, "BENCHED (margin {:+.1f} pts)".format(margin.margin * 100)))
            continue
        if margin.state == "UNPROVEN":
            rejections.append(Rejection(
                name, "UNPROVEN (n={}, {}W/{}L - not enough evidence)"
                .format(margin.n, margin.wins, margin.losses)))
            continue
        # 3. confidence gate - debit structures only.
        # A debit structure wins only if the underlying moves far enough, fast
        # enough, so the view must supply that accuracy directly. A credit
        # structure wins on direction OR theta OR vol contraction, so its
        # realized hit rate already prices in that redundancy - gating it on
        # directional confidence would double-count direction.
        if name in DEBIT_STRUCTURES and view_hit < margin.breakeven_hit:
            rejections.append(Rejection(
                name, "needs {:.1%} to break even; this view implies only "
                "{:.1%}".format(margin.breakeven_hit, view_hit)))
            continue
        if name in CREDIT_STRUCTURES and margin.margin <= 0:
            rejections.append(Rejection(name, "margin not positive"))
            continue

        cand = candidates.get(name)
        if not cand:
            rejections.append(Rejection(name, "no candidate contract found"))
            continue
        cap_req = float(cand.get("capital_required") or 0.0)
        max_profit = float(cand.get("max_profit") or 0.0)
        if cap_req <= 0:
            rejections.append(Rejection(name, "unknown capital requirement"))
            continue
        # 4. affordability
        if cap_req > capital_usd:
            rejections.append(Rejection(
                name, "needs ${:.0f}, you have ${:.0f}".format(
                    cap_req, capital_usd)))
            continue
        # 5. cost drag
        legs = LEG_COUNT.get(name, 2)
        cost = round_trip_cost(legs, commission, slippage)
        drag = 100.0 * cost / max_profit if max_profit > 0 else float("inf")
        if drag > max_cost_drag_pct:
            rejections.append(Rejection(
                name, "cost drag {:.0f}% of max profit (limit {:.0f}%)".format(
                    drag, max_cost_drag_pct)))
            continue

        out.append(Expression(
            strategy=name, margin=margin.margin,
            breakeven_hit=margin.breakeven_hit,
            realized_hit=margin.realized_hit, capital_required=cap_req,
            cost_drag_pct=drag, legs=legs))

    out.sort(key=lambda e: (-e.margin, e.cost_drag_pct))
    return out, rejections
