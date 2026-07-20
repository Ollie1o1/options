"""Terminal surfaces for the holdings desk: startup banner + board.

House style: everything through src/ui.py + fmt.style semantic names —
never raw Colors. Plain mode must stay readable."""
from typing import Dict, List, Optional

import src.formatting as fmt
from src import ui
from .fills import DEFAULT_DB
from .plan import Plan, PlanName, Tranche, tranche_size_usd
from .zones import FILLED, IN_ZONE, NEAR, WATCHING, ZoneRead

_STATE_STYLE = {IN_ZONE: "good", NEAR: "warn", WATCHING: "muted", FILLED: "label"}
_LEVEL_TOL = 1e-6


def suggested_size(plan: Plan, name: PlanName, tranche: Tranche,
                   remaining_usd: float) -> float:
    return min(tranche_size_usd(plan, name, tranche), max(0.0, remaining_usd))


def _name_for(plan: Plan, ticker: str) -> Optional[PlanName]:
    for n in plan.names:
        if n.ticker == ticker:
            return n
    return None


def _tranche_at(name: PlanName, level: float) -> Optional[Tranche]:
    for t in name.tranches:
        if abs(t.level - level) < _LEVEL_TOL:
            return t
    return None


def banner(reads: List[ZoneRead], plan: Plan, remaining_usd: float,
           earnings: Optional[Dict[str, str]] = None, width: int = 100) -> str:
    triggered = [r for r in reads if r.state in (IN_ZONE, NEAR)]
    if not triggered:
        return ""
    earnings = earnings or {}
    lines = [ui.rule(width, "LONG-TERM BUY ZONES")]
    for r in triggered:
        name = _name_for(plan, r.ticker)
        tranche = _tranche_at(name, r.next_level) if name and r.next_level is not None else None
        size = suggested_size(plan, name, tranche, remaining_usd) if tranche else 0.0
        segs = [
            fmt.style(r.ticker, "emph"),
            fmt.style(f"{r.spot:,.2f}", "value"),
            fmt.style(r.state, _STATE_STYLE[r.state]),
            fmt.style(f"tranche @ {r.next_level:g} ({r.distance_pct:+.1f}%)", "value"),
            fmt.style(f"drawdown {r.drawdown_pct:+.1f}% ATH", "label"),
            fmt.style(f"tranche ${size:,.0f}", "accent"),
        ]
        if r.ticker in earnings:
            segs.append(fmt.style(f"{fmt.GLYPHS['warn']} earnings {earnings[r.ticker]}", "warn"))
        lines.append("  " + f"  {fmt.style(fmt.GLYPHS['dot'], 'muted')}  ".join(segs))
    lines.append("  " + fmt.style(
        f"cash pool: ${remaining_usd:,.0f} remaining of ${plan.cash_pool_usd:,.0f}", "muted"))
    return "\n".join(lines)


def render_board(plan: Plan, reads: List[ZoneRead], book: Dict[str, dict],
                 remaining_usd: float, earnings: Optional[Dict[str, str]] = None,
                 width: int = 100) -> str:
    earnings = earnings or {}
    lines = [ui.rule(width, "HOLDINGS — LONG-TERM ACCUMULATION")]
    if not plan.names:
        lines.append("  " + fmt.style(
            "plan is empty — ADD MU 750/650/550 to start a ladder", "label"))
        return "\n".join(lines)
    by_ticker = {r.ticker: r for r in reads}
    for name in plan.names:
        r = by_ticker.get(name.ticker)
        head = [fmt.style(name.ticker, "heading")]
        if r:
            head.append(fmt.style(f"{r.spot:,.2f}", "emph"))
            head.append(fmt.style(r.state, _STATE_STYLE[r.state]))
            head.append(fmt.style(f"{r.drawdown_pct:+.1f}% ATH", "label"))
            if r.sigma_dist is not None:
                head.append(fmt.style(f"{r.sigma_dist:+.1f}σ to zone", "label"))
            if r.above_ma200 is not None:
                head.append(fmt.style(
                    "above 200dma" if r.above_ma200 else "below 200dma", "muted"))
        else:
            head.append(fmt.style("no data", "bad"))
        if name.ticker in earnings:
            head.append(fmt.style(
                f"{fmt.GLYPHS['warn']} earnings {earnings[name.ticker]}", "warn"))
        lines.append("")
        lines.append("  " + f"  {fmt.style(fmt.GLYPHS['dot'], 'muted')}  ".join(head))
        if name.thesis:
            lines.append("    " + fmt.style(name.thesis, "muted"))
        filled_here = set()
        if r and r.next_level is not None:
            filled_here = {t.level for t in name.tranches if t.level > r.next_level}
        elif r and r.state == FILLED:
            filled_here = {t.level for t in name.tranches}
        for i, t in enumerate(name.tranches):
            joint = "└─" if i == len(name.tranches) - 1 else "├─"
            size = suggested_size(plan, name, t, remaining_usd)
            if t.level in filled_here:
                mark = fmt.style(f"{fmt.GLYPHS['check']} filled", "good")
            elif r and r.next_level is not None and abs(t.level - r.next_level) < _LEVEL_TOL:
                mark = fmt.style(f"{r.state}  ({r.distance_pct:+.1f}%)",
                                 _STATE_STYLE[r.state])
            else:
                mark = fmt.style("open", "muted")
            lines.append(f"    {joint} " + fmt.style(f"{t.level:g}", "value")
                         + "  " + fmt.style(f"${size:,.0f}", "label") + "  " + mark)
        held = book.get(name.ticker)
        if held and held["shares"]:
            pnl = (r.spot - held["avg_price"]) * held["shares"] if r else None
            seg = (f"held {held['shares']:,.2f} sh @ avg {held['avg_price']:,.2f}"
                   + (f"  P&L {pnl:+,.0f}" if pnl is not None else ""))
            lines.append("    " + fmt.style(seg, "good" if (pnl or 0) >= 0 else "bad"))
    lines.append("")
    lines.append("  " + fmt.style(
        f"cash pool ${plan.cash_pool_usd:,.0f}  {fmt.GLYPHS['dot']}  "
        f"remaining ${remaining_usd:,.0f}", "label"))
    return "\n".join(lines)
