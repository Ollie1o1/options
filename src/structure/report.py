"""Terminal rendering. Every recommendation shows its own evidence."""
from typing import Dict, List

from src import ui

from .types import Expression, Rejection, View

WIDTH = 100


def _league_lines(table: Dict) -> List[str]:
    """The league table itself - so the user can see what earned its place."""
    rows = sorted(table.values(), key=lambda m: -m.margin)
    out = [ui.rule(WIDTH, "STRUCTURE LEAGUE TABLE (rolling 90d)")]
    out.append("    {:<14s} {:>5s} {:>9s} {:>8s} {:>9s}  {}".format(
        "structure", "n", "B/E", "real", "margin", "state"))
    for m in rows:
        mark = " ~" if m.ci_includes_zero else ""
        out.append("    {:<14s} {:>5d} {:>8.1f}% {:>7.1f}% {:>+8.1f}  {}{}"
                   .format(m.strategy, m.n, m.breakeven_hit * 100,
                           m.realized_hit * 100, m.margin * 100, m.state, mark))
    out.append("    ~ = 95% CI includes zero: present, not trusted")
    return out


def render(view: View, expressions: List[Expression],
           rejections: List[Rejection], table: Dict,
           capital_usd: float) -> str:
    lines = []
    lines.append(ui.heavy_rule(WIDTH, "STRUCTURE EXPRESSION"))
    lines.append("  capital ${:.0f} USD".format(capital_usd))
    lines.append("")
    lines.append("  VIEW: {}  {}  confidence {:.2f}".format(
        view.symbol, view.direction, view.confidence))
    for d in view.drivers:
        lines.append("    - {}".format(d))

    if view.direction == "NEUTRAL":
        lines.append("")
        lines.append("  no directional edge today - directional structures "
                     "suppressed")

    if not table:
        lines.append("")
        lines.append("  no structure evidence available "
                     "(paper_trades.db empty or unreadable)")
        lines.append(ui.rule(WIDTH))
        return "\n".join(lines)

    lines.append("")
    lines.extend(_league_lines(table))

    lines.append("")
    if expressions:
        lines.append(ui.rule(WIDTH, "EXPRESS AS"))
        for i, e in enumerate(expressions, 1):
            lines.append("  {}. {}   cost ${:.0f}  [fits]".format(
                i, e.strategy, e.capital_required))
            lines.append("       B/E {:.1f}%  |  realized {:.1f}%  |  "
                         "margin {:+.1f} pts".format(
                             e.breakeven_hit * 100, e.realized_hit * 100,
                             e.margin * 100))
            lines.append("       cost drag {:.1f}% of max profit".format(
                e.cost_drag_pct))
            if e.warning:
                lines.append("       ! {}".format(e.warning))
    else:
        # Distinguish "the filters rejected everything" from "no contracts were
        # supplied to filter" - conflating them would misreport a plumbing gap
        # as an edge verdict. Wrong-direction rejections do not count as the
        # filters having done real work: those structures were never in scope
        # for this view.
        judged = [r for r in rejections
                  if "no candidate contract" not in r.reason
                  and "wrong direction" not in r.reason]
        if rejections and not judged:
            lines.append("  no candidate contracts supplied - run a scan and "
                         "pass its picks in to see expressions.")
        else:
            lines.append("  nothing clears its own breakeven today.")

    if rejections:
        lines.append("")
        lines.append(ui.rule(WIDTH, "REJECTED"))
        for r in rejections:
            lines.append("    {:<14s} {}".format(r.strategy, r.reason))

    lines.append(ui.rule(WIDTH))
    lines.append("  display-only - real money OFF")
    return "\n".join(lines)
