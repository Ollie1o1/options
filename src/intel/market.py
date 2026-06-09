"""Market overview sub-view — 'what's happening out there right now'.

Composes the existing regime + direction dashboards and adds a movers panel
(biggest 5-day moves across a liquid basket) and a compact playbook-style read.
Reuses regime_dashboard/levels rather than re-fetching primitives.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

from src.intel import ui

# Names scanned for the movers panel (indices + semis + mega-caps).
_MOVERS = (
    "SPY", "QQQ", "IWM", "SMH", "NVDA", "AMD", "AAPL", "MSFT",
    "AMZN", "META", "GOOGL", "TSLA", "XLE", "XLF", "XLK",
)


def gather_movers(symbols=_MOVERS) -> List[Tuple[str, float, str]]:
    """Return [(symbol, ret_5d, verdict)] sorted weakest-first."""
    from src.regime_dashboard import fetch_index_directions
    data = fetch_index_directions(symbols)
    rows = []
    for sym, info in data.items():
        r5 = info.get("ret_5d")
        if r5 is not None:
            rows.append((sym, r5, info.get("verdict", "")))
    rows.sort(key=lambda r: r[1])
    return rows


def render_movers(rows: List[Tuple[str, float, str]], width: int = 64) -> List[str]:
    if not rows:
        return ui.box("MARKET MOVERS", "5-day", ["  no data"], width)
    body: List[str] = []
    losers = rows[:4]
    winners = list(reversed(rows[-4:]))
    body.append(ui.color("  Weakest (5d)", ui._C.RED) if ui._HAS else "  Weakest (5d)")
    for sym, r5, _ in losers:
        body.append(f"    {sym:<5} {ui.color(f'{r5:+6.1%}', ui.direction_color(r5))}")
    body.append(ui.color("  Strongest (5d)", ui._C.GREEN) if ui._HAS else "  Strongest (5d)")
    for sym, r5, _ in winners:
        body.append(f"    {sym:<5} {ui.color(f'{r5:+6.1%}', ui.direction_color(r5))}")
    return ui.box("MARKET MOVERS", "5-day", body, width)


def print_market_overview(width: int = 90) -> None:
    """Print the full market overview: regime + direction + movers."""
    from src.regime_dashboard import print_regime_dashboard

    # Regime + direction (these already render their own clean boxes).
    print_regime_dashboard(width)

    rows = gather_movers()
    print()
    for line in render_movers(rows, width=min(width, 64)):
        print(line)
