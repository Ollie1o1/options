#!/usr/bin/env python3
"""Zero-clutter macOS Menu-Bar Status Pill (SwiftBar / xbar compatible).

Symlink into your SwiftBar plugins folder (e.g. `options-desk.15m.py`) or run directly:
    python3 scripts/menubar_pill.py

Displays one unobtrusive pill in the top macOS menu bar (`◆ XLE/XLV/XLP VIX 27` or
`▲ 1 EXIT`), with a clean dropdown showing Regime, TFSA Core weights, Open Picks,
Data Pipeline LEDs, and 1-click Terminal launch into `opt`. Zero popups, zero clutter.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.notifications import format_status_pill  # noqa: E402
from src.tui.state import DeskDataProvider  # noqa: E402


def main() -> None:
    provider = DeskDataProvider()
    snap = provider.get_radar(refresh=False)
    book = provider.get_book()
    health = provider.get_health()

    pill = format_status_pill(snap, book, health)
    print(f"{pill} | size=12 font=Menlo")
    print("---")
    print(f"Regime: {snap.regime_posture} · SPY ${snap.spy_last or 0:.2f} · VIX {snap.vix or 0:.1f} ({snap.vix_term or '—'})")
    print(f"Next Macro: {snap.next_macro_event}")
    print("---")
    print("Core TFSA Sleeve (PROVEN)")
    for c in snap.core_allocation or []:
        tk = "SGOV/CASH" if str(c["ticker"]) == "CASH" else str(c["ticker"])
        print(f"-- {tk:<10} {float(c['target_weight']) * 100:5.1f}% | font=Menlo")
    exits = book.get("exit_alerts") or []
    if exits:
        print("---")
        print(f"EXIT ALERTS ({len(exits)}) | color=#e06c75")
        for al in exits[:6]:
            print(f"-- {al} | color=#d6a452")
    pipes = health.get("pipes") or {}
    if pipes:
        print("---")
        print("Data Pipelines")
        for name, info in pipes.items():
            st = str(info.get("status") or "OK")
            dot = "●"
            col = "#5ec98d" if st == "OK" else ("#d6a452" if st == "WARN" else "#e06c75")
            print(f"-- {dot} {name.upper():<10} {info.get('detail', '')} | color={col} font=Menlo")
    print("---")
    print(f"Open Options Desk (opt) | bash=/usr/bin/open param1=-a param2=Terminal param3={ROOT} terminal=false")
    print("Refresh Now | refresh=true")


if __name__ == "__main__":
    main()
