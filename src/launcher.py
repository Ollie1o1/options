"""Top-level dispatcher between equity options, crypto, leverage, and the
read-only research tools (breakout / vol-intelligence / equity-VRP).

Behavior:
  - With no flags: show the interactive quant-desk menu.
  - With ANY argv flags: forward to src.options_screener.main() unchanged
    (preserves cron, --enforce-exits, --default-scoring shortcuts, every
    historical CLI behavior). Other modes are opt-in via the menu only.

Equity callers (cron, run.py with shortcut flags) hit the second path and
never see the menu — zero behavior change for the existing screener.

The menu renders through src/ui.py + fmt.style (the quant-desk theme), the
same components the rest of the screener uses — never raw Colors.
"""
from __future__ import annotations

import sys

try:
    from src import ui
    from src import formatting as fmt
    HAS_UI = True
except ImportError:  # extremely defensive — ui is core
    ui = None  # type: ignore
    fmt = None  # type: ignore
    HAS_UI = False

WIDTH = 74


def _row(key: str, name: str, desc: str, tag: str = "", muted_key: bool = False) -> str:
    """One aligned menu row: [key]  NAME       description            [tag]."""
    if not HAS_UI:
        pad_name = name.ljust(9)
        return f"  [{key}]  {pad_name}  {desc}" + (f"   [{tag}]" if tag else "")
    key_style = "muted" if muted_key else "accent"
    k = fmt.style(f"[{key}]", key_style, bold=not muted_key)
    n = fmt.style(ui.pad(name, 9), "heading")
    d = fmt.style(desc, "muted")
    row = f"  {k}  {n}  {d}"
    if tag:
        row += "  " + fmt.style(f"[{tag}]", "warn")
    return row.rstrip()


def _loading(msg: str) -> None:
    """Immediate feedback before a heavy lazy import / scan, so the screen never
    sits frozen between the menu choice and the first output of the sub-tool.
    Flushed so it shows before the multi-second import blocks."""
    line = f"  {msg}"
    print(fmt.style(line, "muted") if HAS_UI else line, flush=True)


def _art_width() -> int:
    import os
    try:
        return min(os.get_terminal_size().columns, 100)
    except OSError:
        return WIDTH


def _show_menu() -> str:
    # ── Wordmark masthead: the animated art lives HERE, at the very top of
    # the launcher, and nowhere else. The painter below re-styles exactly
    # these rows, so everything printed after the art is counted into
    # `offset` (rows between the art's bottom line and the prompt row).
    art_rows, art_w = 0, _art_width()
    interactive = False
    try:
        interactive = sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        pass
    if HAS_UI and interactive:
        from src import ui_motion
        print()
        for line in ui_motion.art_lines(art_w):
            print(fmt.style(line, "muted"))
            art_rows += 1

    after = []
    if HAS_UI:
        after.append("")
        after.extend(ui.banner("OPTIONS SCREENER  ·  QUANT DESK",
                               width=WIDTH).splitlines())
    else:
        after.extend(("", "=" * WIDTH, "  OPTIONS SCREENER  ·  QUANT DESK",
                      "=" * WIDTH))
    after.append("")
    after.append(_row("1", "STOCKS", "equity options — discover / spreads / iron / sell"))
    after.append(_row("2", "CRYPTO", "BTC/ETH options on Deribit + perp funding/basis"))
    after.append(_row("3", "LEVERAGE", "BTC/ETH perp futures strategy", tag="no edge yet"))
    after.append(_row("4", "RESEARCH", "breakout · vol-intelligence · equity-VRP",
                      tag="read-only"))
    after.append(_row("5", "HOLDINGS", "long-term stock accumulation — buy zones · tranches · TFSA book"))
    after.append(_row("Q", "QUIT", "", muted_key=True))
    after.append("")
    after.append(ui.rule(WIDTH) if HAS_UI else "-" * WIDTH)
    for line in after:
        print(line)

    motion = None
    if art_rows:
        from src import ui_motion
        if ui_motion.motion_allowed(True):
            motion = ui_motion.HeaderMotion(
                art_rows, lambda _w: ui_motion.art_frame(art_w),
                offset=len(after))
            motion.start()
    try:
        choice = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print()
        return "Q"
    finally:
        if motion is not None:
            motion.stop()
    return choice.upper()


def _research_menu() -> None:
    """Read-only research/analytics tools (no scores, no trades, no real money)."""
    while True:
        if HAS_UI:
            print()
            print(ui.rule(WIDTH, "RESEARCH / ANALYTICS  (read-only)"))
        else:
            print("\n-- RESEARCH / ANALYTICS (read-only) --")
        print(_row("1", "BREAKOUT", "breakout/breakdown probabilities", tag="no edge vs vol"))
        print(_row("2", "VOL-INTEL", "IV movers + implied-vs-realized (VRP)", tag="monitor"))
        print(_row("3", "EQUITY-VRP", "delta-hedged short-straddle backtest",
                   tag="no single-name edge"))
        print(_row("B", "BACK", "", muted_key=True))
        print(ui.rule(WIDTH) if HAS_UI else "-" * WIDTH)
        try:
            choice = (input("  Choice [1]: ").strip() or "1").upper()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if choice in ("B", "BACK", "Q", "QUIT", ""):
            return
        if choice in ("1", "BREAKOUT"):
            _loading("Loading breakout engine…")
            from src.breakout.engine import menu as _breakout_menu
            _breakout_menu()
        elif choice in ("2", "VOL-INTEL", "VOL"):
            _loading("Loading vol-intelligence…")
            from src.vol_intel.engine import main as _vol_main
            _vol_main([])
        elif choice in ("3", "EQUITY-VRP", "VRP"):
            _loading("Running equity-VRP backtest over dolt chains…")
            from src.equity_vol.report import main as _evrp_main
            _evrp_main([])
        else:
            print(f"  Unknown choice: {choice!r} — pick 1, 2, 3, or B")


def main() -> None:
    # If any CLI arg was provided, skip the menu and dispatch straight to the
    # equity screener. This keeps cron, --default-scoring shortcuts, and every
    # power-user flag working exactly as before.
    if len(sys.argv) > 1:
        from src.options_screener import main as _stocks_main
        _stocks_main()
        return

    while True:
        choice = _show_menu()
        if choice in ("1", "STOCKS", "S"):
            _loading("Loading equity options screener…")
            from src.options_screener import main as _stocks_main
            _stocks_main()
            continue  # back to the top menu, not exit
        if choice in ("2", "CRYPTO", "C"):
            _loading("Loading crypto screener…")
            from src.crypto.screener import main as _crypto_main
            _crypto_main()
            continue
        if choice in ("3", "LEVERAGE", "L"):
            _loading("Loading leverage menu…")
            from src.leverage.__main__ import menu as _leverage_menu
            _leverage_menu()
            continue
        if choice in ("4", "RESEARCH", "R"):
            _research_menu()
            continue
        if choice in ("5", "HOLDINGS", "H"):
            _loading("Loading holdings desk…")
            from src.longterm.board import menu as _holdings_menu
            _holdings_menu()
            continue
        if choice in ("Q", "QUIT", "EXIT", ""):
            print("  Goodbye.")
            return
        print(f"  Unknown choice: {choice!r} — pick 1, 2, 3, 4, 5, or Q")


if __name__ == "__main__":
    main()
