"""Top-level dispatcher between equity options, crypto, leverage, and breakout.

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
    return row


def _show_menu() -> str:
    if HAS_UI:
        print()
        print(ui.banner("OPTIONS SCREENER  ·  QUANT DESK", width=WIDTH))
    else:
        print("\n" + "=" * WIDTH + "\n  OPTIONS SCREENER  ·  QUANT DESK\n" + "=" * WIDTH)
    print()
    print(_row("1", "STOCKS", "equity options — discover / spreads / iron / sell"))
    print(_row("2", "CRYPTO", "BTC/ETH options on Deribit + perp funding/basis"))
    print(_row("3", "LEVERAGE", "BTC/ETH perp futures strategy", tag="no edge yet"))
    print(_row("4", "BREAKOUT", "stock breakout/breakdown probabilities",
               tag="no edge vs vol"))
    print(_row("Q", "QUIT", "", muted_key=True))
    print()
    print(ui.rule(WIDTH) if HAS_UI else "-" * WIDTH)
    try:
        choice = input("  Choice [1]: ").strip() or "1"
    except (EOFError, KeyboardInterrupt):
        print()
        return "Q"
    return choice.upper()


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
            from src.options_screener import main as _stocks_main
            _stocks_main()
            return
        if choice in ("2", "CRYPTO", "C"):
            from src.crypto.screener import main as _crypto_main
            _crypto_main()
            return
        if choice in ("3", "LEVERAGE", "L"):
            from src.leverage.__main__ import menu as _leverage_menu
            _leverage_menu()
            return
        if choice in ("4", "BREAKOUT", "B"):
            from src.breakout.engine import menu as _breakout_menu
            _breakout_menu()
            return
        if choice in ("Q", "QUIT", "EXIT", ""):
            print("  Goodbye.")
            return
        print(f"  Unknown choice: {choice!r} — pick 1, 2, 3, 4, or Q")


if __name__ == "__main__":
    main()
