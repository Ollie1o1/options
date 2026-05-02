"""Top-level dispatcher between equity options and crypto modes.

Behavior:
  - With no flags: show interactive menu [1] STOCKS [2] CRYPTO [Q] QUIT.
  - With ANY argv flags: forward to src.options_screener.main() unchanged
    (preserves cron, --enforce-exits, --default-scoring shortcuts, every
    historical CLI behavior). The crypto mode is opt-in via the menu only.

Equity callers (cron, run.py with shortcut flags) hit the second path and
never see the menu — zero behavior change for the existing screener.
"""
from __future__ import annotations

import sys

# Use the same formatter as both screeners so colors/banners match.
try:
    from src import formatting as fmt
    HAS_FMT = True
except ImportError:
    fmt = None  # type: ignore
    HAS_FMT = False


def _banner(text: str) -> None:
    bar = "═" * 80
    if HAS_FMT and fmt:
        print(fmt.colorize(bar, fmt.Colors.BRIGHT_CYAN))
        print(fmt.colorize(f"  {text}", fmt.Colors.BRIGHT_CYAN, bold=True))
        print(fmt.colorize(bar, fmt.Colors.BRIGHT_CYAN))
    else:
        print(bar)
        print(f"  {text}")
        print(bar)


def _color(text: str, color_attr: str, bold: bool = False) -> str:
    if HAS_FMT and fmt:
        return fmt.colorize(text, getattr(fmt.Colors, color_attr), bold=bold)
    return text


def _show_menu() -> str:
    _banner("OPTIONS SCREENER  +  CRYPTO STRATEGIST")
    print()
    print(f"  {_color('[1]', 'BRIGHT_YELLOW', bold=True)} STOCKS    "
          "— equity options screener (DISCOVER / SPREADS / IRON / etc.)")
    print(f"  {_color('[2]', 'BRIGHT_YELLOW', bold=True)} CRYPTO    "
          "— BTC/ETH options on Deribit + perp funding/basis  "
          + _color('[NEW]', 'BRIGHT_GREEN', bold=True))
    print(f"  {_color('[Q]', 'DIM', bold=False)} QUIT")
    print()
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
        if choice in ("Q", "QUIT", "EXIT", ""):
            print("  Goodbye.")
            return
        print(f"  Unknown choice: {choice!r} — pick 1, 2, or Q")


if __name__ == "__main__":
    main()
