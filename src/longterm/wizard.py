"""Pure input-collection helpers for the HOLDINGS guided menu.

No input()/print() here — board.py's menu() drives the interactive I/O and
calls these to parse answers and build the exact command strings
handle_command() already accepts. Keeping this pure (no I/O) means every
piece of the guided menu's logic is unit-testable without mocking input().
"""
import re
from typing import List

_LEVEL_SPLIT = re.compile(r"[,/\s]+")


def _fmt_num(x: float) -> str:
    """Format a number preserving full precision without scientific notation."""
    s = f"{x:.10f}".rstrip("0").rstrip(".")
    return s if s else "0"


def parse_levels(text: str) -> List[float]:
    """Turns "750, 650, 550" / "750/650/550" / "750 650 550" into
    [750.0, 650.0, 550.0] — sorted highest first regardless of typed order.
    Raises ValueError with a friendly message on unparseable input."""
    tokens = [tok for tok in _LEVEL_SPLIT.split(text.strip()) if tok]
    if not tokens:
        raise ValueError("enter at least one price level")
    try:
        levels = [float(tok) for tok in tokens]
    except ValueError:
        raise ValueError(f"'{text}' isn't a list of prices — try e.g. 750, 650, 550")
    return sorted(levels, reverse=True)


def build_add_command(ticker: str, levels: List[float]) -> str:
    ladder = "/".join(_fmt_num(lvl) for lvl in levels)
    return f"ADD {ticker.upper()} {ladder}"


def build_edit_command(ticker: str, levels: List[float]) -> str:
    ladder = "/".join(_fmt_num(lvl) for lvl in levels)
    return f"EDIT {ticker.upper()} {ladder}"


def build_remove_command(ticker: str) -> str:
    return f"REMOVE {ticker.upper()}"


def build_cash_command(amount: float) -> str:
    return f"CASH {_fmt_num(amount)}"


def build_fill_command(ticker: str, level: float, shares: float, price: float) -> str:
    return f"FILL {ticker.upper()} {_fmt_num(level)} {_fmt_num(shares)} {_fmt_num(price)}"
