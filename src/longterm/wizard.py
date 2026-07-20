"""Pure input-collection helpers for the HOLDINGS guided menu.

No input()/print() here — board.py's menu() drives the interactive I/O and
calls these to parse answers and build the exact command strings
handle_command() already accepts. Keeping this pure (no I/O) means every
piece of the guided menu's logic is unit-testable without mocking input().
"""
import re
from typing import List

_LEVEL_SPLIT = re.compile(r"[,/\s]+")


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
