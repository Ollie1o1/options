"""The STRATEGIES desk: browse setups, read one in full, filter by account.

Display only, and structurally so — this module imports the board, the library
and nothing from any execution path. A setup here is a written intention with a
cost attached, never an order.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from .. import formatting as fmt
from .board import format_board, format_detail
from .record import StrategyRecord

_PROMPT = ("  [setup id or #] detail   [T] TFSA-only toggle   [B] back  > ")


def _resolve(token: str, records: List[StrategyRecord]) -> Optional[StrategyRecord]:
    """A row number or a setup id. Anything else is not a selection."""
    token = token.strip()
    if not token:
        return None
    if token.isdigit():
        idx = int(token) - 1
        if 0 <= idx < len(records):
            return records[idx]
        return None
    for r in records:
        if r.spec.id.lower() == token.lower():
            return r
    return None


def run_menu(records: Optional[Iterable[StrategyRecord]] = None,
             account: Optional[str] = None,
             table: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
    """Browse the setup library. `account` starts the view filtered."""
    if records is None:
        from .seed import LIBRARY
        records = LIBRARY
    all_records = list(records)

    while True:
        shown = ([r for r in all_records if r.tradeable_in(account)]
                 if account else all_records)
        print()
        print(format_board(shown, account=account, table=table))
        print()
        try:
            choice = input(_PROMPT).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return

        upper = choice.upper()
        if upper in ("B", "BACK", "Q", "QUIT", ""):
            return
        if upper in ("T", "TFSA"):
            account = None if account else "tfsa"
            continue

        record = _resolve(choice, shown)
        if record is None:
            print(fmt.style(f"  No setup matches {choice!r}.", "warn"))
            continue
        print()
        print(format_detail(record, table=table))
        print()
        try:
            input("  [enter] back to the board  > ")
        except (EOFError, KeyboardInterrupt):
            print()
            return
