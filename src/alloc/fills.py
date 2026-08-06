"""Execution prices. The single rule: you always cross the spread.

Sell legs fill at the bid, buy legs at the ask, entry and exit alike. Nothing
here ever computes a mid.

This module is deliberately small and dependency-free. It is one of the two
places in the system where a subtle error silently manufactures an edge — the
ledger's own execution audit found that mid-priced entries cost 27% of credit
once actually crossed, and that re-pricing at the real fill INVERTED the
ranking between strategies. A backtest filling at mid would not look wrong; it
would look good, and recommend the wrong structure.

A missing or crossed quote produces no fill at all. It is reported and counted,
never modelled, because a strategy that only works when you ignore bad quotes
has not worked.
"""
from __future__ import annotations

from collections import namedtuple
from typing import Dict, List, Optional, Tuple

Leg = namedtuple("Leg", "strike type action")   # action: "buy" | "sell"

SKIP_MISSING = "missing_quote"
SKIP_CROSSED = "crossed_quote"

Quotes = Dict[Tuple[float, str], Tuple[Optional[float], Optional[float]]]

# Strikes arrive both from the chain and from arithmetic (short strike minus
# width), so they must be compared at a tolerance rather than exactly.
_STRIKE_DP = 4


def _key(strike: float, typ: str) -> Tuple[float, str]:
    return (round(float(strike), _STRIKE_DP), str(typ).lower())


def fill_with_reason(legs: List[Leg],
                     quotes: Quotes) -> Tuple[Optional[float], Optional[str]]:
    """Net fill per share for the whole structure.

    Positive = net credit received. Negative = net debit paid.
    Returns (None, reason) when any leg cannot be filled honestly.
    """
    net = 0.0
    for leg in legs:
        quote = quotes.get(_key(leg.strike, leg.type))
        if quote is None:
            return None, SKIP_MISSING
        bid, ask = quote
        if bid is None or ask is None or bid <= 0 or ask <= 0:
            return None, SKIP_MISSING
        if bid > ask:
            return None, SKIP_CROSSED
        # Cross the spread, always against us.
        net += bid if leg.action == "sell" else -ask
    return round(net, 4), None


def fill_price(legs: List[Leg], quotes: Quotes) -> Optional[float]:
    return fill_with_reason(legs, quotes)[0]


def reverse(legs: List[Leg]) -> List[Leg]:
    """The closing side of a position.

    Closing crosses the spread a second time, which is the entire reason
    holding to expiry is cheaper than managing at a profit target: held to
    expiry you pay the opening legs only.
    """
    return [Leg(l.strike, l.type, "buy" if l.action == "sell" else "sell")
            for l in legs]


def quotes_from_chain(chain) -> Quotes:
    """Build the quote lookup a chain row list implies."""
    return {_key(c["strike"], c["type"]): (c.get("bid"), c.get("ask"))
            for c in chain}
