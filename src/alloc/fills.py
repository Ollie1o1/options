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

# Expiration is part of a leg's identity, not decoration. A chain holds many
# expirations at once, so keying a quote on (strike, type) alone lets a March
# 100-put collide with a June 100-put — the fill then comes from an arbitrary
# expiry and the resulting P&L is meaningless.
Leg = namedtuple("Leg", "expiration strike type action")  # action: buy | sell

SKIP_MISSING = "missing_quote"
SKIP_CROSSED = "crossed_quote"

Quotes = Dict[Tuple[str, float, str], Tuple[Optional[float], Optional[float]]]

# Strikes arrive both from the chain and from arithmetic (short strike minus
# width), so they must be compared at a tolerance rather than exactly.
_STRIKE_DP = 4


def _key(expiration: str, strike: float, typ: str) -> Tuple[str, float, str]:
    return (str(expiration)[:10], round(float(strike), _STRIKE_DP),
            str(typ).lower())


def fill_with_reason(legs: List[Leg], quotes: Quotes,
                     allow_worthless: bool = False
                     ) -> Tuple[Optional[float], Optional[str]]:
    """Net fill per share for the whole structure.

    Positive = net credit received. Negative = net debit paid.
    Returns (None, reason) when any leg cannot be filled honestly.

    `allow_worthless` matters enormously on the CLOSING side. An option that has
    expired out of the money genuinely has a zero bid — that is a price, not an
    absence of data. Treating it as missing meant winning positions could never
    be closed while losing ones always could, which by itself drove a 25-delta
    put spread to a 13% win rate. Entries keep the strict rule: opening a
    position on a zero bid would be inventing a fill.
    """
    net = 0.0
    for leg in legs:
        quote = quotes.get(_key(leg.expiration, leg.strike, leg.type))
        if quote is None:
            return None, SKIP_MISSING
        bid, ask = quote
        if bid is None or ask is None:
            return None, SKIP_MISSING
        if not allow_worthless and (bid <= 0 or ask <= 0):
            return None, SKIP_MISSING
        if bid < 0 or ask < 0:
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
    return [Leg(l.expiration, l.strike, l.type,
                "buy" if l.action == "sell" else "sell")
            for l in legs]


def quotes_from_chain(chain) -> Quotes:
    """Build the quote lookup a chain row list implies.

    Keyed by expiration as well as strike and type — a chain spans many
    expirations, and collapsing them would silently price one expiry's leg off
    another's quote.
    """
    return {_key(c["expiration"], c["strike"], c["type"]):
            (c.get("bid"), c.get("ask")) for c in chain}
