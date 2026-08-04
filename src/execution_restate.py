"""Ledger rows -> priced legs, so a logged trade can be re-read at its true fill.

`execution_truth` knows about quotes and policies and nothing else. This module
is the adapter that knows the ledger's shape: which columns hold which leg of
which structure, and what to do when they don't hold enough (refuse — never
half-price a structure).

Used by `scripts/restate_execution.py` to backfill the v18 columns.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from . import execution_truth as et

# Which ledger columns describe the legs of each structure, and on which side.
# `strike` is the short leg for the two-leg credit structures; `long_strike`
# the protective wing.
_SPREADS = {
    "Bull Put": ("put", "put"),
    "Bear Call": ("call", "call"),
}
_SINGLES = {
    "Long Call": ("call", "buy"),
    "Long Put": ("put", "buy"),
    "Short Put": ("put", "sell"),
    "Short Call": ("call", "sell"),
}

# A quote lookup takes (strike, option_type) and returns (bid, ask) or None.
QuoteLookup = Callable[[float, str], Optional[tuple]]


def _f(row: Dict[str, Any], key: str) -> Optional[float]:
    v = row.get(key)
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f


def legs_from_trade(row: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The legs a logged trade actually consists of, or None if the row does
    not carry enough to say.

    Refusing is the point. 13 of 187 logged iron condors stored only their put
    legs; pricing those as a two-leg structure would halve their measured
    friction and flatter the book exactly where it is already flattered."""
    strategy = row.get("strategy_name")

    if strategy == "Iron Condor":
        sc, lc = _f(row, "short_call_strike"), _f(row, "long_call_strike")
        sp, lp = _f(row, "short_put_strike"), _f(row, "long_put_strike")
        if None in (sc, lc, sp, lp):
            return None
        return [
            {"strike": sc, "type": "call", "side": "sell"},
            {"strike": lc, "type": "call", "side": "buy"},
            {"strike": sp, "type": "put", "side": "sell"},
            {"strike": lp, "type": "put", "side": "buy"},
        ]

    if strategy in _SPREADS:
        short_type, long_type = _SPREADS[strategy]
        short, long = _f(row, "strike"), _f(row, "long_strike")
        if short is None or long is None:
            return None
        return [
            {"strike": short, "type": short_type, "side": "sell"},
            {"strike": long, "type": long_type, "side": "buy"},
        ]

    if strategy in _SINGLES:
        opt_type, side = _SINGLES[strategy]
        strike = _f(row, "strike")
        if strike is None:
            return None
        return [{"strike": strike, "type": opt_type, "side": side}]

    return None


_UNPRICED = {
    "entry_price_mid": None,
    "entry_price_fill": None,
    "entry_price_cross": None,
    "fill_policy": None,
    "fill_source": "unknown",
}


def restate(row: Dict[str, Any], quotes: QuoteLookup,
            policy: str = "limit", k: Optional[float] = None,
            source: str = "live_quote") -> Dict[str, Any]:
    """The five v18 column values for one ledger row.

    Always returns a dict. When the row cannot be priced the values are NULL
    and `fill_source` is 'unknown' — an absent number, never an invented one."""
    legs = legs_from_trade(row)
    if legs is None:
        return dict(_UNPRICED)

    priced: List[Dict[str, Any]] = []
    for leg in legs:
        q = quotes(leg["strike"], leg["type"])
        if q is None:
            return dict(_UNPRICED)
        priced.append({"bid": q[0], "ask": q[1], "side": leg["side"]})

    report = et.edge_report(priced, width=_f(row, "spread_width") or 1.0, k=k)
    if report is None:
        return dict(_UNPRICED)

    return {
        "entry_price_mid": report.fills["mid"].price,
        "entry_price_fill": report.fills[policy].price,
        "entry_price_cross": report.fills["cross"].price,
        "fill_policy": policy,
        "fill_source": source,
    }
