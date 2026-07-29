"""One definition of the dollars a paper trade puts at risk.

Why this exists: capital at risk was being re-derived ad hoc at every call
site, usually as ``max_loss_usd or entry_price * 100``. That formula is right
for defined-risk structures and long premium and badly wrong for naked shorts —
it costs a cash-secured put at the credit received rather than the collateral
tied up, which understates the risk of a WFC 77.5 put by ~50x. Strategy
comparisons and the auto-log budget gate both depend on this number, so it is
defined once, here, and tested in tests/test_capital_risk.py.

The number answers "how much of the account does this position consume until
it closes" — not notional, and not the theoretical worst case of an
unhedgeable position. Where risk cannot be bounded (naked calls) the answer is
``None``, which callers must treat as unsizable rather than free.
"""
from __future__ import annotations

from typing import Optional

from .utils import is_short_position

# Names are matched as substrings, not exact strings: the ledger and the
# lottery sleeve both qualify them ("Lottery Long Call", "Bull Call Spread"),
# and exact-set matching silently classified those as unsizable.
#
# Defined-risk credit structures. Their loss is width minus credit, never the
# credit — note none of these contain a word that is_short_position() detects,
# so they must be named here explicitly.
_CREDIT_STRUCTURE_KEYS = ("bull put", "bear call", "iron condor", "credit")
# Collateral-backed short single legs.
_SHORT_PUT_KEYS = ("short put", "cash-secured put", "cash secured put", "naked put")
# Unbounded upside — cannot be sized from stored fields.
_SHORT_CALL_KEYS = ("short call", "naked call")


def _multiplier(ticker: Optional[str]) -> float:
    """Contract multiplier: 1 for crypto (whole-coin rows), 100 for equity."""
    return 1.0 if (ticker or "").upper() in ("BTC", "ETH") else 100.0


def _finite(value) -> Optional[float]:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out and out not in (float("inf"), float("-inf")) else None


def capital_at_risk(
    strategy_name: str,
    entry_price=None,
    strike=None,
    max_loss_usd=None,
    spread_width=None,
    net_credit=None,
    quantity=1.0,
    ticker: Optional[str] = None,
) -> Optional[float]:
    """Dollars tied up by one position, or None when it cannot be bounded.

    Resolution order:
      1. A stored positive ``max_loss_usd`` — defined-risk structures record
         their true worst case at log time, so it always wins.
      2. A credit structure without a stored max loss — derive width minus
         credit, and report None if the width is unknown. Never the credit.
      3. Short put — collateral, ``(strike - credit) x multiplier``.
      4. Short call — None; naked upside cannot be bounded.
      5. Anything else that is not a short position — the debit paid. This
         covers long premium, the lottery sleeve, and debit spreads.
    """
    qty = _finite(quantity)
    if qty is None or qty <= 0:
        qty = 1.0
    mult = _multiplier(ticker)

    stored = _finite(max_loss_usd)
    if stored is not None and stored > 0:
        return stored * qty

    name = (strategy_name or "").strip().lower()
    price = _finite(entry_price)

    if any(key in name for key in _CREDIT_STRUCTURE_KEYS):
        width = _finite(spread_width)
        credit = _finite(net_credit)
        if credit is None:
            credit = price
        if width is None or credit is None or width <= 0:
            return None
        return max(0.0, width - credit) * mult * qty

    if any(key in name for key in _SHORT_PUT_KEYS):
        k = _finite(strike)
        if k is None or price is None:
            return None
        return max(0.0, k - price) * mult * qty

    if any(key in name for key in _SHORT_CALL_KEYS):
        return None

    if is_short_position(name):
        # Some other short structure with no stored max loss — unsizable.
        return None

    if price is None or price <= 0 or not name:
        return None
    return price * mult * qty


def within_budget(risk: Optional[float], cap: Optional[float]) -> bool:
    """True if a position fits the cap. No cap means no constraint.

    Unknown risk fails a set cap deliberately: a position whose loss cannot be
    bounded cannot be shown to fit inside a budget.
    """
    limit = _finite(cap)
    if limit is None or limit <= 0:
        return True
    if risk is None:
        return False
    return risk <= limit


def capital_at_risk_for_row(row) -> Optional[float]:
    """capital_at_risk for a sqlite3.Row / dict / mapping of a trades row.

    Tolerates rows missing the multi-leg columns (legacy CSV-era imports).
    """
    def get(key):
        try:
            return row[key]
        except (KeyError, IndexError, TypeError):
            return None

    return capital_at_risk(
        strategy_name=get("strategy_name") or "",
        entry_price=get("entry_price"),
        strike=get("strike"),
        max_loss_usd=get("max_loss_usd"),
        spread_width=get("spread_width"),
        net_credit=get("net_credit"),
        quantity=get("quantity") if get("quantity") is not None else 1.0,
        ticker=get("ticker"),
    )
