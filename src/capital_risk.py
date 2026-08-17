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


def _pick_get(pick, key):
    """Field lookup tolerant of dicts, pandas Series, and missing columns."""
    try:
        return pick[key]
    except (KeyError, IndexError, TypeError):
        return None


def _pick_premium(pick) -> Optional[float]:
    """The per-share premium of a scan row.

    A scan row carries no ``entry_price``; the auto-log loop derives it as
    ``ask`` -> ``lastPrice`` -> ``premium``, treating a zero/absent quote as
    missing rather than free. Sizing must use the identical ladder, or a pick
    prices as unknown here and as a real debit two lines later.
    """
    for key in ("entry_price", "ask", "lastPrice", "premium"):
        value = _finite(_pick_get(pick, key))
        if value is not None and value > 0:
            return value
    return None


def capital_at_risk_for_pick(pick, strategy_name: str) -> Optional[float]:
    """capital_at_risk for a *scanner result* row, as ranked before logging.

    Scan rows are shaped differently from trades rows: the ticker lives in
    ``symbol``, the condor worst case in ``max_risk``, and the vertical worst
    case in ``max_loss`` — the names ``log_spread``/``log_iron_condor`` map to
    ``max_loss_usd`` only at insert time. Sizing a candidate therefore cannot go
    through ``capital_at_risk_for_row``, which expects the post-insert names.

    ``strategy_name`` is passed in rather than read off the row because the scan
    encodes it positionally (mode + leg type), not as a column.
    """
    name = (strategy_name or "").strip().lower()
    # Only a defined-risk multi-leg structure records a TRUE worst case. Every
    # single-leg row also carries a `max_loss`, but trade_analysis computes it as
    # `entry_price * 100` — the long-premium debit, stamped on shorts too. Trusting
    # it there sized a cash-secured put at $50 instead of $31,850.
    stored = None
    if any(key in name for key in _CREDIT_STRUCTURE_KEYS):
        stored = _pick_get(pick, "max_risk")
        if _finite(stored) is None or _finite(stored) <= 0:
            stored = _pick_get(pick, "max_loss")

    credit = _pick_get(pick, "net_credit")
    if credit is None:
        credit = _pick_get(pick, "total_credit")

    return capital_at_risk(
        strategy_name=strategy_name or "",
        entry_price=_pick_premium(pick),
        strike=_pick_get(pick, "strike"),
        max_loss_usd=stored,
        spread_width=_pick_get(pick, "spread_width"),
        net_credit=credit,
        quantity=_pick_get(pick, "quantity") if _pick_get(pick, "quantity") is not None else 1.0,
        ticker=_pick_get(pick, "symbol"),
    )


def max_profit_for_pick(pick, strategy_name: str) -> Optional[float]:
    """Most a *scanner result* row can make, in dollars, or None if unbounded.

    The reward side of `capital_at_risk_for_pick`, and deliberately its
    neighbour: both read the premium through `_pick_premium`, so the two cells
    of one `Reward/$risk` figure can never come from different quotes.

    Only credit structures and short single legs have an answer:

    * a credit structure keeps its credit — the stored `max_profit` when the
      spread builder set one, else the credit itself;
    * a short leg keeps its credit, which is the entire upside;
    * a LONG leg returns None. For a call that is simply true. For a put it is
      a judgement: (strike - premium) x 100 is arithmetically real but is only
      collected if the underlying goes to ZERO, and on a per-dollar-of-risk
      axis it would print ~44x and outrank every credit structure on the
      board on an outcome that has not happened to a listed name in this
      universe. A blank cell says "not answerable"; the alternative is a
      column that silently recommends long puts.

    None never means zero — see `budget_view.per_risk`.
    """
    name = (strategy_name or "").strip().lower()
    mult = _multiplier(_pick_get(pick, "symbol"))
    price = _finite(_pick_premium(pick))

    if any(key in name for key in _CREDIT_STRUCTURE_KEYS):
        stored = _finite(_pick_get(pick, "max_profit"))
        if stored is not None and stored > 0:
            return stored
        credit = _finite(_pick_get(pick, "net_credit"))
        if credit is None:
            credit = _finite(_pick_get(pick, "total_credit"))
        if credit is None:
            credit = price
        return credit * mult if credit is not None and credit > 0 else None

    if any(key in name for key in _SHORT_PUT_KEYS + _SHORT_CALL_KEYS):
        return price * mult if price is not None and price > 0 else None

    if is_short_position(name):
        return price * mult if price is not None and price > 0 else None

    return None


def pick_within_budget(pick, strategy_name: str, cap: Optional[float]) -> bool:
    """True if a scanner candidate fits the budget, so it is worth a top-N slot.

    Applied to the candidate pool *before* the top-N cut. Filtering after the
    cut lets unaffordable picks consume every slot and log nothing — observed
    2026-07-30, when all 5 short-put slots went to $13k-$74k collateral trades.
    """
    return within_budget(capital_at_risk_for_pick(pick, strategy_name), cap)
