"""Turn an option chain into one representative candidate per structure.

The league table says WHICH structure deserves your money; this module finds a
concrete contract to express it with, and prices the two numbers the expression
filters need: capital at risk and realistically attainable profit.

"Realistically attainable" means profit at the configured take-profit target,
not theoretical max. That is the number the exit model actually acts on, so it
is the honest denominator for cost drag - a 50%-of-credit take-profit on a $35
credit targets $17.50, and a $22.60 round trip eats more than all of it.
"""
from typing import Dict, Optional

import pandas as pd

# Target short-leg delta for credit structures - the conventional ~0.30 wing.
_SHORT_DELTA = 0.30
_WING_PCT = 0.05          # width of a vertical, as a fraction of spot
_CONDOR_BODY_PCT = 0.07   # condor short strikes, as a fraction from spot


def _mid(row) -> Optional[float]:
    """Mid price, or None when the quote is unusable."""
    try:
        bid, ask = float(row["bid"]), float(row["ask"])
    except (TypeError, ValueError, KeyError):
        return None
    if bid <= 0 and ask <= 0:
        return None
    if bid <= 0 or ask <= 0:
        return max(bid, ask)
    if ask < bid:          # crossed quote - unusable
        return None
    return 0.5 * (bid + ask)


def _nearest(df: pd.DataFrame, column: str, target: float):
    """Row whose `column` is closest to `target`, or None on an empty frame."""
    if df is None or df.empty or column not in df.columns:
        return None
    diffs = (pd.to_numeric(df[column], errors="coerce") - target).abs()
    if diffs.isna().all():
        return None
    return df.loc[diffs.idxmin()]


def _vertical(long_row, short_row, credit: bool):
    """Capital at risk and gross max profit for a two-leg vertical.

    Credit spread: risk = width - credit, max profit = credit.
    Debit spread:  risk = debit,          max profit = width - debit.
    """
    lm, sm = _mid(long_row), _mid(short_row)
    if lm is None or sm is None:
        return None
    width = abs(float(long_row["strike"]) - float(short_row["strike"]))
    if width <= 0:
        return None
    if credit:
        net = sm - lm
        if net <= 0:
            return None
        return {"capital_required": (width - net) * 100.0,
                "gross_max_profit": net * 100.0}
    net = lm - sm
    if net <= 0:
        return None
    return {"capital_required": net * 100.0,
            "gross_max_profit": (width - net) * 100.0}


def _widest_affordable(legs: pd.DataFrame, short_row, capital_usd: float,
                       below: bool):
    """Widest vertical off `short_row` whose risk still fits the account.

    Width is chosen by budget, not by a fixed percentage of spot. A fixed 5%
    wing is $37 wide on SPY - correct arithmetic, useless at 700 CAD. Wider is
    better when it fits, because the fixed round-trip cost is a smaller share
    of a bigger credit; that is the "same trade, bigger credit" lever.
    """
    short_k = float(short_row["strike"])
    side = legs[legs["strike"] < short_k] if below else legs[legs["strike"] > short_k]
    if side.empty:
        return None
    # Try widest first so we take the most credit the budget allows.
    ordered = side.sort_values("strike", ascending=below)
    best = None
    for _, long_row in ordered.iterrows():
        built = _vertical(long_row, short_row, credit=True)
        if not built:
            continue
        if built["capital_required"] <= capital_usd:
            return built            # widest that fits
        best = built                # remember the narrowest seen so far
    # Nothing fit: return the narrowest so the engine can reject it honestly
    # with a real number rather than silently omitting the structure.
    for _, long_row in side.sort_values("strike", ascending=not below).iterrows():
        built = _vertical(long_row, short_row, credit=True)
        if built:
            return built
    return best


def _take_profit_fraction(exit_rules: dict, strategy: str) -> float:
    """Fraction of gross max profit the exit model actually targets."""
    rules = exit_rules or {}
    if strategy in ("Long Call", "Long Put"):
        return float((rules.get("long_option") or {}).get("take_profit", 1.0))
    if strategy in ("Bull Put", "Bear Call", "Iron Condor"):
        return float((rules.get("spread") or {}).get("take_profit", 0.5))
    if strategy == "Short Put":
        return float((rules.get("short_premium") or {})
                     .get("take_profit_ge_21_dte", 0.5))
    return 1.0


def build_candidates(chain: pd.DataFrame, spot: float,
                     exit_rules: Optional[dict] = None,
                     capital_usd: float = 511.0) -> Dict[str, dict]:
    """One representative candidate per structure from a single-expiry chain.

    Returns {strategy: {"capital_required": float, "max_profit": float}} where
    max_profit is the take-profit target, not the theoretical maximum. Any
    structure that cannot be built from this chain is simply absent - the
    expression engine then reports "no candidate contract found" for it rather
    than inventing a price.
    """
    out: Dict[str, dict] = {}
    if chain is None or chain.empty or not spot or spot <= 0:
        return out

    chain = chain.copy()
    chain["strike"] = pd.to_numeric(chain["strike"], errors="coerce")
    chain = chain[chain["strike"].notna()]
    calls = chain[chain["type"] == "call"].sort_values("strike")
    puts = chain[chain["type"] == "put"].sort_values("strike")
    if calls.empty or puts.empty:
        return out

    raw: Dict[str, dict] = {}

    # --- debit singles: closest-to-ATM long that the account can afford ------
    # ATM is preferred, but on a $737 underlying the ATM call is ~$1,600. Walk
    # OTM until the premium fits, so the engine gets a real contract to judge
    # rather than an automatic affordability rejection.
    for name, legs, otm_ascending in (("Long Call", calls, True),
                                      ("Long Put", puts, False)):
        atm = _nearest(legs, "strike", spot)
        if atm is None:
            continue
        ordered = legs.sort_values("strike", ascending=otm_ascending)
        ordered = ordered[ordered["strike"] >= float(atm["strike"])] \
            if otm_ascending else ordered[ordered["strike"] <= float(atm["strike"])]
        chosen = None
        for _, row in ordered.iterrows():
            mid = _mid(row)
            if mid is None:
                continue
            if chosen is None:
                chosen = row        # fall back to ATM if nothing fits
            if mid * 100.0 <= capital_usd:
                chosen = row
                break
        if chosen is not None and _mid(chosen) is not None:
            cost = _mid(chosen) * 100.0
            raw[name] = {"capital_required": cost, "gross_max_profit": cost}

    # --- Bull Put: short put below spot, long put as far below as fits ------
    short_p = _nearest(puts, "strike", spot * (1 - _WING_PCT))
    if short_p is not None:
        built = _widest_affordable(puts, short_p, capital_usd, below=True)
        if built:
            raw["Bull Put"] = built

    # --- Bear Call: short call above spot, long call as far above as fits ---
    short_c = _nearest(calls, "strike", spot * (1 + _WING_PCT))
    if short_c is not None:
        built = _widest_affordable(calls, short_c, capital_usd, below=False)
        if built:
            raw["Bear Call"] = built

    # --- Iron Condor: both credit wings, risk is the wider single side ------
    if "Bull Put" in raw and "Bear Call" in raw:
        credit = (raw["Bull Put"]["gross_max_profit"]
                  + raw["Bear Call"]["gross_max_profit"])
        widest = max(raw["Bull Put"]["capital_required"]
                     + raw["Bull Put"]["gross_max_profit"],
                     raw["Bear Call"]["capital_required"]
                     + raw["Bear Call"]["gross_max_profit"])
        risk = widest - credit
        if risk > 0:
            raw["Iron Condor"] = {"capital_required": risk,
                                  "gross_max_profit": credit}

    # --- Short Put: cash-secured, capital is the strike ----------------------
    if short_p is not None and _mid(short_p) is not None:
        premium = _mid(short_p) * 100.0
        raw["Short Put"] = {
            "capital_required": float(short_p["strike"]) * 100.0 - premium,
            "gross_max_profit": premium}

    for strategy, vals in raw.items():
        frac = _take_profit_fraction(exit_rules, strategy)
        out[strategy] = {
            "capital_required": round(vals["capital_required"], 2),
            "max_profit": round(vals["gross_max_profit"] * frac, 2)}
    return out
