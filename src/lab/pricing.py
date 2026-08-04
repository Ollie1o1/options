"""What an option was worth on a given day, and how we know.

Three tiers, and every price says which one it came from:

  real_marks  a genuine two-sided quote from `data/dolt_options.db` (9 symbols,
              <=67 DTE, real IV too) or `data/chain_archive.db` (15 symbols,
              <=120 DTE, 22 snapshot days).
  modeled     Black-Scholes off `data/squeeze_prices.db` (21k symbols, 2017-2026)
              with a MODELLED vol. The only tier that reaches past 120 DTE, and
              the only one carrying an assumption.

The two are never pooled into one number. A backtest that mixes a measured
result with a modelled one and reports a single figure is how the mid-fill
assumption got into the ledger; here the tier travels with the price.

A modelled quote always has two sides. A synthetic mid with no spread makes
friction free, and free friction is the defect this whole layer exists to stop.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from ..utils import bs_price

SOURCES = ("real_marks", "modeled")

# Median half-spread as a fraction of premium, |delta| 0.30-0.60, measured on
# data/chain_archive.db (2026-06-10 to 2026-08-04, 15 symbols, n=82,669 quotes).
# The relative cost falls with maturity because the premium grows faster than
# the quoted spread.
_HALF_SPREAD_BY_DTE = (
    (30, 0.017),    # 5-30 DTE,   n=41,234
    (60, 0.011),    # 31-60 DTE,  n=22,966
    (120, 0.007),   # 61-120 DTE, n=18,469
)

DEFAULT_RATE = 0.045


@dataclass(frozen=True)
class Quote:
    """A two-sided market, and its provenance."""
    bid: float
    ask: float
    source: str
    iv: Optional[float] = None

    def __post_init__(self):
        if self.source not in SOURCES:
            raise ValueError(
                f"unknown price source {self.source!r}, expected one of {SOURCES}")

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def is_real(self) -> bool:
        """True only for a genuine market quote. Callers split on this rather
        than averaging across it."""
        return self.source == "real_marks"


def modeled_half_spread_frac(dte: int) -> float:
    """Half-spread as a fraction of premium for a synthetic quote.

    Beyond 120 DTE the archive has no data, so the last measured value is held
    flat. Extrapolating the downward trend would keep making long-dated trades
    look cheaper on no evidence — the direction that flatters the trade is
    exactly the one to refuse."""
    for limit, frac in _HALF_SPREAD_BY_DTE:
        if dte <= limit:
            return frac
    return _HALF_SPREAD_BY_DTE[-1][1]


def bs_quote(option_type: str, spot: float, strike: float, dte: int,
             iv: float, rate: float = DEFAULT_RATE,
             half_spread_frac: Optional[float] = None) -> Quote:
    """Tier-3 price: Black-Scholes, with a spread wide enough to charge for.

    At or past expiry the option is worth its intrinsic and nothing else, and
    is quoted with no spread — settlement is not a trade."""
    if dte <= 0:
        intrinsic = (max(0.0, spot - strike) if option_type == "call"
                     else max(0.0, strike - spot))
        return Quote(bid=intrinsic, ask=intrinsic, source="modeled", iv=iv)

    T = dte / 365.0
    mid = float(bs_price(option_type, spot, strike, T, rate, iv))
    frac = modeled_half_spread_frac(dte) if half_spread_frac is None else half_spread_frac
    half = mid * frac
    return Quote(bid=max(0.0, mid - half), ask=mid + half, source="modeled", iv=iv)


def implied_move(spot: float, iv: float, dte: int) -> float:
    """One standard deviation of the underlying over `dte` calendar days."""
    return spot * iv * math.sqrt(max(0, dte) / 365.0)
