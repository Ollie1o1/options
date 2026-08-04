"""Run an idea over history, and refuse to over-claim from it.

The loop is deliberately dull: walk each symbol's bars, ask the idea for an
entry, price it, hold it for a fixed number of days, price it again, charge
both crossings. What is not dull is the honesty machinery around it.

Tier-3 pricing needs an implied vol that no data supplies past 120 DTE. The
ratio of real ATM implied vol to trailing 30d realized vol, measured on 6,515
DoltHub observations across 9 megacaps, has a p10-p90 band of 0.70x-1.52x — a
2.2x range. Option prices are near-linear in vol over that span, so a single
modelled number is not a result. `sweep_iv` runs the whole band and reports
that a verdict flipping sign inside it is not a verdict.
"""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .. import execution_truth as et
from . import pricing as p
from .core import Context, Entry, Result, Trade

Idea = Callable[[Context], Optional[Entry]]
IVModel = Callable[[Context], float]

# The p10 / median / p90 of real implied-to-realized vol. See module docstring.
IV_MULTIPLIERS: Tuple[float, ...] = (0.70, 1.00, 1.52)


def default_iv_model(ctx: Context) -> float:
    """Trailing realized vol as the implied-vol estimate.

    The measured median implied/realized ratio is 1.02, so realized vol is very
    nearly an unbiased central estimate — but only as a CENTRE. The dispersion
    around it is what `sweep_iv` exists to expose."""
    return max(0.05, ctx.realized_vol_252d or ctx.realized_vol or 0.25)


def _strike_for_delta(ctx: Context, entry: Entry, iv: float) -> float:
    """Strike whose Black-Scholes delta is exactly the requested one.

    Closed form rather than a search. With no dividend yield a call's delta is
    N(d1) and a put's is N(d1) - 1, so

        d1 = Phi^-1(delta)                      for a call
        d1 = -Phi^-1(|delta|)                   for a put
        K  = S * exp((r + sigma^2/2)T - d1*sigma*sqrt(T))

    The 120-step search this replaces was accurate to only ~2e-3 in delta and
    cost roughly 10 million Black-Scholes evaluations per configuration, which
    is what made a full sweep take longer than the analysis was worth."""
    from scipy.stats import norm

    T = entry.dte / 365.0
    sigma = max(1e-6, iv)
    d1 = float(norm.ppf(min(0.999999, max(1e-6, entry.delta))))
    if entry.option_type == "put":
        d1 = -d1
    return float(ctx.spot * math.exp(
        (p.DEFAULT_RATE + 0.5 * sigma * sigma) * T - d1 * sigma * math.sqrt(T)))


def _fill(quote: p.Quote, side: str, policy: str, frictionless: bool) -> float:
    """Cash per share for one leg, signed against the trader."""
    if frictionless:
        return quote.mid
    return et.leg_fill(quote.bid, quote.ask, side, policy)


def run(idea: Idea, universe: Dict[str, Sequence[Tuple[str, float]]],
        iv_model: Optional[IVModel] = None, every_n: int = 5,
        policy: str = "limit", frictionless: bool = False,
        iv_multiplier: float = 1.0, max_trades: Optional[int] = None) -> Result:
    """Walk history, take what the idea asks for, return the round trips.

    `every_n` samples entry days rather than testing all of them: consecutive
    days produce near-identical overlapping positions, which inflates n without
    adding information."""
    iv_model = iv_model or default_iv_model
    trades: List[Trade] = []

    for symbol, bars in universe.items():
        bars = list(bars)
        for i in range(0, len(bars), max(1, every_n)):
            ctx = Context.at(bars, i, symbol)
            if ctx is None:
                continue
            entry = idea(ctx)
            if entry is None:
                continue

            exit_i = i + entry.hold_days
            if exit_i >= len(bars):
                continue

            iv = max(0.02, iv_model(ctx) * iv_multiplier)
            strike = _strike_for_delta(ctx, entry, iv)

            q_in = p.bs_quote(entry.option_type, ctx.spot, strike, entry.dte, iv)
            if q_in.mid <= 0.05:
                continue

            exit_spot = bars[exit_i][1]
            # Vol is carried forward unchanged: with no IV history there is
            # nothing to move it with, and inventing a path would be a second
            # assumption on top of the level. sweep_iv covers the level.
            q_out = p.bs_quote(entry.option_type, exit_spot, strike,
                               entry.dte - entry.hold_days, iv)

            open_px = _fill(q_in, entry.side, policy, frictionless)
            close_px = _fill(q_out, "sell" if entry.side == "buy" else "buy",
                             policy, frictionless)

            if entry.side == "buy":
                # Paid to open, sold to close. Risk is the premium itself.
                cost, proceeds = open_px, close_px
                if cost <= 0:
                    continue
                ret = (proceeds - cost) / cost
            else:
                # Received to open, paid to close. Two things differ from the
                # long case and getting either wrong inverts the answer:
                #   P&L is credit MINUS debit, not the other way round; and
                #   the denominator is the collateral the position ties up,
                #   not the debit paid to close it. Dividing by the debit made
                #   a cheap buy-back look like an enormous gain, which is how
                #   a short put on a flat series came out at -77%.
                credit, debit = open_px, close_px
                collateral = strike if entry.option_type == "put" else ctx.spot
                if collateral <= 0:
                    continue
                cost, proceeds = credit, debit
                ret = (credit - debit) / collateral

            trades.append(Trade(
                symbol=symbol, entry_date=ctx.date, exit_date=bars[exit_i][0],
                kind=entry.kind, strike=strike, dte=entry.dte,
                entry_price=cost, exit_price=proceeds, ret=ret,
                source=q_in.source, iv_entry=iv))

            if max_trades and len(trades) >= max_trades:
                return Result.of(trades)

    return Result.of(trades)


@dataclass(frozen=True)
class IVSweep:
    """The same idea run across the plausible implied-vol band."""
    results: Dict[float, Result] = field(default_factory=dict)

    @property
    def robust(self) -> bool:
        """True only when every multiplier agrees on the sign.

        A result that is positive at 0.70x and negative at 1.52x is a statement
        about the assumption, not about the market."""
        means = [r.mean_return for r in self.results.values()]
        if not means:
            return False
        return all(m > 0 for m in means) or all(m <= 0 for m in means)

    @property
    def verdict(self) -> str:
        if not self.results:
            return "no trades"
        means = [r.mean_return for r in self.results.values()]
        lo, hi = min(means), max(means)
        if not self.robust:
            return (f"NOT ROBUST — sign flips across the implied-vol band "
                    f"({lo:+.1%} to {hi:+.1%}); the assumption decides the answer")
        if hi <= 0:
            return f"negative across the whole band ({lo:+.1%} to {hi:+.1%})"
        return f"positive across the whole band ({lo:+.1%} to {hi:+.1%})"


def sweep_iv(idea: Idea, universe: Dict[str, Sequence[Tuple[str, float]]],
             base_iv: Optional[IVModel] = None, **kw) -> IVSweep:
    """Run `idea` at each multiplier of the measured implied/realized band."""
    return IVSweep(results={
        m: run(idea, universe, iv_model=base_iv, iv_multiplier=m, **kw)
        for m in IV_MULTIPLIERS
    })
