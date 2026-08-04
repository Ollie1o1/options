"""The vocabulary for expressing a tradeable idea.

An idea is a pure function `(Context) -> Entry | None`. It sees only what was
knowable on the day, returns a single-leg structure or nothing, and the engine
does the rest. Keeping ideas pure is what makes them cheap to write and hard to
cheat with.

Single leg only, deliberately. Measured on real archived quotes, one crossing
costs 0.7-1.7% of a single leg's premium against 33% of a two-leg credit
spread's credit — friction on a spread is denominated in the legs while the
reward is denominated in their difference. Nothing here builds spreads.
"""
from __future__ import annotations

import math
import statistics as st
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# What a single leg can be, and which way it points.
KINDS = {
    "long_call": ("call", "buy"),
    "long_put": ("put", "buy"),
    "short_call": ("call", "sell"),
    "short_put": ("put", "sell"),
}

MIN_HISTORY = 60          # bars needed before any feature is trustworthy


@dataclass(frozen=True)
class Entry:
    """One single-leg position: what to buy, how far out, how long to hold.

    `hold_days` is a fixed holding period with no stop and no target — one
    parameter, the least overfittable exit there is, and the cleanest read on
    whether the ENTRY signal has edge rather than the exit tuning."""
    kind: str
    delta: float
    dte: int
    hold_days: int

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown kind {self.kind!r}, expected one of {sorted(KINDS)}")
        if not 0.0 < self.delta < 1.0:
            raise ValueError(f"delta must be in (0,1), got {self.delta}")
        if self.dte <= 0:
            raise ValueError(f"dte must be positive, got {self.dte}")
        if self.hold_days <= 0:
            raise ValueError(f"hold_days must be positive, got {self.hold_days}")
        if self.hold_days > self.dte:
            raise ValueError(
                f"hold_days {self.hold_days} outlives the option's {self.dte} DTE")

    @property
    def option_type(self) -> str:
        return KINDS[self.kind][0]

    @property
    def side(self) -> str:
        return KINDS[self.kind][1]

    @property
    def n_legs(self) -> int:
        return 1


@dataclass(frozen=True)
class Context:
    """What was knowable on the day. Price-derived only.

    Nothing here depends on implied vol: `iv_cache.db` holds 83 days across 251
    tickers, which cannot support a 9-year backtest, and an idea that silently
    depended on it would only be testable on the last quarter of the sample."""
    symbol: str
    date: str
    spot: float
    realized_vol: float
    realized_vol_252d: float
    momentum_21d: float
    momentum_63d: float
    drawdown: float
    index: int = 0

    @staticmethod
    def at(bars: Sequence[Tuple[str, float]], index: int,
           symbol: str) -> Optional["Context"]:
        """Features as of `bars[index]`, or None if history is too thin.

        Every window looks strictly backwards. Returning None rather than a
        partial context is what keeps a thin-history symbol from quietly
        entering the sample with noisier features than everything else."""
        if index < MIN_HISTORY or index >= len(bars):
            return None
        closes = [c for _, c in bars[: index + 1]]
        spot = closes[-1]
        if spot <= 0:
            return None

        def vol(window: int) -> float:
            cs = closes[-(window + 1):]
            rs = [math.log(cs[i + 1] / cs[i]) for i in range(len(cs) - 1)
                  if cs[i] > 0 and cs[i + 1] > 0]
            if len(rs) < 10:
                return 0.0
            return st.stdev(rs) * math.sqrt(252)

        peak = max(closes[-252:]) if len(closes) >= 2 else spot
        return Context(
            symbol=symbol,
            date=bars[index][0],
            spot=spot,
            realized_vol=vol(30),
            realized_vol_252d=vol(min(252, len(closes) - 1)),
            momentum_21d=spot / closes[-22] - 1.0 if len(closes) > 22 else 0.0,
            momentum_63d=spot / closes[-64] - 1.0 if len(closes) > 64 else 0.0,
            drawdown=spot / peak - 1.0 if peak > 0 else 0.0,
            index=index,
        )


@dataclass(frozen=True)
class Trade:
    """One round trip, net of what it cost to get in and out."""
    symbol: str
    entry_date: str
    exit_date: str
    kind: str
    strike: float
    dte: int
    entry_price: float
    exit_price: float
    ret: float
    source: str
    iv_entry: float


@dataclass(frozen=True)
class Result:
    """What a run produced. `source_counts` is reported, never averaged over."""
    trades: List[Trade]
    n: int
    mean_return: float
    median_return: float
    win_rate: float
    source_counts: Dict[str, int] = field(default_factory=dict)

    @staticmethod
    def of(trades: List[Trade]) -> "Result":
        if not trades:
            return Result([], 0, 0.0, 0.0, 0.0, {})
        rets = [t.ret for t in trades]
        counts: Dict[str, int] = {}
        for t in trades:
            counts[t.source] = counts.get(t.source, 0) + 1
        return Result(
            trades=trades,
            n=len(trades),
            mean_return=st.mean(rets),
            median_return=st.median(rets),
            win_rate=sum(1 for r in rets if r > 0) / len(rets),
            source_counts=counts,
        )
