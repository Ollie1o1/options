"""Daily leverage signal candidates behind one interface.

Every candidate exposes `.name` and `.walk_forward(frames, funding, costs,
train_frac)` returning (in_sample_trades, oos_trades) as lists of `Trade`, net
of per-asset cost. The harness (validate.py) scores them identically, so adding
a candidate never touches the harness.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from . import swing as S
from . import funding as FND


@dataclass(frozen=True)
class Trade:
    symbol: str
    side: str
    entry_date: object
    exit_date: object
    entry: float
    exit: float
    ret_net: float
    r_multiple: float
    bars_held: int
    reason: str


def to_trade(symbol: str, st: S.SwingTrade) -> Trade:
    """Adapt a single-asset swing SwingTrade (already net of the cost passed to
    swing.backtest) into the common Trade record."""
    return Trade(symbol=symbol, side=st.side, entry_date=st.entry_date,
                 exit_date=st.exit_date, entry=st.entry, exit=st.exit,
                 ret_net=st.ret, r_multiple=st.r_multiple,
                 bars_held=st.bars_held, reason=st.reason)


def _split(n: int, train_frac: float) -> int:
    return max(S.DEFAULT_LOOKBACK + 1, int(n * train_frac))


class TrendCandidate:
    """Baseline: the existing daily Donchian breakout + ATR-chandelier stop.
    Stop multiple k is calibrated on the train slice per symbol, then held fixed
    on the OOS tail. Longs allowed on all symbols; shorts on all (crypto trends
    both ways at the daily horizon)."""
    name = "trend_breakout"

    def __init__(self, allow_short: bool = True):
        self.allow_short = allow_short

    def walk_forward(self, frames: Dict[str, pd.DataFrame],
                     funding: Dict[str, pd.Series],
                     costs: Dict[str, float],
                     train_frac: float = 0.6) -> Tuple[List[Trade], List[Trade]]:
        is_all: List[Trade] = []
        oos_all: List[Trade] = []
        for sym, df in frames.items():
            f = S.compute_features(df)
            n = len(f)
            split = _split(n, train_frac)
            k_long = S.calibrate_stop_k(f, "long", max_i=split)
            k_short = S.calibrate_stop_k(f, "short", max_i=split)
            cost = costs.get(sym, S.DEFAULT_COST)
            is_t = S.backtest(f.iloc[:split], k_long, k_short, cost=cost,
                              allow_short=self.allow_short)
            oos_t = S.backtest(f.iloc[split:], k_long, k_short, cost=cost,
                               allow_short=self.allow_short)
            is_all += [to_trade(sym, t) for t in is_t]
            oos_all += [to_trade(sym, t) for t in oos_t]
        return is_all, oos_all


class FundingContrarianCandidate:
    """Fade crowded funding extremes. When trailing-window funding z-score rises
    above +z_threshold (longs crowded, paying to hold), go SHORT; below
    -z_threshold (shorts crowded), go LONG. Fixed `horizon`-day hold with an
    ATR stop at entry. Thresholds are FIXED (not fit) so the harness judges a
    real, un-tuned edge. One position per symbol at a time (no overlap)."""
    name = "funding_contrarian"

    def __init__(self, z_threshold: float = 1.5, horizon: int = 5,
                 stop_k: float = 2.0, window: int = 30):
        self.z = z_threshold
        self.h = horizon
        self.k = stop_k
        self.window = window

    def _trades(self, sym, df, fund, cost) -> List[Trade]:
        f = df.copy()
        f["atr"] = S.atr(f)
        z = FND.zscore(FND.align_daily(fund, f.index), self.window)
        c = f["close"].values
        h = f["high"].values
        l = f["low"].values
        a = f["atr"].values
        zz = z.values
        n = len(c)
        out: List[Trade] = []
        i = self.window
        idx = list(f.index)
        while i < n - 1:
            if not np.isfinite(zz[i]) or not np.isfinite(a[i]) or a[i] <= 0:
                i += 1
                continue
            if zz[i] >= self.z:
                side = "short"
            elif zz[i] <= -self.z:
                side = "long"
            else:
                i += 1
                continue
            e = c[i]
            risk = self.k * a[i]
            stop = e + risk if side == "short" else e - risk
            end = min(i + self.h, n - 1)
            exit_px, exit_j, reason = c[end], end, "time"
            for j in range(i + 1, end + 1):
                if side == "long" and l[j] <= stop:
                    exit_px, exit_j, reason = stop, j, "stop"
                    break
                if side == "short" and h[j] >= stop:
                    exit_px, exit_j, reason = stop, j, "stop"
                    break
            if side == "long":
                r = (exit_px - e) / risk
                ret = (exit_px - e) / e - cost
            else:
                r = (e - exit_px) / risk
                ret = (e - exit_px) / e - cost
            out.append(Trade(sym, side, idx[i], idx[exit_j], e, exit_px,
                             round(ret, 5), round(r, 3), exit_j - i, reason))
            i = exit_j + 1
        return out

    def walk_forward(self, frames, funding, costs, train_frac: float = 0.6):
        is_all: List[Trade] = []
        oos_all: List[Trade] = []
        for sym, df in frames.items():
            cost = costs.get(sym, S.DEFAULT_COST)
            all_t = self._trades(sym, df, funding.get(sym), cost)
            split_date = df.index[_split(len(df), train_frac)]
            is_all += [t for t in all_t if t.entry_date < split_date]
            oos_all += [t for t in all_t if t.entry_date >= split_date]
        return is_all, oos_all


class TrendCarryCandidate:
    """The trend breakout, but don't pay to fight the crowd: veto a long entry
    when funding is crowded-long (z >= +threshold) and a short entry when
    crowded-short (z <= -threshold). Implemented by masking swing's breakout
    flags with the funding gate, then reusing swing.backtest unchanged."""
    name = "trend_carry"

    def __init__(self, z_threshold: float = 1.5, allow_short: bool = True,
                 window: int = 30):
        self.z = z_threshold
        self.allow_short = allow_short
        self.window = window

    def walk_forward(self, frames, funding, costs, train_frac: float = 0.6):
        is_all: List[Trade] = []
        oos_all: List[Trade] = []
        for sym, df in frames.items():
            f = S.compute_features(df)
            z = FND.zscore(FND.align_daily(funding.get(sym), f.index), self.window)
            zf = z.reindex(f.index)
            # Mask breakouts that fight the carry. NaN z -> allow (no veto).
            long_ok = ~(zf >= self.z).fillna(False)
            short_ok = ~(zf <= -self.z).fillna(False)
            f["long_brk"] = f["long_brk"] & long_ok.values
            f["short_brk"] = f["short_brk"] & short_ok.values
            n = len(f)
            split = _split(n, train_frac)
            k_long = S.calibrate_stop_k(f, "long", max_i=split)
            k_short = S.calibrate_stop_k(f, "short", max_i=split)
            cost = costs.get(sym, S.DEFAULT_COST)
            is_t = S.backtest(f.iloc[:split], k_long, k_short, cost=cost,
                              allow_short=self.allow_short)
            oos_t = S.backtest(f.iloc[split:], k_long, k_short, cost=cost,
                               allow_short=self.allow_short)
            is_all += [to_trade(sym, t) for t in is_t]
            oos_all += [to_trade(sym, t) for t in oos_t]
        return is_all, oos_all
