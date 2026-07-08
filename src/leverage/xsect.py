"""Cross-sectional momentum candidate for the leverage universe.

Each rebalance (every `hold` days) rank symbols by trailing `lookback`-day
return; long the strongest, short the weakest, hold to the next rebalance. A
distinct edge from outright trend-following and lower directional beta. Fixed
lookback/hold (not fit). Conforms to the candidates.py interface; `funding` is
ignored. r_multiple is defined as the net return itself (no ATR stop here), so
downstream expectancy-in-R equals expectancy-in-return for this candidate.
"""
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

from .candidates import Trade


class CrossSectionalCandidate:
    name = "xsect_momentum"

    def __init__(self, lookback: int = 30, hold: int = 10):
        self.lookback = lookback
        self.hold = hold

    def _closes(self, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        cols = {sym: df["close"] for sym, df in frames.items()}
        return pd.DataFrame(cols).dropna()

    def _legs(self, frames, costs) -> List[Trade]:
        px = self._closes(frames)
        if len(px.columns) < 2 or len(px) <= self.lookback + self.hold:
            return []
        idx = list(px.index)
        out: List[Trade] = []
        i = self.lookback
        n = len(px)
        while i + self.hold < n:
            trailing = px.iloc[i] / px.iloc[i - self.lookback] - 1.0
            ranked = trailing.sort_values()
            weak, strong = ranked.index[0], ranked.index[-1]
            exit_i = i + self.hold
            for sym, side in ((strong, "long"), (weak, "short")):
                e = float(px.iloc[i][sym])
                xp = float(px.iloc[exit_i][sym])
                cost = costs.get(sym, 0.0013)
                if side == "long":
                    ret = (xp - e) / e - cost
                else:
                    ret = (e - xp) / e - cost
                out.append(Trade(sym, side, idx[i], idx[exit_i], e, xp,
                                 round(ret, 5), round(ret, 5), self.hold,
                                 "rebalance"))
            i = exit_i
        return out

    def walk_forward(self, frames, funding, costs,
                     train_frac: float = 0.6) -> Tuple[List[Trade], List[Trade]]:
        px = self._closes(frames)
        if len(px) == 0:
            return [], []
        split_date = px.index[int(len(px) * train_frac)]
        legs = self._legs(frames, costs)
        is_t = [t for t in legs if t.entry_date < split_date]
        oos_t = [t for t in legs if t.entry_date >= split_date]
        return is_t, oos_t
