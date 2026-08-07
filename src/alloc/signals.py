"""Per-symbol signal history, computed strictly from the past.

Everything here answers "what did this look like on date D, using only data
available on or before D". That constraint is the whole point: a signal that
peeks even one day ahead will manufacture an edge out of nothing, and it is the
single easiest way to fool a backtest.

Two families, matching the two questions worth asking of this book:

  IV RANK  Where does today's at-the-money implied vol sit inside its own
           trailing history? The canonical short-premium timing signal — sell
           when premium is rich relative to what this name usually offers. It
           has never been measured in this repo.

  TREND    Where is spot relative to its own moving average, and how far has it
           moved recently? This is the "where will the stock go" idea, expressed
           as a condition on WHEN to sell rather than as a directional bet.

Spot and ATM IV both come from the option chain itself — spot by put-call
parity, ATM IV from the strike nearest spot — so no external price feed is
needed and the signal cannot disagree with the chain it is traded against.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from src.alloc.settle import implied_spot_any


@dataclass(frozen=True)
class Snapshot:
    """What one symbol looked like on one date."""
    date: str
    spot: Optional[float]
    atm_iv: Optional[float]


def atm_iv(chain: Sequence[Dict[str, Any]], spot: float) -> Optional[float]:
    """Implied vol of the strike nearest spot, averaged across call and put."""
    if not chain or spot is None:
        return None
    best: Dict[str, Any] = {}
    for c in chain:
        iv = c.get("iv")
        if iv is None or float(iv) <= 0:
            continue
        d = abs(float(c["strike"]) - spot)
        typ = str(c["type"]).lower()
        if typ not in best or d < best[typ][0]:
            best[typ] = (d, float(iv))
    vals = [v for _d, v in best.values()]
    return float(statistics.mean(vals)) if vals else None


def snapshot(chain: Sequence[Dict[str, Any]], date: str) -> Snapshot:
    spot = implied_spot_any(chain)
    return Snapshot(date=date, spot=spot,
                    atm_iv=atm_iv(chain, spot) if spot is not None else None)


class SignalHistory:
    """Accumulates snapshots and answers questions about the past only.

    The engine feeds each day in as it walks the calendar, so by construction
    nothing later than the acting date has been observed yet. `update` must be
    called before `features` for a given date, and never for a future one.
    """

    def __init__(self, lookback: int = 52):
        self.lookback = lookback
        self._hist: Dict[str, List[Snapshot]] = {}

    def update(self, symbol: str, snap: Snapshot) -> None:
        series = self._hist.setdefault(symbol, [])
        if series and snap.date <= series[-1].date:
            return                      # never rewrite or go backwards
        series.append(snap)

    def forget(self, symbol: str) -> None:
        """Drop a symbol's history entirely.

        Called on a split. The price series either side of a 20:1 split are not
        comparable, and a trend signal that spans one reads a 95% crash that
        never happened — so the history restarts rather than being rescaled from
        a factor this data does not record.
        """
        self._hist.pop(symbol, None)

    def features(self, symbol: str) -> Dict[str, Optional[float]]:
        """Signals as of the most recent snapshot fed in for this symbol."""
        series = self._hist.get(symbol, [])
        if not series:
            return {}
        now = series[-1]
        window = series[-self.lookback:]

        out: Dict[str, Optional[float]] = {
            "spot": now.spot, "atm_iv": now.atm_iv,
            "iv_rank": None, "trend": None, "ret_4w": None,
        }

        ivs = [s.atm_iv for s in window if s.atm_iv is not None]
        if now.atm_iv is not None and len(ivs) >= 10:
            below = sum(1 for v in ivs if v < now.atm_iv)
            out["iv_rank"] = 100.0 * below / (len(ivs) - 1)

        spots = [s.spot for s in window if s.spot is not None]
        if now.spot is not None and len(spots) >= 10:
            out["trend"] = 100.0 * (now.spot / statistics.mean(spots) - 1.0)

        prior = [s for s in series[:-1] if s.spot is not None]
        if now.spot is not None and len(prior) >= 4:
            past = prior[-4].spot
            if past:
                out["ret_4w"] = 100.0 * (now.spot / past - 1.0)
        return out


def passes(features: Dict[str, Optional[float]],
           conditions: Dict[str, Any]) -> bool:
    """Whether a day's features satisfy a strategy's entry conditions.

    A condition on a feature that could not be computed FAILS. Treating an
    unknown as a pass would quietly convert a signalled strategy back into the
    unconditional one and make the two indistinguishable.
    """
    checks = (
        ("iv_rank_min", "iv_rank", lambda v, t: v >= t),
        ("iv_rank_max", "iv_rank", lambda v, t: v <= t),
        ("trend_min", "trend", lambda v, t: v >= t),
        ("trend_max", "trend", lambda v, t: v <= t),
        ("ret_4w_min", "ret_4w", lambda v, t: v >= t),
        ("ret_4w_max", "ret_4w", lambda v, t: v <= t),
    )
    for key, feature, test in checks:
        if key not in conditions:
            continue
        value = features.get(feature)
        if value is None or not test(value, float(conditions[key])):
            return False
    return True
