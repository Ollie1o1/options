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

import datetime as _dt
import math
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from src.alloc.settle import implied_spot_any


@dataclass(frozen=True)
class Snapshot:
    """What one symbol looked like on one date.

    The last two are SHAPE features — they describe the surface across
    expirations and across strikes on that one day, so they are computed here
    from the chain rather than accumulated from history. They default to None
    so a caller that only has a level can still build a Snapshot.
    """
    date: str
    spot: Optional[float]
    atm_iv: Optional[float]
    term_slope: Optional[float] = None
    skew_25d: Optional[float] = None


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


MIN_TERM_GAP_DAYS = 7   # closer than this is one point on the surface, not a slope
SKEW_DELTA = 0.25
SKEW_DELTA_TOL = 0.07   # a 45-delta contract does not price a 25-delta wing


def _by_expiration(chain: Sequence[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    out: Dict[str, List[Dict[str, Any]]] = {}
    for c in chain:
        exp = c.get("expiration")
        if exp:
            out.setdefault(str(exp), []).append(c)
    return out


def term_slope(chain: Sequence[Dict[str, Any]],
               spot: Optional[float]) -> Optional[float]:
    """Near-dated ATM IV minus far-dated, in the same units as ``atm_iv``.

    Positive is backwardation: the market is charging more for the next few
    weeks than for the next few months, which is what it does when stress is
    arriving. Negative is the ordinary contango of a calm surface.

    This is the reason to bother, given every LEVEL feature tested in
    ``docs/ATTRIBUTION_20260808.md`` came back flat or failed its holdout:
    ``iv_rank`` is a level against a name's own history and is coincident with
    stress, whereas the shape inverts as stress arrives.

    Bounded by the cache: DTE runs 10-67, so this is a short-dated slope
    (roughly 10d against 60d) and NOT the 1M/3M slope the literature usually
    means. That segment does invert first, but the constraint has to travel
    with any result measured on it.
    """
    if spot is None:
        return None
    # Bound to a concrete float rather than relying on the narrowing above to
    # survive the loop: `atm_iv` takes a float, and this module is inside the
    # blocking half of the mypy ratchet.
    s = float(spot)
    ivs: List[Tuple[str, float]] = []
    for exp, rows in _by_expiration(chain).items():
        iv = atm_iv(rows, s)
        if iv is not None:
            ivs.append((exp, iv))
    if len(ivs) < 2:
        return None
    ivs.sort(key=lambda p: p[0])
    (near_exp, near_iv), (far_exp, far_iv) = ivs[0], ivs[-1]
    try:
        gap = (_dt.date.fromisoformat(far_exp)
               - _dt.date.fromisoformat(near_exp)).days
    except (TypeError, ValueError):
        return None
    if gap < MIN_TERM_GAP_DAYS:
        return None
    return float(near_iv - far_iv)


def skew_25d(chain: Sequence[Dict[str, Any]]) -> Optional[float]:
    """25-delta put IV minus 25-delta call IV, on the nearest expiration.

    Selling a bull put IS selling the put wing, and how rich that wing is
    relative to the call side is the price of the thing being sold. Nothing
    measured so far captures it: ``atm_iv`` is taken at the money and
    ``iv_rank`` is a time-series rank of that same at-the-money number.

    A wing is only priced off a contract genuinely near 25 delta. Taking the
    nearest listed contract however far away it is would be the defect
    ``_nearest_delta`` had before ``DELTA_TOLERANCE`` — the quantity measured
    would not be the quantity named.
    """
    by_exp = _by_expiration(chain)
    if not by_exp:
        return None
    rows = by_exp[min(by_exp)]

    def _wing(opt_type: str) -> Optional[float]:
        best: Optional[Tuple[float, float]] = None
        for c in rows:
            if str(c.get("type", "")).lower() != opt_type:
                continue
            d, iv = c.get("delta"), c.get("iv")
            if d is None or iv is None or float(iv) <= 0:
                continue
            miss = abs(abs(float(d)) - SKEW_DELTA)
            if best is None or miss < best[0]:
                best = (miss, float(iv))
        if best is None or best[0] > SKEW_DELTA_TOL:
            return None
        return best[1]

    put_iv, call_iv = _wing("put"), _wing("call")
    if put_iv is None or call_iv is None:
        return None
    return float(put_iv - call_iv)


def snapshot(chain: Sequence[Dict[str, Any]], date: str) -> Snapshot:
    spot = implied_spot_any(chain)
    return Snapshot(date=date, spot=spot,
                    atm_iv=atm_iv(chain, spot) if spot is not None else None,
                    term_slope=term_slope(chain, spot),
                    skew_25d=skew_25d(chain))


MAX_STEP_DAYS = 10      # beyond this the two observations straddle a data hole
MIN_RV_STEPS = 10
TRADING_DAYS = 252.0
_CAL_TO_TRADING = 5.0 / 7.0

# IV velocity is measured against the most recent snapshot at least
# IV_VEL_MIN_DAYS back — wide enough that the every-other-day cadence before
# 2025 and the daily cadence after both clear it — and refused entirely beyond
# IV_VEL_MAX_DAYS, because a comparison spanning a hole fabricates an event.
IV_VEL_MIN_DAYS = 5
IV_VEL_MAX_DAYS = 21
MIN_VOV_POINTS = 10


def _iv_velocity(series: Sequence[Snapshot]) -> Optional[float]:
    """Change in ATM implied vol, in IV units per week.

    A LEVEL cannot distinguish "premium is rich and calming down" — the
    textbook short-premium entry — from "premium is rich because a crash is
    underway". ``docs/ATTRIBUTION_20260808.md`` §4f found high IV rank selects
    INTO a crash, which is exactly the failure a direction term addresses.

    Expressed per week rather than per observation so the cache's change of
    cadence in 2025 cannot register as a change in the signal. That is the same
    trap ``_realized_vol`` scales for.
    """
    if len(series) < 2:
        return None
    now = series[-1]
    if now.atm_iv is None:
        return None
    # Concrete floats, not narrowed attributes: the arithmetic below sits
    # behind a loop and a try/except, and this module blocks the mypy gate.
    now_iv = float(now.atm_iv)
    try:
        now_date = _dt.date.fromisoformat(now.date)
    except (TypeError, ValueError):
        return None
    for prior in reversed(list(series[:-1])):
        if prior.atm_iv is None:
            continue
        prior_iv = float(prior.atm_iv)
        try:
            gap = (now_date - _dt.date.fromisoformat(prior.date)).days
        except (TypeError, ValueError):
            continue
        if gap < IV_VEL_MIN_DAYS:
            continue
        if gap > IV_VEL_MAX_DAYS:
            return None            # the nearest usable observation is across a hole
        return float((now_iv - prior_iv) / gap * 7.0)
    return None


def _realized_vol(window: Sequence[Snapshot]) -> Optional[float]:
    """Annualised realized vol of the observed spot path, or None.

    Two properties this data forces, both tested:

    SPACING   The cache samples every other trading day before 2025 and daily
              after. A raw stdev of consecutive returns would read the denser
              era as lower vol purely because each step spans less time, so the
              backfill itself would look like a regime change. Each return is
              scaled by the square root of its own elapsed time.

    HOLES     Consecutive rows can straddle a 21-month gap. Annualising that as
              one return fabricates a vol explosion, so steps longer than
              MAX_STEP_DAYS are dropped rather than scaled.
    """
    # Carry (date, spot) as concrete floats rather than filtering Snapshots:
    # a comprehension filter does not narrow the element type, so the division
    # below would still be operating on Optional[float].
    pts: List[Tuple[str, float]] = [
        (s.date, float(s.spot)) for s in window
        if s.spot is not None and s.spot > 0]
    scaled: List[float] = []
    for (prev_date, prev_spot), (cur_date, cur_spot) in zip(pts, pts[1:]):
        try:
            gap = (_dt.date.fromisoformat(cur_date)
                   - _dt.date.fromisoformat(prev_date)).days
        except (TypeError, ValueError):
            continue
        if gap <= 0 or gap > MAX_STEP_DAYS:
            continue
        steps = max(gap * _CAL_TO_TRADING, 0.5)
        scaled.append(math.log(cur_spot / prev_spot) / math.sqrt(steps))
    if len(scaled) < MIN_RV_STEPS:
        return None
    return float(statistics.stdev(scaled) * math.sqrt(TRADING_DAYS))


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
            "rv": None, "iv_minus_rv": None,
            # Shape of the surface on the day, carried straight off the
            # snapshot: both are point-in-time, not accumulated.
            "term_slope": now.term_slope, "skew_25d": now.skew_25d,
            "iv_velocity": None, "vol_of_vol": None,
        }

        ivs = [s.atm_iv for s in window if s.atm_iv is not None]
        if len(ivs) >= MIN_VOV_POINTS:
            # Dispersion of the level itself. Unlike `rv` this needs no
            # time-scaling: it is the spread of a level, not of a return.
            out["vol_of_vol"] = float(statistics.stdev(ivs))
        out["iv_velocity"] = _iv_velocity(series)

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

        rv = _realized_vol(window)
        out["rv"] = rv
        if rv is not None and now.atm_iv is not None:
            # The variance risk premium as it could be seen on the day: today's
            # implied against what this name has actually been doing. Positive
            # means options are pricing more movement than has been delivered —
            # the case for selling. Negative is the case for buying.
            out["iv_minus_rv"] = float(now.atm_iv) - rv
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
        ("term_slope_min", "term_slope", lambda v, t: v >= t),
        ("term_slope_max", "term_slope", lambda v, t: v <= t),
        ("skew_25d_min", "skew_25d", lambda v, t: v >= t),
        ("skew_25d_max", "skew_25d", lambda v, t: v <= t),
        ("iv_velocity_min", "iv_velocity", lambda v, t: v >= t),
        ("iv_velocity_max", "iv_velocity", lambda v, t: v <= t),
    )
    for key, feature, test in checks:
        if key not in conditions:
            continue
        value = features.get(feature)
        if value is None or not test(value, float(conditions[key])):
            return False
    return True
