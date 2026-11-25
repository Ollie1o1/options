"""Test O1 leg rule — one implementation shared by the backtest replay and the live scanner.

`scripts/backtest_odx_wide_60_120.py::run_side` is the estimator behind the
`etf_wide_spread` LEAD verdict (docs/RESULT_ODX_WIDE_60_120_20261124.md). Its
geometry:

  expiration  nearest 90 DTE inside [60, 120]
  short leg   nearest -0.25 delta (puts) / +0.25 (calls), within +/-0.08, two-sided quote
  long leg    strike nearest 5% of spot further OTM, two-sided quote
  admit       0 < crossed credit < width, and round-trip spread <= 25% of crossed credit

The live scanner used to place the short leg at a fixed 6% OTM. That equals
-0.25 delta only when 90-day IV is near 20%: on a 13%-IV fund it sells about
-0.16 delta, on a 40%-IV fund about -0.34 — a different strategy from the one
that was tested. `select_legs` is now the only copy of the rule;
tests/radar/test_wide_spread_rule_parity.py replays `run_side` on a synthetic
chain and asserts every entry picks the same expiration and strikes.

Yahoo chains carry no greeks, so the live path computes each strike's delta
from the IV implied by its mid (Black-Scholes, European, r = 3-month T-bill).
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

TARGET_DTE = 90.0
DTE_LO = 60
DTE_HI = 120
TARGET_ABS_DELTA = 0.25
DELTA_TOL = 0.08
WING_PCT = 0.05
MAX_FRICTION_TO_CREDIT = 0.25

# Reasons match the estimator's `skipped` keys so live warnings read like the backtest log.
REASONS = ("no_expiry", "no_short_delta", "no_long_leg", "zero_width", "non_pos_credit", "friction_gt_25pct")


@dataclass(frozen=True)
class LegQuote:
    strike: float
    bid: Optional[float]
    ask: Optional[float]
    delta: Optional[float]
    iv: Optional[float] = None
    open_interest: Optional[float] = None
    volume: Optional[float] = None

    @property
    def two_sided(self) -> bool:
        return (self.bid is not None and self.ask is not None
                and self.bid > 0 and self.ask >= self.bid)

    @property
    def mid(self) -> float:
        return 0.5 * (float(self.bid or 0.0) + float(self.ask or 0.0))


@dataclass
class WideSpreadLegs:
    side: str                 # "bull_put" | "bear_call"
    expiration: str
    dte: float
    spot: float
    short: LegQuote
    long: LegQuote
    width: float
    credit_cross: float
    credit_mid: float
    round_trip_spread: float
    friction_ratio: float     # round-trip spread / crossed credit

    @property
    def width_pct(self) -> float:
        return self.width / self.spot if self.spot > 0 else float("nan")

    @property
    def net_delta(self) -> Optional[float]:
        """Position delta per share of the credit spread (short leg sold, long bought)."""
        if self.short.delta is None or self.long.delta is None:
            return None
        return -float(self.short.delta) + float(self.long.delta)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["width_pct"] = self.width_pct
        d["net_delta"] = self.net_delta
        return d


def choose_expiration(exp_dtes: Mapping[str, float], *, lo: float = DTE_LO, hi: float = DTE_HI,
                      target: float = TARGET_DTE) -> Optional[str]:
    """Nearest `target` DTE inside [lo, hi]; ties break to the earlier expiration string."""
    inside = {e: float(d) for e, d in exp_dtes.items() if lo <= float(d) <= hi}
    if not inside:
        return None
    return min(inside, key=lambda e: (abs(inside[e] - target), e))


def select_legs(quotes: Sequence[LegQuote], *, spot: float, side: str, expiration: str, dte: float,
                max_friction_to_credit: float = MAX_FRICTION_TO_CREDIT,
                ) -> Tuple[Optional[WideSpreadLegs], str]:
    """Apply the O1 leg rule to one expiration's quotes (puts for bull_put, calls for bear_call).

    Returns (legs, "ok") or (None, reason) with reason in REASONS.
    """
    is_put = str(side).lower() != "bear_call"
    target = -TARGET_ABS_DELTA if is_put else TARGET_ABS_DELTA
    valid_short = [q for q in quotes
                   if q.delta is not None and abs(float(q.delta) - target) <= DELTA_TOL and q.two_sided]
    if not valid_short:
        return None, "no_short_delta"
    short = min(valid_short, key=lambda q: abs(float(q.delta or 0.0) - target))
    k_short = float(short.strike)
    if is_put:
        target_long = k_short - WING_PCT * spot
        valid_long = [q for q in quotes if float(q.strike) < k_short and q.two_sided]
    else:
        target_long = k_short + WING_PCT * spot
        valid_long = [q for q in quotes if float(q.strike) > k_short and q.two_sided]
    if not valid_long:
        return None, "no_long_leg"
    long = min(valid_long, key=lambda q: abs(float(q.strike) - target_long))
    width = abs(k_short - float(long.strike))
    if width <= 0:
        return None, "zero_width"
    s_bid, s_ask = float(short.bid or 0.0), float(short.ask or 0.0)
    l_bid, l_ask = float(long.bid or 0.0), float(long.ask or 0.0)
    c_cross = s_bid - l_ask
    if c_cross <= 0 or c_cross >= width:
        return None, "non_pos_credit"
    rt = (s_ask - s_bid) + (l_ask - l_bid)
    fric = rt / c_cross
    if fric > max_friction_to_credit:
        return None, "friction_gt_25pct"
    return WideSpreadLegs(
        side="bull_put" if is_put else "bear_call",
        expiration=str(expiration), dte=float(dte), spot=float(spot),
        short=short, long=long, width=width,
        credit_cross=c_cross, credit_mid=short.mid - long.mid,
        round_trip_spread=rt, friction_ratio=fric,
    ), "ok"


# ── Live inputs: Yahoo rows → LegQuote with computed delta ───────────────────

def risk_free_rate(cache_path: str = "data/macro_rates_cache.json", default: float = 0.04) -> Tuple[float, str]:
    """(rate, source label): 3-month T-bill from the macro-rates cache, else `default`."""
    try:
        from src.paths import repo_path

        with open(repo_path(cache_path), encoding="utf-8") as fh:
            snap = json.load(fh)
        v = snap.get("dgs3mo")
        if v is not None and math.isfinite(float(v)) and 0 <= float(v) < 20:
            as_of = (snap.get("as_of") or {}).get("DGS3MO") or "?"
            return float(v) / 100.0, f"3m T-bill {float(v):.2f}% ({snap.get('source', 'cache')}, {as_of})"
    except (OSError, ValueError, TypeError):
        pass
    return default, f"default {default:.2%} (rates cache unavailable)"


def _f(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def quotes_from_rows(rows: Iterable[Mapping[str, Any]], *, spot: float, dte: float, opt_type: str,
                     r: float, q: float = 0.0) -> List[LegQuote]:
    """Yahoo option rows → LegQuotes. Delta from the mid-implied vol; None when no IV solves."""
    from src.data_quality import implied_vol_from_price
    from src.utils import bs_delta

    T = max(float(dte), 0.0) / 365.0
    out: List[LegQuote] = []
    for row in rows:
        k = _f(row.get("strike"))
        b, a = _f(row.get("bid")), _f(row.get("ask"))
        if k is None or k <= 0:
            continue
        delta: Optional[float] = None
        iv: Optional[float] = None
        if b is not None and a is not None and b > 0 and a >= b and T > 0 and spot > 0:
            iv = implied_vol_from_price(opt_type, spot, k, T, r, 0.5 * (a + b), q)
            if iv is not None:
                delta = float(bs_delta(opt_type, spot, k, T, r, iv, q))
        out.append(LegQuote(strike=k, bid=b, ask=a, delta=delta, iv=iv,
                            open_interest=_f(row.get("openInterest")), volume=_f(row.get("volume"))))
    out.sort(key=lambda x: x.strike)
    return out


__all__ = [
    "TARGET_DTE", "DTE_LO", "DTE_HI", "TARGET_ABS_DELTA", "DELTA_TOL", "WING_PCT",
    "MAX_FRICTION_TO_CREDIT", "REASONS", "LegQuote", "WideSpreadLegs", "choose_expiration",
    "select_legs", "risk_free_rate", "quotes_from_rows",
]
