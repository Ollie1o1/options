"""Pure short-squeeze setup detector — no I/O, no scoring side effects.

Grades a ticker from fields the scan pipeline already computes. The verdict
is display-layer only: it never touches quality_score (calibration-cohort
discipline) and never suppresses picks. Sign convention for ``iv_skew``
follows the pipeline: put IV − call IV, so negative = call-skewed (upside bid).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

NONE = "NONE"
WATCH = "WATCH"
SETUP = "SETUP"

# Grading thresholds. SI fractions of float.
SI_WATCH_MIN = 0.15
SI_SETUP_MIN = 0.20
SI_HEAVY = 0.20
DTC_STRONG = 5.0
DTC_MODERATE = 2.5
SKEW_CALL_BID = -0.02      # put−call IV ≤ −2vp → upside is bid
LATE_SHORT_RET5D = -10.0   # % 5-day return — shown, not scored (measured -1.96pp)
MOMENTUM_RET5D = 10.0      # % 5-day return — scored +2 (measured +3.31pp)
RVOL_HOT = 1.5
# Rescaled with the point budget, not tuned to the metric. Dropping the three
# disproven bonuses (dtc, late-shorts, RVOL) cut the maximum scored points from
# 8 to 6 — and from 7 to 5 inside the backtest, which has no historical options
# chain and so never awards the call-skew leg. Holding SETUP at 4 made the grade
# demand SI>=20% AND +10% momentum together: 18 observations in 480,744.
# 4/7 of the old budget maps to ~3 of the new.
SETUP_MIN_POINTS = 3
WATCH_MIN_POINTS = 2


@dataclass
class SqueezeSetup:
    grade: str = NONE
    points: int = 0
    evidence: list = field(default_factory=list)
    si_pct: Optional[float] = None          # percent of float (display scale)
    days_to_cover: Optional[float] = None
    trend: Optional[str] = None


def _num(value) -> Optional[float]:
    try:
        if value is None:
            return None
        f = float(value)
        return None if math.isnan(f) else f
    except (TypeError, ValueError):
        return None


def _si_fraction(value) -> Optional[float]:
    """Short interest as a 0-1 fraction; tolerates 0-100 percent inputs."""
    f = _num(value)
    if f is None or f <= 0:
        return None
    return f / 100.0 if f > 1.0 else f


def assess_squeeze(fields: dict) -> SqueezeSetup:
    """Grade a short-squeeze setup: SETUP / WATCH / NONE + evidence lines."""
    si = _si_fraction(fields.get("short_interest"))
    dtc = _num(fields.get("short_interest_dtc"))
    trend = fields.get("short_interest_trend")
    trend = str(trend) if trend and not (isinstance(trend, float) and math.isnan(trend)) else None
    skew = _num(fields.get("iv_skew"))
    ret_5d = _num(fields.get("ret_5d"))
    rvol = _num(fields.get("rvol"))
    gex_flip = _num(fields.get("gex_flip_price"))
    spot = _num(fields.get("spot"))

    setup = SqueezeSetup(
        si_pct=round(si * 100.0, 2) if si is not None else None,
        days_to_cover=dtc,
        trend=trend,
    )
    if si is None or si < SI_WATCH_MIN:
        return setup

    points = 0
    ev = setup.evidence

    if si >= SI_HEAVY:
        points += 2
        ev.append(f"{si * 100.0:.1f}% of float short (heavy)")
    else:
        points += 1
        ev.append(f"{si * 100.0:.1f}% of float short (elevated)")

    # Days-to-cover: shown, NOT scored. It was the grader's largest bonus (+2)
    # and the backtest measured it at -2.38pp asymmetry [-4.82, -0.75] — a CI
    # clear of zero, so this is a measured harm, not noise.
    if dtc is not None and dtc >= DTC_STRONG:
        ev.append(f"{dtc:.1f} days to cover (crowded exit — not scored, see backtest)")
    elif dtc is not None and dtc >= DTC_MODERATE:
        ev.append(f"{dtc:.1f} days to cover (not scored)")

    if trend == "rising":
        points += 1
        ev.append("short interest rising month-over-month (shorts adding)")

    if skew is not None and skew <= SKEW_CALL_BID:
        points += 1
        ev.append(f"25Δ skew {skew * 100.0:+.1f}vp call-skewed (upside is bid)")

    # Upward momentum: the single strongest measured factor (+3.31pp
    # [+1.31, +5.77]) and previously unscored. Squeezes follow strength.
    if ret_5d is not None and ret_5d >= MOMENTUM_RET5D:
        points += 2
        ev.append(f"5d return {ret_5d:+.1f}% with heavy SI (squeeze underway)")
    elif ret_5d is not None and ret_5d <= LATE_SHORT_RET5D:
        # Kept visible because it reads as a squeeze setup and is not: the
        # "late shorts pressing" rule measured -1.96pp. Shown, never scored.
        ev.append(f"5d return {ret_5d:+.1f}% (weakness — not scored, see backtest)")

    # RVOL: shown, not scored (-1.39pp, CI spans zero — no evidence it helps).
    if rvol is not None and rvol > RVOL_HOT:
        ev.append(f"RVOL {rvol:.1f}x (volume active — not scored)")

    # Dealer-gamma context: reported, not scored (v1).
    if gex_flip is not None and spot is not None and spot > 0:
        rel = (spot - gex_flip) / spot * 100.0
        side = "above" if rel >= 0 else "below"
        note = ("dealers long gamma (dampening)" if rel >= 0
                else "dealers short gamma (moves amplify)")
        ev.append(f"spot {abs(rel):.0f}% {side} gamma flip ${gex_flip:,.0f} — {note}")

    setup.points = points
    if si >= SI_SETUP_MIN and points >= SETUP_MIN_POINTS:
        setup.grade = SETUP
    elif points >= WATCH_MIN_POINTS:
        setup.grade = WATCH
    return setup


def _ret_5d_as_percent(value) -> Optional[float]:
    """Pipeline ret_5d is a FRACTION; assess_squeeze thresholds are PERCENT.

    ``data_fetching.calculate_momentum_indicators`` returns
    ``close[-1]/close[-6] - 1.0``, so a +12% week arrives as 0.12. Comparing
    that against ±10.0 can never be true — which is precisely why the old
    "late shorts" rule was measured dead in the backtest. Converting here keeps
    assess_squeeze's percent contract intact for direct callers and tests.
    """
    f = _num(value)
    return None if f is None else f * 100.0


def assess_squeeze_row(row) -> SqueezeSetup:
    """Adapter: grade from a scan DataFrame row (or any mapping-like)."""
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    return assess_squeeze({
        "short_interest": get("short_interest"),
        "short_interest_dtc": get("short_interest_dtc"),
        "short_interest_trend": get("short_interest_trend"),
        "iv_skew": get("iv_skew"),
        "ret_5d": _ret_5d_as_percent(get("ret_5d")),
        "rvol": get("rvol"),
        "gex_flip_price": get("gex_flip_price"),
        "spot": get("spot", get("underlying_price")),
    })
