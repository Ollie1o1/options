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
LATE_SHORT_RET5D = -10.0   # % 5-day return
RVOL_HOT = 1.5
SETUP_MIN_POINTS = 4
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

    if dtc is not None and dtc >= DTC_STRONG:
        points += 2
        ev.append(f"{dtc:.1f} days to cover (crowded exit)")
    elif dtc is not None and dtc >= DTC_MODERATE:
        points += 1
        ev.append(f"{dtc:.1f} days to cover")

    if trend == "rising":
        points += 1
        ev.append("short interest rising month-over-month (shorts adding)")

    if skew is not None and skew <= SKEW_CALL_BID:
        points += 1
        ev.append(f"25Δ skew {skew * 100.0:+.1f}vp call-skewed (upside is bid)")

    if ret_5d is not None and ret_5d <= LATE_SHORT_RET5D:
        points += 1
        ev.append(f"5d return {ret_5d:+.1f}% with heavy SI (late shorts pressing)")

    if rvol is not None and rvol > RVOL_HOT:
        points += 1
        ev.append(f"RVOL {rvol:.1f}x (volume confirming)")

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


def assess_squeeze_row(row) -> SqueezeSetup:
    """Adapter: grade from a scan DataFrame row (or any mapping-like)."""
    get = row.get if hasattr(row, "get") else lambda k, d=None: d
    return assess_squeeze({
        "short_interest": get("short_interest"),
        "short_interest_dtc": get("short_interest_dtc"),
        "short_interest_trend": get("short_interest_trend"),
        "iv_skew": get("iv_skew"),
        "ret_5d": get("ret_5d"),
        "rvol": get("rvol"),
        "gex_flip_price": get("gex_flip_price"),
        "spot": get("spot", get("underlying_price")),
    })
