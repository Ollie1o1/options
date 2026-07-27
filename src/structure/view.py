"""Directional view with honest, measured confidence.

No new signal is invented here. The numeric composite is produced by the already
backtested src/intel verdict machinery; this module only converts it into a
direction plus a float confidence, and applies the measured bearish penalty.

src/intel/verdict.decide() returns Verdict.confidence as a STRING
(low/medium/high), which cannot be gated on arithmetically - so build_view takes
the numeric Verdict.composite instead. src/intel is not modified.
"""
from typing import List, Optional

from .types import BEARISH_BASE_RATE, BULLISH_BASE_RATE, View

# docs/OUTLOOK_FINDINGS.md measures absolute bearish calls at ~30% hit rate
# versus 66-72% bullish. Encode the asymmetry rather than pretending it away.
BEARISH_CONFIDENCE_CAP = 0.4
NEUTRAL_BELOW = 0.35


def build_view(symbol: str, composite: float,
               drivers: Optional[List[str]] = None,
               bearish_cap: float = BEARISH_CONFIDENCE_CAP,
               neutral_below: float = NEUTRAL_BELOW) -> View:
    """Map a [-1,+1] composite to a direction plus float confidence."""
    drivers = list(drivers or [])
    conf = min(1.0, abs(float(composite)))
    direction = "BULLISH" if composite > 0 else "BEARISH"

    if direction == "BEARISH":
        conf = min(conf, bearish_cap)

    if conf < neutral_below:
        return View(symbol=symbol, direction="NEUTRAL", confidence=conf,
                    drivers=drivers)
    return View(symbol=symbol, direction=direction, confidence=conf,
                drivers=drivers)


def implied_hit(view: View) -> float:
    """Expected directional hit rate this view can supply.

    implied = 0.5 + confidence * (base_rate - 0.5)

    Because the measured bearish base rate is below 0.5, more bearish
    confidence LOWERS implied hit. That is deliberate - it means only a very
    forgiving bearish debit structure can ever clear the gate.
    """
    if view.direction == "BULLISH":
        base = BULLISH_BASE_RATE
    elif view.direction == "BEARISH":
        base = BEARISH_BASE_RATE
    else:
        return 0.5
    return 0.5 + float(view.confidence) * (base - 0.5)
