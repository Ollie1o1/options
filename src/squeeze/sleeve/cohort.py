"""Who counts as treated, who is an eligible control, who is neither.

This is the only module with an opinion about short-interest percentiles. The
statistics layer reads the label and does not know how it was decided, so
changing the cohort never means editing a bootstrap.

The treated definition is the SLEEVE's, not the study's: top five percent by
short interest AND a five-day return of at least ten percent. The study's
asymmetry result was measured on a broader cohort, but a test run on a cohort
nobody intends to trade authorises a strategy nobody intends to trade — the same
reasoning that forces D_hist to use the sleeve's own exit ladder.

Ranking is WITHIN a settlement date. The short-interest distribution moves with
the market, so a rank pooled across dates would load the treated arm with
whichever dates happened to be crowded rather than with the names that stood out
on their own day.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

TREATED_SI_PCT = 0.05     # top 5% by si_ratio, within the date
TREATED_RET5D = 0.10      # and a +10% five-day return (a FRACTION, not percent)
CONTROL_SI_MAX = 0.50     # bottom 50% by si_ratio

TREATED = "treated"
CONTROL = "control"


def label(rows: Sequence[dict]) -> List[Optional[str]]:
    """Arm label per row, in input order. One settlement date's rows at a time."""
    n = len(rows)
    if n == 0:
        return []

    ranked = sorted(
        (i for i in range(n) if rows[i].get("si_ratio") is not None),
        key=lambda i: float(rows[i]["si_ratio"]),
    )
    m = len(ranked)
    out: List[Optional[str]] = [None] * n
    if m == 0:
        return out

    treated_from = m - max(1, int(round(m * TREATED_SI_PCT)))
    control_to = int(m * CONTROL_SI_MAX)

    for pos, i in enumerate(ranked):
        if pos >= treated_from:
            ret_5d = rows[i].get("ret_5d")
            if ret_5d is not None and float(ret_5d) >= TREATED_RET5D:
                out[i] = TREATED
        elif pos < control_to:
            # Spec 3.1: a control must carry a ret_5d value at all. ret_5d is
            # a matching covariate, and a row without one would need a
            # fabricated value to be matchable — so it joins neither arm.
            if rows[i].get("ret_5d") is not None:
                out[i] = CONTROL
    return out
