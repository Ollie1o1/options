"""The squeeze-call gate: five verdicts, one posterior, a bounded clock.

One statistic and two bands, following gate v2. The Phase-1 gate failed because
it demanded an effect size AND a significance level whose sample requirements
differed by an order of magnitude, which made the stated effect threshold
decorative — the significance arm was the only thing ever binding. Here both
bands read the same posterior at the same n, so neither can quietly become
ornamental.

The bands are deliberately asymmetric in WHICH valuation they read. GO is judged
on the conservative payoff, so authorisation is hard to earn. STOP is judged on
the central one, because a strategy killed by its own conservatism buffer is not
a finding about the strategy.

NO-GO is not STOP. STOP means the premium affirmatively eats the tail; NO-GO
means the question was not resolved inside the budget. Keeping them separate is
what stops "keep gathering" from becoming a permanent answer, which is the
failure mode the previous gate actually died of.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np

GO_POSTERIOR = 0.90          # Sidak-tightened from v2's 0.85 for two tenors
STOP_POSTERIOR = 0.10
SIGN_AGREEMENT = 0.50
MAX_EXTENSIONS = 2
CYCLES_PER_EXTENSION = 2
BASE_CYCLES = 6              # ~3 months at 2 FINRA cycles/month
MIN_CYCLES = 4               # ~9 weeks; one vol regime must not decide
HARD_STOP_CYCLES = 10        # ~5 months
MIN_COVERED_OF_FIRST_SIX = 4

GO = "GO"
STOP = "STOP"
EXTEND = "EXTEND"
NO_GO = "NO-GO"
INVALID = "INVALID"


def combine(d_draws: np.ndarray, p_draws: np.ndarray,
            f_draws: np.ndarray) -> np.ndarray:
    """E = D_hist - P_live - F_live, drawn pairwise.

    The two bootstraps are independent by construction — 2018-2026 underlying
    outcomes against 2026 forward quotes — so pairing draw b with draw b is a
    convolution, not a claim that the draws correspond to anything.
    """
    n = min(len(d_draws), len(p_draws), len(f_draws))
    if n == 0:
        return np.array([])
    return np.asarray(d_draws)[:n] - np.asarray(p_draws)[:n] - np.asarray(f_draws)[:n]


def posterior_above_zero(draws: np.ndarray) -> Optional[float]:
    """P(E > 0) under a flat prior, read straight off the bootstrap."""
    arr = np.asarray(draws, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None
    return float((arr > 0).mean())


def _agrees(posteriors: Dict[str, float]) -> bool:
    """A partner tenor whose central posterior could not be computed is not
    evidence of sign agreement — None fails the test rather than raising."""
    c = posteriors.get("central")
    return c is not None and c >= SIGN_AGREEMENT


def decide(tenor_posteriors: Dict[int, Dict[str, float]], n_cycles: int,
           covered_of_first_six: int, match_valid: bool,
           extensions_used: int = 0) -> str:
    """The committed verdict. Validity is checked before anything else."""
    if not match_valid:
        return INVALID
    if n_cycles >= MIN_CYCLES and covered_of_first_six < MIN_COVERED_OF_FIRST_SIX:
        return INVALID
    if n_cycles < MIN_CYCLES:
        return EXTEND

    tenors = [t for t in tenor_posteriors if
              tenor_posteriors[t].get("conservative") is not None]

    if len(tenors) >= 2:
        for tenor in tenors:
            if tenor_posteriors[tenor]["conservative"] < GO_POSTERIOR:
                continue
            others = [t for t in tenors if t != tenor]
            if any(_agrees(tenor_posteriors[o]) for o in others):
                return GO

        centrals = [tenor_posteriors[t].get("central") for t in tenors]
        if centrals and all(c is not None and c <= STOP_POSTERIOR
                            for c in centrals):
            return STOP

    if n_cycles >= HARD_STOP_CYCLES or extensions_used >= MAX_EXTENSIONS:
        return NO_GO
    return EXTEND
