"""Pre-registration, enforced by a hash rather than by good intentions.

Four columns times three horizons times two benchmarks is twenty-four ways to
find a result, and this repo has already paid for that lesson once — the
pre-registered ranker test exists because a board that had been ranked by
`quality_score` all along looked fine until someone measured it.

The mechanism: the hypotheses live in code, `write()` renders them to
reports/CATALYST_PREREG.md, and the runner refuses to emit a report unless the
file on disk hashes to what the code declares. Changing a hypothesis changes
the hash, which changes the committed file, which is visible in git history.
You cannot quietly add H5 after seeing the data.

A confidence interval containing zero is reported as NO EVIDENCE. It is never
re-sliced until it does not.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from src.paths import repo_path

DEFAULT_PATH = repo_path(os.path.join("reports", "CATALYST_PREREG.md"))


@dataclass(frozen=True)
class Hypothesis:
    key: str
    statement: str
    horizon_months: int
    primary: bool = False
    exploratory: bool = False


HYPOTHESES: Tuple[Hypothesis, ...] = (
    Hypothesis(
        key="H1",
        statement=("Rows flagged FUNDED THROUGH outperform rows flagged RAISE "
                   "BEFORE over the forward window, XBI-relative."),
        horizon_months=6,
        primary=True,
    ),
    Hypothesis(
        key="H2",
        statement=("Trials whose primary endpoint was amended underperform "
                   "trials with no endpoint amendment."),
        horizon_months=6,
        exploratory=True,
    ),
    Hypothesis(
        key="H3",
        statement="Phase 3 rows outperform Phase 2 rows.",
        horizon_months=6,
        exploratory=True,
    ),
    Hypothesis(
        key="H4",
        statement=("The options-implied move is biased relative to the "
                   "realised move over the event window."),
        horizon_months=3,
        exploratory=True,
    ),
)

_PREAMBLE = """# Catalyst Backtest — Pre-Registration v3

The runner refuses to emit a report unless this file's SHA-256 matches the
hypotheses declared in `src/catalyst/backtest/prereg.py`.

Vintages: quarter-starts 2023-01-01 through 2025-10-01 (12).
Benchmark: absolute and XBI-relative.
Universe: catalyst-calendar rows, $50M-$10B, sponsor resolving to a live ticker.

A confidence interval containing zero is reported as NO EVIDENCE, and is never
re-sliced until it does not.

## v2 supersedes v1 — the ESTIMATOR changed, the hypotheses did not

**v1 (frozen 2026-08-25, run 2026-08-26) used a bootstrap that resampled ROWS.
That was wrong, and this version corrects it.** The hypotheses below are
unchanged, word for word; only the interval around them is.

`outcomes.outcomes_for(ticker, vintage, today, prices, bench)` never receives
an `nct_id`. The forward return is therefore a property of the TICKER and the
VINTAGE alone, so every trial on one ticker at one vintage appended a
BYTE-IDENTICAL value to its arm. Measured 2026-08-27 from the point-in-time
cache: 832 trials resolve to 270 distinct tickers, mean 3.08 trials each,
VNDA alone 17. Those copies are not independent evidence, and resampling rows
counted them as if they were.

**Estimator, v2:** percentile bootstrap resampling TICKERS with replacement,
2,000 iterations, seeded. A ticker appearing in both arms is drawn once and
contributes to both, preserving the within-ticker correlation. UNDERPOWERED is
decided on the CLUSTER count, not the row count.

**This is a RE-ANALYSIS of the same observations, not an independent
replication.** It cannot confirm v1 and must never be reported as a second
study that agreed. Widening an interval that already contained zero leaves it
containing zero, so v1's NO EVIDENCE verdicts are expected to stand; what
changes is any claim that rests on the WIDTH of those intervals — above all
the claim that large effects were ruled out.

v1 remains in git history. Nothing about it is deleted or amended in place.

## v3 supersedes v2 — H2's FEATURE was contaminated by lookahead

**H2 as run in v1 and v2 did not test what it claims, and both of its results
are withdrawn.** The verdict was NO EVIDENCE either way, so nothing false was
published; but a test that reads the future is not evidence of absence either.

`design.amendments_for(nct_id)` fetched the amendment history LIVE and
`parse_history` counted every change ever recorded, with no `as_of` filter.
An endpoint amended in 2025 therefore marked a row "amended" at the
2023-01-01 vintage. Every other feature on the panel — trial state, cash
runway, phase — was already reconstructed point-in-time through `pit.py`;
this one silently was not, and it is the only feature H2 is about.

**H2, v3:** `amended` is TRUE when the trial had two or more PROTOCOL
outcome-measure edits **dated on or before the vintage**, counted from the
cached version list via `pit.amendments_as_of`. A trial whose first version
postdates the vintage is `available=False` and enters neither arm, exactly as
an unknown funded-through does.

The outcome-edit definition is a RESTRICTION of the live statistic, not a new
one: counting versions whose `moduleLabels` contain "Outcome Measures"
reproduced the live `outcomesUpdateCount` on **12 of 12** cached trials
checked 2026-08-27. "Outcome Measures (Results)" is excluded — posting results
is not amending an endpoint — and the live API excludes it too, verified on a
trial carrying three of them.

**H1 and H3 are unchanged in definition.** They are re-run only because the
panel is rebuilt as one object; their v2 results should reappear up to the
data drift noted below.

**Known reproducibility limit, stated not fixed:** the panel is rebuilt from
live ClinicalTrials.gov on every run, so two runs are not guaranteed identical
inputs. Between v1 and v2 this moved H3's difference from -0.039 to -0.068
with no estimator change. Until the sweep is pinned to the cache, small
between-run differences in the MEANS are data drift, not findings.
"""


def render() -> str:
    lines = [_PREAMBLE]
    for h in HYPOTHESES:
        label = "PRIMARY" if h.primary else "EXPLORATORY"
        lines.append(f"## {h.key} ({label}, {h.horizon_months}-month horizon)\n")
        lines.append(f"{h.statement}\n")
    return "\n".join(lines)


def expected_hash() -> str:
    return hashlib.sha256(render().encode("utf-8")).hexdigest()


def file_hash(path: str) -> Optional[str]:
    """SHA-256 of the file, or None if it is not there. None is not ''."""
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return None


def write(path: str = DEFAULT_PATH) -> str:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        f.write(render())
    return expected_hash()


def verify(path: str = DEFAULT_PATH) -> bool:
    return file_hash(path) == expected_hash()
