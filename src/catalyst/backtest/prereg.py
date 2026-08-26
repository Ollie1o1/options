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

_PREAMBLE = """# Catalyst Backtest — Pre-Registration

Written BEFORE any result was computed. The runner refuses to emit a report
unless this file's SHA-256 matches the hypotheses declared in
`src/catalyst/backtest/prereg.py`.

Vintages: quarter-starts 2023-01-01 through 2025-10-01 (12).
Benchmark: absolute and XBI-relative.
Universe: catalyst-calendar rows, $50M-$10B, sponsor resolving to a live ticker.

A confidence interval containing zero is reported as NO EVIDENCE, and is never
re-sliced until it does not.
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
