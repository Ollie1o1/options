"""The harness's own truth test.

Replaying a position under the policy it was ACTUALLY traded with must
reproduce its recorded P&L. If it cannot, every alternative policy's score is
measured against a baseline the harness does not understand, and the run is
worthless.

MANDATORY CORRECTION (Task 6, 2026-09-22): a single 10% abort threshold is not
achievable on Corpus A and the difference is a property of the corpus, not a
harness bug. Measured on 6,000 real Corpus A Bull Put positions before this
task was written: reproduction is **41.6%** under the recorder's own policy
with imputed costs (`CostModel.cross()`), and **77.9%** under pure mid costs
comparing in the recorder's own premium-fraction units. Failures by recorded
exit reason: time_exit 18.4%, take_profit 20.2%, stop_loss 46.1%. Three
corpus properties explain this, none of them a bug here: `entry_price`
equals the entry-day mark mid in only 1 of 1,432 cases (it is a scan-time
price; marks are a separate daily snapshot); `exit_price` equals the
exit-day mark mid in 89.6% of cases, so roughly 1 in 10 exits are priced off
something the marks do not contain; and many positions whose marks breach
2.0x credit are recorded as `time_exit` rather than `stop_loss`, so the live
recorder's stop ran on a different cadence than the daily mark mid this
harness replays against.

Corpus B carries real per-leg quotes and a ledger-recorded P&L and keeps the
strict 10% bar. Corpus A callers must pass a looser `max_fail_rate` (0.30 is
the measured regression guard: comfortably below the 77.9% mid-cost
reproduction rate, so it still catches a harness regression, without
pretending 90% reproduction was ever reachable on this corpus). This is why
`max_fail_rate` is a parameter of `calibration_report`, not a module
constant baked into the pass/fail logic — do not collapse it back to one
constant; the two corpora are not comparable and never will be while Corpus
A's exit and entry prices are sourced the way they are today.

The paired comparison in later tasks does not read `actual_pnl_frac` at all
(both arms replay the same path, so a basis error cancels in the
difference), so a raw fail rate alone can hide a biased failure mode: if the
irreproducible slice correlates with outcome rather than being random —
stop_loss fails at 46.1%, more than twice the other reasons above — the
paired difference could still be biased even though the aggregate rate looks
acceptable. `CalibrationReport.by_reason` exists so that skew is visible
rather than silently absorbed into one number. `PricePath` does not currently
carry the recorded exit reason, only `strategy`, so this breakdown groups by
`path.strategy` for now; a future task should thread the recorded exit
reason through `paths.py` so this becomes exit-reason-level, matching the
measurement above.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Sequence, Tuple

from src.policy_lab.costs import CostModel
from src.policy_lab.policies import ExitPolicy
from src.policy_lab.replay import replay
from src.policy_lab.types import PricePath

TOL_ABS = 0.02   # 2 percentage points of capital at risk
TOL_REL = 0.05   # or 5% relative, whichever is looser
MAX_FAIL_RATE = 0.10  # strict bar; Corpus A callers must pass a looser value


class CorpusUnusable(RuntimeError):
    """Raised when a corpus cannot reproduce its own recorded outcomes."""


@dataclass(frozen=True)
class CalibrationReport:
    checked: int
    passed: int
    failed: int
    by_reason: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    """label -> (checked, passed), grouped by `path.strategy` (see module
    docstring) until the recorded exit reason is threaded through `paths.py`.
    """

    @property
    def fail_rate(self) -> float:
        return self.failed / self.checked if self.checked else 1.0


def reproduces(path: PricePath, policy: ExitPolicy, costs: CostModel,
               tol_abs: float = TOL_ABS, tol_rel: float = TOL_REL) -> bool:
    """Does replaying `path` under its live policy match what was recorded?"""
    try:
        got = replay(path, policy, costs).pnl_frac_car
    except ValueError:
        return False
    want = path.actual_pnl_frac
    band = max(tol_abs, abs(want) * tol_rel)
    return abs(got - want) <= band


def calibration_report(paths: Sequence[PricePath], policy: ExitPolicy,
                       costs: CostModel,
                       max_fail_rate: float = MAX_FAIL_RATE) -> CalibrationReport:
    """Check every path, refusing the corpus if too many disagree.

    `max_fail_rate` is corpus-specific — see the module docstring. Corpus B
    keeps the strict default; Corpus A callers must pass a looser bound
    measured against the real corpus, not the aspirational 10%.
    """
    if not paths:
        raise CorpusUnusable("no paths to calibrate against")

    reason_checked: Dict[str, int] = {}
    reason_passed: Dict[str, int] = {}
    passed = 0
    for p in paths:
        ok = reproduces(p, policy, costs)
        if ok:
            passed += 1
        label = p.strategy
        reason_checked[label] = reason_checked.get(label, 0) + 1
        reason_passed[label] = reason_passed.get(label, 0) + (1 if ok else 0)

    by_reason = {
        label: (reason_checked[label], reason_passed.get(label, 0))
        for label in reason_checked
    }

    rep = CalibrationReport(checked=len(paths), passed=passed,
                            failed=len(paths) - passed, by_reason=by_reason)
    if rep.fail_rate > max_fail_rate:
        raise CorpusUnusable(
            f"{rep.failed}/{rep.checked} positions ({100 * rep.fail_rate:.1f}%) "
            f"do not reproduce their recorded P&L under the live policy; "
            f"the limit is {100 * max_fail_rate:.0f}%. The corpus, the loader "
            f"or the assumed live policy is wrong — do not sweep knobs on it.")
    return rep
