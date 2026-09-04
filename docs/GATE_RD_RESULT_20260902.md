# Gate-vs-Refused Regression Discontinuity — Result (2026-09-02)

Task B of the 2026-09-02 work order: does the `candidate_verdict` friction
gate (round-trip friction > 25% → refused) add value? One look, run exactly
as pre-registered in `docs/PREREG_GATE_RD_20260902.md`, via
`scripts/gate_rd_test.py`. Reporting only — no gate, scoring, or scan-path
changes.

## Verdict

**NULL.** `t = 0.18` on the primary design (5-day horizon, ±0.05 bandwidth),
nowhere near Harvey's hurdle (`|t| ≥ 3.0`) in either direction. The friction
gate's 25% cutoff does **not** measurably separate 5-day-forward-return
quality between candidates just above it (refused) and just below it (would
pass). This is a finding about the friction gate **specifically** — it says
nothing about `negative_ev` (the largest refusal category, not tested this
round), `condor_universe`, or the gate system as a whole, per the prereg's
own scope limit (§10).

Recorded per the pre-registered decision rule (§7) as **non-predictive
complexity**: the gate still does something arithmetically real (it caps how
much of a spread's own reward gets handed back to the market on entry), but
that cap, at 25%, does not measurably predict which side of it does better
five days out.

## The primary number

| | value |
|---|---|
| ITT (refused − passed, local-linear intercept difference) | +0.0209 |
| 95% CI (cluster bootstrap, 4,000 resamples) | [−0.2124, +0.2380] |
| t-statistic | 0.18 |
| n candidates | 13,599 (7,364 below cutoff, 6,235 above) |
| n symbol-day clusters | 292 below, 299 above |
| distinct symbols | 78 below, 82 above |

The point estimate is essentially zero and the interval is wide relative to
it — a return of roughly ±0.22 on net premium at the 95% level, which is
large in absolute terms for a 5-day window. This is a genuinely underpowered
*point estimate* even though the design cleared its pre-registered 30-cluster
eligibility floor comfortably: clustering by symbol-day (not by contract)
costs real precision, which is exactly the point of doing it that way rather
than the naive, overcounted alternative.

## Secondary and robustness

- **10-day horizon (±0.05 bandwidth):** ITT = −0.1832, CI [−0.8719, +0.4115]
  — also NULL, and the CI is wider still (fewer clusters: 70 below / 66
  above at registration). The point estimate flips sign relative to the
  5-day result, which is expected noise around a true null, not a
  contradiction — neither interval excludes zero or the other's point
  estimate.
- **±0.10 bandwidth robustness (5-day horizon):** ITT = +0.0346, CI
  [−0.1508, +0.2313] — consistent with the primary result: small positive
  point estimate, wide interval containing zero.

Neither carries decision authority over the primary result, per §7; both are
reported because they agree with it, not selectively.

## Guards

- **Density (manipulation check):** 7,364 below / 6,235 above, ratio 1.18.
  No sharp jump in candidate volume right at the cutoff — no sign the
  running variable is being gamed or mismeasured at the threshold.
- **Covariate smoothness:** `abs_delta` jump = +0.0093, CI [−0.0025,
  +0.0214] (contains zero, though the lower bound sits close to it); `dte`
  jump = −1.2233, CI [−3.1248, +0.6541] (contains zero comfortably). Neither
  raises a validity concern for the primary result, though `abs_delta`'s
  proximity to the boundary is worth a note if this design is ever rerun
  with a narrower bandwidth.
- **Negative control:** shuffling outcomes within each symbol-day cell gives
  ITT = +0.0145, CI [−0.1685, +0.1984] — small and centered near zero, as
  expected when there is no real effect to lose by shuffling.
- **Sign consistency:** first half of the window ITT = −0.1377, second half
  = +0.3341. The sign flips between halves, which is consistent with noise
  around a true null (a real, stable effect would be expected to hold sign
  across both halves) rather than evidence of instability in the design
  itself.

None of the four guards surfaces a concern about the primary NULL verdict.

## Stratified matching (secondary design)

1,056 matched pairs (refused candidate ↔ nearest eligible passed candidate
on the same symbol-day, within tolerance on DTE/delta/relative spread).
Mean(refused − matched passed) = +0.0233, 95% CI [−0.0582, +0.1042] — also
contains zero, agreeing with the primary RD result. As registered, this
design's identification is weaker (this repo's gates are deterministic
functions of the matched features, so matching does not fully remove the
confound) and it carries no decision authority; it is reported here only as
a robustness cross-check, and it does not overturn or add to the primary
finding.

## What this does not show

- **Scope, per the prereg's §10:** `negative_ev` (63,359 refusals — the
  *largest* category, larger than `friction`'s 58,382) was not tested this
  round; `decide_verdict` is not a single continuous threshold and needs its
  own registration. `condor_universe` (306 refusals) is categorical, not
  continuous — no RD applies. `top_quintile` has never fired. A NULL result
  on `friction` says nothing about any of these.
- **Only 5-day and 10-day horizons are measured** — this repo's own prior
  finding (0% coverage at ≥21 days across the marked-candidate population)
  means no longer horizon can be tested with this data.
- **The outcome is a raw forward mark-to-market return**, not a managed exit
  outcome — deliberately, so passed and refused candidates are compared on
  the same symmetric basis (neither benefits from active management). This
  says whether the friction level predicts raw quality at the margin; it
  does not say whether a managed exit would behave differently.
- Nothing here surfaced a limitation the prereg did not already anticipate
  — no new deviation to record beyond what §3–§5 of the prereg already
  state (the running-variable recomputation, the population restriction to
  multi-leg structures, the fixed non-outcome-derived bandwidth).

## Next step

The friction gate at its current 25% threshold is documented as
non-predictive complexity for the 5-day forward-return margin — a valid,
recorded finding per the work order, not a call to change the threshold or
remove the gate (it still serves as a hard arithmetic floor against handing
back excessive reward to cross a spread, independent of whether that floor
predicts anything). Per the prereg's decision rule, this specific
registration is closed: a materially different question (a different
refusal reason, a different outcome horizon, or a larger data window) needs
its own fresh pre-registration, not a re-run of this one.
