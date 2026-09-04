# Pre-registration — does the friction gate add value? (regression discontinuity)

**Frozen 2026-09-02 18:36 UTC. Immutable.**

Written before any outcome (forward return, PF, win rate) for any candidate
was computed or looked at. Only design-level facts were inspected first: gate
source code, schema, the stability of the threshold, and coverage/count
diagnostics needed to fix a bandwidth and a decision rule — the same
"sample sizes were inspected, not outcomes" discipline `PREREG_SECTOR_
CONDITIONING_20260813.md` used. Every count in §6 is a coverage count, never
an outcome.

## 1. Why this, and why RD rather than a naive comparison

`data/candidates.db` (147,086 candidates as of 2026-08-27, 162,855 as of
today) records both the **33,001+ candidates the live boards passed** and the
**110,552+ they refused**, with forward price marks on a growing subset. The
usual objection to testing a selection rule on the population it selected
("a cohort chosen by rule X cannot test rule X") does not apply here, because
the refused population is a genuine, recorded control group — this is new
evidence the repo has never had for its own gates.

**Naive passed-vs-refused is confounded.** The refused set is dominated by
deep-OTM, wide, thin-liquidity structures; comparing its outcomes to the
passed set's conflates the gate's effect with everything else that differs
between "the kind of candidate that tends to fail" and "the kind that tends
to pass." **Stratified matching has a subtler failure**: this repo's gates
are *deterministic functions* of the features you would match on, so matching
on all of them matches away the treatment, and matching on some leaves the
confound in.

**Regression discontinuity is the design that survives both objections.**
Comparing candidates *just above* vs *just below* a fixed, deterministic
threshold isolates the threshold's own effect: within a narrow band the two
groups differ, on average, only in which side of the cutoff they landed on —
not in the underlying quality that put them there. This is the cleanest
causal identification available for a rule that is a hard cutoff on a
continuous, recorded variable.

## 2. Hypothesis

**H-GATE.** The friction gate's marginal decision separates genuinely worse
trades from genuinely better ones: among candidates within a narrow band of
the 25% round-trip-friction cutoff, those just above it (refused) have a
lower forward return on their own net premium than those just below it
(would have passed), net of the discontinuity.

**Estimand.** ITT = E[outcome | round_trip_pct → 0.25⁺] − E[outcome |
round_trip_pct → 0.25⁻], estimated by the difference in intercepts of two
local-linear regressions of outcome on `round_trip_pct − 0.25`, fit
separately on each side, within the registered bandwidth (§5).

**Sign convention, fixed now:** if the gate adds value, ITT < 0 (refused
candidates' forward returns are worse). ITT ≈ 0 means the gate does not
separate quality at the margin — "non-predictive complexity," a valid finding
per the work order this registration answers. ITT > 0 (refused candidates do
*better*) would mean the gate is actively wrong at the margin and is reported
as such, not reframed.

## 3. The gate targeted — LOCKED

**Source:** `src/candidate_verdict.verdict_for` (`round_trip_pct >
DEFAULT_MAX_FRICTION` → refused, reason `"friction … exceeds the 25%
ceiling"`), reached via `src/pick_ranking.gate_board`'s `friction` gate
(second in `_GATE_ORDER`, after `unquotable`). `DEFAULT_MAX_FRICTION = 0.25`
has been unchanged since the module was introduced (verified: one definition
in git history, never edited) and applies uniformly across the whole
candidates.db collection window (2026-08-19 → today).

**Why this gate and not another.** `refused_by` carries six values today:
`negative_ev` (63,359 — the largest, but `decide_verdict`'s rule is not a
single continuous cutoff and is out of scope for this registration, §9),
`friction` (58,382), `condor_universe` (306, categorical — not continuous, no
RD possible), `credit_gone` (4, too few to register), `top_quintile` (0
observed to date — the gate has never fired). **`friction` is the only
refusal reason with a stable, recorded, continuous threshold and enough mass
on both sides to support an RD design**, so it is the sole target of this
registration. §10 states this scoping explicitly as a limit, not a finding.

**Running variable is recomputed, not read from the stored column — verified
necessary and verified correct.** `candidates.round_trip_pct` is `NULL` for
**100% of `friction`-refused rows** (58,382 of 58,382) — the modes that
produce a friction refusal (`Credit Spreads`, `Iron Condor`) never persist
the flat column; the modes that persist it (`Premium Selling`, `None`) never
produce a friction refusal. This is a recording gap, not a missing-data gap:
`features_json` carries every per-leg quote (`short_bid`/`short_ask`/
`long_bid`/`long_ask`, `net_credit`, `spread_width`) `candidate_verdict.
verdict_for` needs. Recomputing `round_trip_pct` from `features_json` via
`verdict_for` was verified on 16 sampled rows (8 recorded `friction`, 8
recorded `gate_passed=1`) to reproduce the stored `refused_by`/`gate_passed`
decision exactly in every case — the recomputed value is the same quantity
the live gate used, not a proxy for it.

**Population.** Friction refusals occur **only** on multi-leg structures —
Bull Put (33,102), Bear Call (25,270), Iron Condor (10). Single legs never
approach the cutoff (measured elsewhere in this repo at 0.7–1.7% round-trip,
`candidate_verdict.py`'s own docstring) and are absent from this population
by construction, not by filtering choice. Of 86,189 multi-leg candidates,
`round_trip_pct` (recomputed where needed) is available for 86,185; 4 are
excluded as `credit_gone` (a different economic condition — the credit
vanishes once crossed, independent of the friction margin — mixing it into
this RD would conflate two different mechanisms); 0 are unpriceable.

## 4. Data

`data/candidates.db`, `candidates` joined to `candidate_marks` on
`contract_key`. Window 2026-08-19 → 2026-09-02 (14 calendar days), 110
distinct symbols across the multi-leg population. `candidate_marks.mid` for a
multi-leg `contract_key` is the structure's **net** mark (verified via
`candidate_marks.entry_price_for`/`legs_for`/`pnl_pct`: entry is priced at
the `limit` fill across all legs via `execution_truth.structure_fill`, and
`pnl_pct` derives return from the signed net entry vs. the net mark — the
same convention the real book's `pnl_pct` uses), not a single leg's price, so
it is directly comparable to the candidate's own net premium.

## 5. Design — LOCKED

* **Outcome.** Forward return on net premium at a fixed calendar horizon:
  `candidate_marks.pnl_pct(entry_signed, mark_abs)` where `entry_signed =
  candidate_marks.entry_price_for(row)` (recomputed from `features_json` the
  same way as §3, verified reachable for this population) and `mark_abs` is
  the **first** `candidate_marks` row for that `contract_key` dated at or
  after `entry_day + horizon` calendar days (earliest qualifying mark, never
  the closest one — "closest" could select a mark *before* the horizon and
  silently shorten it). No mark before `entry_day` is ever used (no
  lookahead, same rule `candidate_marks.py` already states).
* **Horizon — LOCKED, primary and secondary.** Primary **5 calendar days**,
  secondary **10 calendar days**. **21 days is not registered**: this repo's
  own prior measurement across 11,898 marked contracts found 0% forward
  coverage at ≥21 days (median span 4d, max 14d) — registering a horizon with
  no achievable coverage would be decorative, the exact trap `PREREG_RANKER_
  TEST.md`'s own history warns against (a condition that can never bind).
* **Bandwidth — LOCKED.** Primary **±0.05** round-trip-friction points around
  the 0.25 cutoff (i.e. `round_trip_pct ∈ [0.20, 0.30]`); secondary,
  robustness only, **±0.10**. Fixed now, not chosen by minimizing anything
  computed from outcomes — a data-driven (e.g. MSE-optimal) bandwidth is
  explicitly rejected here because optimizing it requires the outcome, which
  would make the bandwidth choice itself a peek.
* **Clustering — LOCKED.** One observation is one **symbol-day**
  (`(symbol, date(candidates.ts))` — the candidate's own scan/entry day),
  never one contract. Within the bandwidth, candidates are first collapsed to
  **one point per symbol-day per side of the cutoff**: the mean
  `round_trip_pct − 0.25` and the mean outcome across every candidate that
  symbol-day contributed to that side. The local-linear regressions in §2 are
  fit on these symbol-day-level means, not on raw candidate rows — this is
  the most literal reading of "one observation = one symbol-day" available,
  and it removes the row/cluster overcounting trap
  (`[[project_prereg_ranker_test]]`, the catalyst bootstrap) by construction
  rather than by a post-hoc cluster-robust correction.
* **Estimator.** Local-linear regression (uniform kernel — every point inside
  the bandwidth weighted equally, nothing outside it used), fit separately
  above and below the cutoff on the symbol-day means; ITT = intercept(above)
  − intercept(below) at `round_trip_pct = 0.25`.
* **Interval / significance.** Percentile 95% CI from a cluster bootstrap:
  resample symbol-days with replacement, separately within each side, refit
  both local-linear regressions and recompute ITT, 4,000 resamples, seeded
  (`20260902`, this document's freeze date, same convention as every other
  seeded bootstrap in this repo). `t = ITT / bootstrap_SE`, `bootstrap_SE` the
  standard deviation of the resampled ITT draws — the same clustered-t
  construction `src/alloc/report.clustered_tstat` uses for one-sample tests,
  adapted here to a two-sample intercept difference.

## 6. Coverage — measured, not a peek (counts only, no outcome values read)

Primary design (±0.05 bandwidth, 5-day horizon):

| side | candidates with a ≥5d mark | symbol-days | distinct symbols |
|---|---:|---:|---:|
| below cutoff (would pass) | 2,960 | 292 | 78 |
| above cutoff (refused) | 2,483 | 299 | 82 |

Secondary horizon (±0.05 bandwidth, 10-day horizon):

| side | candidates with a ≥10d mark | symbol-days |
|---|---:|---:|
| below cutoff | 425 | 70 |
| above cutoff | 307 | 66 |

Total candidates in the ±0.05 band regardless of mark coverage: 13,599 (7,364
below / 6,235 above), so the primary design uses 5-day-mark coverage of
**40.2% below / 39.8% above** of the band — stated up front, not discovered
after the fact. ±0.10 band (robustness): 27,197 candidates total, 591/632
symbol-days (unrestricted by mark presence; mark-coverage counts for this
band are computed at run time, not pre-registered, since it is the
robustness check and not the primary design).

**Eligibility floor — LOCKED.** A design below **30 symbol-day clusters per
side** is reported as `UNDERPOWERED`, matching this repo's existing cell-
minimum convention (`PREREG_SECTOR_CONDITIONING_20260813.md`'s ≥4-symbol,
≥100-trade floors; `src/alloc/report.MIN_N = 20`). Both primary (292/299) and
secondary-horizon (70/66) designs clear this floor; it exists for the ±0.10
robustness check and any future refit against a thinner window, not because
either registered design is expected to fail it.

## 7. Decision rule

One look, run once the primary design (§5, 5-day horizon, ±0.05 bandwidth)
has been computed — no re-running with a different bandwidth or horizon after
seeing the result. Bar is Harvey's hurdle, not `p < 0.05`
(`src/alloc/report.MIN_TSTAT = 3.0`), because of how many things have already
been tested against this data.

| outcome | condition | consequence |
|---|---|---|
| REAL (gate adds value) | `t ≤ -3.0` (ITT < 0, `\|t\| ≥ 3.0`) | gate's friction cutoff is measured to separate quality; report and consider whether the 0.25 level itself is well-placed |
| INVERTED | `t ≥ +3.0` (ITT > 0) | refused candidates measured to do *better* — reported as a defect in the gate, not softened |
| NULL | `-3.0 < t < 3.0` | CI straddles no-effect region at the hurdle; recorded as **non-predictive complexity** per the work order — a valid, valuable finding, not a failure |
| UNDERPOWERED | fewer than 30 symbol-day clusters on either side at run time | say so and stop; no verdict drawn |

There is no EXTEND state, for the same reason `PREREG_RANKER_TEST.md` refused
one: an open-ended "gather more data and re-check" is how the LC gate ran
forever. A NULL or UNDERPOWERED result here does not authorize waiting for a
bigger window and re-running this same registration — a materially larger
window (e.g. 60+ days) would need its own registration.

Secondary (10-day horizon, same ±0.05 bandwidth) and the ±0.10 bandwidth
robustness check are reported **beside** the primary result with no decision
authority of their own — they may motivate a new registration; they cannot
overturn this one.

## 8. Guards

* **Manipulation / density check.** No sharp jump in the *count* of
  candidates just below vs. just above the cutoff within a narrow band
  (`McCrary`-style, informal: candidate counts in [0.20,0.25) vs [0.25,0.30]
  should be the same order of magnitude — a large jump would suggest the
  running variable is being gamed or mismeasured right at the threshold,
  which a deterministic arithmetic gate should not exhibit). Reported, not a
  gate on running the primary result.
* **Covariate smoothness.** Mean `entry_delta` and mean DTE at entry,
  computed the same symbol-day-mean way as the outcome, should not jump
  discontinuously at the cutoff — a jump would mean the two sides differ in
  more than friction, undermining the "otherwise similar" premise the design
  relies on. Reported beside the primary result.
* **Negative control.** Outcome shuffled across sides within each symbol-day
  cell (breaking the link between `round_trip_pct` and outcome while holding
  the day/cluster structure fixed) must return a null ITT. Same purpose as
  `PREREG_RANKER_TEST.md`'s negative control.
* **Sign consistency.** ITT computed separately on the first half vs. second
  half of the window, split at the median entry day. Reported, not gating.

## 9. Secondary design — stratified matching (robustness, weaker identification)

Reported beside the RD result, never in place of it. Match each refused
candidate to a passed candidate on `(symbol, entry day)` with `|ΔDTE| ≤ 5`,
`|Δentry_delta| ≤ 0.05`, and comparable relative spread — as specified in the
work order. **Identification weakness, stated up front rather than
discovered after running it:** this repo's gates are deterministic functions
of the matched features, so matching on all of them matches away the
treatment and matching on a subset leaves the rest of the confound in place.
This design exists only as a robustness cross-check against the RD result,
never as the headline, and its own write-up must repeat this weakness beside
its numbers.

## 10. Scope — explicitly not attempted in this registration

* **`negative_ev`** (63,359 refusals, the largest category) is not registered
  here. `decide_verdict` is not a single continuous threshold on one running
  variable in the way `friction` is; designing an RD (or any causal test) for
  it is a separate registration.
* **`condor_universe`** (306 refusals) is a categorical membership test
  (ticker in/not in a fixed index list), not a continuous cutoff — no RD
  applies. A simple group comparison could be registered separately if
  warranted; 306 rows is thin.
* **`top_quintile`** has fired on **zero** recorded candidates to date. There
  is nothing to test.

A finding here is a finding about the **friction gate only**. It says nothing
about whether the gate system as a whole adds value.

## 11. Honest prior

The friction ceiling (25% round-trip) was set by measured single-vs-two-leg
friction levels (`candidate_verdict.py`'s own docstring: 0.7–1.7% single-leg,
~33% per crossing two-leg), not fit to any outcome — there is no prior reason
to expect it sits exactly at the quality-separating point rather than merely
somewhere reasonable. **Most likely outcome, stated before running: NULL.**
This system's history is that absolute per-contract rankers die and
conditioners survive (`PREREG_SECTOR_CONDITIONING_20260813.md` §1); a single
friction percentage is closer in kind to an absolute ranker (one number, no
conditioning on structure or regime) than to a conditioner, so the prior
here leans toward "no measured separation" rather than toward REAL. A NULL
result would not surprise; it would still be reported as this design's own
finding rather than reframed as expected.

## 12. What ships if it works

Nothing automatic, matching every other verdict this repo produces from a
gate study. A REAL result would motivate — as its own separate, later
decision — examining whether 0.25 is the right level, never an automatic
change to it. An INVERTED result would motivate an immediate look at why the
gate is wrong at the margin. A NULL result documents the friction gate as
non-predictive complexity at its current threshold, which is itself
information worth having, independent of whether the gate is kept for other
reasons (as a hard arithmetic floor, it still prevents a spread from handing
back more than a quarter of its own reward to trade).

---

## Parameters

```
gate: friction (candidate_verdict.verdict_for, round_trip_pct > 0.25)
running_variable: round_trip_pct (recomputed from features_json where the
                   stored column is NULL — verified to reproduce refused_by)
cutoff: 0.25
population: Bull Put, Bear Call, Iron Condor candidates only
excluded: credit_gone (4 rows), unpriceable (0 rows)
outcome: candidate_marks.pnl_pct(entry_price_for(row), first mark >= entry_day + horizon)
horizon_primary_days: 5
horizon_secondary_days: 10
bandwidth_primary: 0.05
bandwidth_secondary_robustness: 0.10
cluster: (symbol, date(candidates.ts)) — collapsed to one mean point per side per cluster
estimator: local-linear regression, uniform kernel, separate fit each side,
           ITT = intercept(above) - intercept(below)
n_boot: 4000
seed: 20260902
alpha: 0.05
min_tstat: 3.0     # Harvey's hurdle, src/alloc/report.MIN_TSTAT
min_clusters_per_side: 30
coverage_at_registration:
  primary_5d:   below n=2960 clusters=292 symbols=78 | above n=2483 clusters=299 symbols=82
  secondary_10d: below n=425 clusters=70 | above n=307 clusters=66
  band_total_candidates_pm0.05: 13599
  band_total_candidates_pm0.10: 27197
```

## Result

Not yet run. `scripts/gate_rd_test.py` (not yet written) will append here,
once, after the primary design is computed.
