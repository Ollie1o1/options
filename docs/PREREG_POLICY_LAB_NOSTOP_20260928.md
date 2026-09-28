# Preregistration — no-stop hypothesis, small grid

**Written:** 2026-09-28, before this grid has been run.
**Prior run:** `docs/POLICY_LAB_FINDINGS_20260928.md` (null, 750-cell grid)
**Code:** `src/policy_lab/`, branch `feat/policy-lab`

## Why this exists, and the trap it must avoid

The 750-cell sweep returned null. Its best measurable policy carried **no
stop** and reached **+0.0032 of capital at risk per trade**, against that
run's detection floor of +0.012.

**That hypothesis was selected by looking at the results.** Re-testing it on
the same Bull Put data with a smaller grid would not be confirmation — it
would be reducing the multiple-testing penalty on a hypothesis chosen after
the fact, which is the textbook post-hoc selection error and exactly what
this repository's research process exists to prevent.

So the confirmatory test runs on **Bear Call**, which played no part in
generating the hypothesis.

## Hypothesis

**H1 (confirmatory, held out).** On **Bear Call**, removing the stop from the
recorder baseline produces a positive paired mean difference in return on
capital at risk, with family-wise `p < 0.05` across the preregistered 6-cell
grid and a ticker-day-clustered 95% CI excluding zero.

**Bull Put is reported alongside as the in-sample source of the hypothesis and
is explicitly NOT confirmation.** Any Bull Put figure in the result document
must carry that label.

## Why Bear Call is a valid held-out set

Verified before writing this document:

- Its recorder policy is **identical** to Bull Put's: take-profits have
  exit/entry p95 = **0.493** (threshold 0.50), stops have p5 = **2.013**
  (threshold 2.0). So the same baseline applies and no new baseline recovery
  is needed.
- **90,796 closed positions across 2,032 ticker-day clusters** — more clusters
  than Bull Put's 1,870.
- It was excluded from every analysis that produced the no-stop hypothesis.

It is held out in the *structure* dimension, not the time dimension. It shares
the same 26-day window and the same market regime, so it is **not** an
independent regime test and must not be described as one.

## Grid (frozen, 6 cells)

Baseline: `corpusA_recorder` = `ExitPolicy(0.50, 2.0, None, None)`.

| # | take-profit | stop |
|---|---|---|
| 1 | 0.50 | **none** |
| 2 | 0.50 | 3.0 |
| 3 | 0.50 | 1.5 |
| 4 | 0.65 | **none** |
| 5 | 0.65 | 2.0 |
| 6 | 0.35 | **none** |

No DTE or max-hold knobs are armed: the 750-cell run showed `time_exit_dte`
of 21 and 28 are degenerate on this corpus (median entry DTE 17), and holding
them at `None` keeps the grid honest and small.

Expanding this grid after seeing results invalidates the run and requires a
new preregistration.

## Power, measured in advance

Calibrated on synthetic data at C=1,870 clusters, sd=0.046, 30 repetitions per
cell, using the shipped `family_wise_p`:

| planted effect | power at G=6 | power at G=750 |
|---|---|---|
| +0.002 | 0.27 | 0.03 |
| +0.003 | 0.63 | 0.13 |
| +0.004 | 0.80 | 0.40 |
| +0.006 | 0.97 | 0.93 |
| +0.008 | 1.00 | 1.00 |

False-positive rate with no effect planted: **1/30 = 0.03 at G=6** and 0/30 at
G=750 — at or below the nominal 0.05, so the test is correctly calibrated and
slightly conservative.

**The effect being chased measured +0.0032 in the exploratory sweep, so power
here is roughly 0.65-0.70.** A null result at this power is therefore weak
evidence of absence, not strong evidence — it must be reported as
"underpowered to resolve an effect of this size", not as a refutation.

## Decision rule

`src/policy_lab/stats.py::policy_verdict`, unchanged. Promotion requires all
its conditions, including the bounded-walk difference-of-differences.

## Declared in advance

- **The prior is still null.** Twenty-five closed experiments plus the 750-cell
  sweep.
- **A raw take-profit advantage remains inadmissible** — options are bounded
  below by zero and any early-exit rule banks a real advantage from that alone.
  Only the difference-of-differences against the synthetic bounded walk counts.
- **The friction model is unchanged and still modelled**, not observed: 100% of
  Corpus A marks carry no two-sided quote, and the 0.095 half-spread is a
  single-leg figure applied to a two-leg round trip, so it likely under-states
  friction.
- **Corpus B cannot corroborate direction.** `chain_snapshots.snap_date` does
  not label a contemporaneous price, so it cannot be reconciled with ledger
  fills. This run has no independent sign check either.
- **One regime.** 26 trading days, SPY's worst drawdown -4.49%. Nothing here
  speaks to a volatility event.

## What promotion would authorise

A config-only PR adjusting `exit_rules` stop thresholds. Nothing here
authorises real money, changes the auto-log path, or alters any selection rule.
