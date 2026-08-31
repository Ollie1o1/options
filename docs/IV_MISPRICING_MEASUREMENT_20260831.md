# iv_mispricing: measured, no evidence — 2026-08-31

`iv_mispricing` carries a live weight of **0.05** in `CURRENT_WEIGHTS`
(`src/backtest_optimizer.py:73`). This is what it is worth on the ledger as of
2026-08-31.

## Result: no evidence it orders outcomes

Cohort: 599 closed, non-`paper_only` trades with a recorded
`iv_mispricing_score`.

| basis | IC (Pearson) | p | 95% CI |
| --- | ---: | ---: | --- |
| rows, n=599 | **-0.060** | 0.143 | — |
| clustered on entry day, n=73 | **+0.117** | 0.325 | **[-0.120, +0.341]** |

Spearman agrees: -0.055 (rows), +0.070 (clustered).

Three things to read off this, in order of importance:

1. **The CI contains zero.** No evidence the score orders outcomes.
2. **The CI is wide.** `[-0.120, +0.341]` does **not** rule out a large effect.
   This is "not measured well enough to say", not "measured and found null" —
   the same distinction `promotion_verdict` draws between `insufficient` and
   `reject`.
3. **The sign flips with aggregation.** Negative on rows, positive on entry-day
   clusters. A statistic whose sign depends on how you count is not a finding
   in either direction, and row-level significance here would have been the
   count-clusters-not-rows defect all over again.

## Why the cohort understates the feature

**39.1% of these 599 scores are exactly 0.50** — the neutral value forced when
no SVI surface fit exists:

```python
# src/options_screener.py:2777
iv_mispricing_score = iv_mispricing_score.where(surf_conf > 0.05, 0.5)
```

That matches the independently measured pre-fix SVI fit rate of ~38%. So for
roughly two in five of these trades the feature was **absent, not neutral** —
the score carried no information about the contract at all.

A further 30.7% sit at exactly 0.0. Those are *fitted* contracts clipped to zero
by `np.clip(±resid * 5, 0, 1)` for being on the wrong side of fair, which is a
legitimate score rather than a missing one.

**This cohort therefore predates the fitter repairs** (PRs #83, #84, 2026-08-31),
which took the fit rate from 38% to ~100% on realistic slices. The measurement
above describes the feature as it *was*, not as it now is.

## What was NOT done, and why

**The 0.05 weight was left unchanged.**

Dropping it would itself be an unmeasured change to live scoring, and the CI does
not rule out a real positive effect. Raising it is obviously unsupported.
Refuse-don't-rank applies to weights as much as to boards: an interval this wide
licenses no move in either direction.

## What would settle it

Re-measure on trades logged **after** 2026-08-31, where the SVI fit actually
lands and the score reflects the contract rather than a neutral fill. Until that
cohort exists, `iv_mispricing` is carrying weight on no evidence — which is worth
knowing even though it is not yet worth acting on.

**Do not quote the +0.117 as a positive result.** Its CI contains zero and its
sign is not stable.

## A claim checked and withdrawn

While investigating, `fit_svi_surface` was found to leave
`iv_surface_residual = 0.0` on expiries skipped before fitting is attempted
(too few rows, non-positive spot or tenor), while an *attempted and failed* fit
is set to `NaN`. That looked like "not recorded" encoded as zero, which the
schema rule forbids.

It does not reach the score. `surf_conf` is 0.0 for those expiries, so the guard
at `src/options_screener.py:2777` forces the neutral 0.5 before the residual can
matter. The scoring path is sound and no fix was made.

The distinction still exists in the raw `iv_surface_residual` column, which is
consumed directly by `src/cli_display.py`, `src/ai_scorer.py` and
`src/pick_context.py` without a confidence check — where an unevaluated expiry
reads as "priced exactly fairly". That is a display-honesty question, not a
scoring defect, and is left recorded rather than fixed.
