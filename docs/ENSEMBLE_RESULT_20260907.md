# Ridge ensemble — Result (2026-09-07)

Does a ridge combination of the credit-richness-residualized
`ATTRIBUTION_FEATURES` beat the single best residualized feature on holdout?
One look, run exactly as pre-registered in `docs/PREREG_ENSEMBLE_20260905.md`,
via `scripts/ensemble_test.py`, for both `bull_put` and `long_call`.
Reporting only — no scoring, gating, or allocation-path changes.

## Verdict

**NULL on both structures.** Neither clears Harvey's `|t| ≥ 3.0` hurdle on
holdout, and in both cases the ensemble's holdout IC is smaller in magnitude
than the single best individually-residualized feature measured on the same
holdout population — combining features did not just fail to add a
significant new signal, it underperformed simply picking the best single
one.

| | bull_put | long_call |
|---|---:|---:|
| in-sample n / holdout n | 105 / 69 | 105 / 69 |
| model n_fit (after residualization) | 66 | 63 |
| features used (of 19 candidates) | 16 | 14 |
| chosen alpha | 0.3 | 100.0 |
| **in-sample ensemble IC (t_clustered)** | +0.5252 (**3.944**) | +0.2969 (2.57) |
| **holdout ensemble IC (t_clustered) — decision stat** | +0.1236 (**0.261**) | −0.1495 (−0.758) |
| best single residualized \|IC\| on the same holdout | 0.2465 | 0.3744 |

## The overfitting signature, named plainly

`bull_put`'s in-sample ensemble actually *cleared* the significance hurdle
(`t_clustered = 3.944`) — if this script stopped at in-sample, it would have
looked like a real discovery. It collapsed to `t_clustered = 0.261` on
holdout, the same magnitude drop every dead feature in
`docs/HOLDOUT_20260809.md` showed. `long_call` shows the sharper version of
the same failure: the ensemble's sign **flipped** from positive in-sample to
negative on holdout — not just weaker, but pointing the wrong way. Both are
textbook signatures of a model that found structure in noise despite the
blocked, non-shuffled, in-sample-only cross-validation the design used
specifically to guard against this.

**The most informative single number here may be the "beats best single
feature" comparison, not the t-stat.** Ridge combining 14-16 residualized
features on 63-66 in-sample trades is a genuinely thin fit — regularization
controls variance, it does not manufacture signal that was not in the data
to begin with. On this sample size, the honest reading is that there was not
enough independent information across 14-16 features to combine profitably;
picking one feature and living with its own uncertainty did better than
trying to blend many.

## What this does not show

- **Not evidence that no combination could ever work** — this is evidence
  that *this* combination (ridge, this feature list, this alpha grid, this
  sample size) does not. A materially different design (fewer, pre-selected
  features chosen on grounds independent of this measurement; more data;
  a different penalty) is a different hypothesis requiring its own
  registration, not a patch to this one, per the prereg's own scope note.
- **Does not reopen any single-feature result** — `long_delta`'s standing
  survival on `long_call` and every other feature's standing death are
  unchanged; this is a statement about combining them, not about any one of
  them individually.
- **Small n throughout.** 63-69 holdout trades per structure is thin by any
  standard; a much larger sample might tell a different story, but that is
  a statement about statistical power, not a reason to relitigate this
  specific frozen result.

## Next step

Both `bull_put` and `long_call` ensembles are documented as NULL, with the
overfitting signature named so it isn't rediscovered by surprise later. Per
the prereg's decision rule, this specific registration is closed. Combined
with `docs/OUTLOOK_FEATURE_RESULT_20260905.md`'s NULL result, both of this
session's "combine or transfer instead of hunting a new raw feature"
experiments have now returned negative — the honest state of this book's
scoring/ranking problem remains what `docs/HOLDOUT_20260809.md` already
established: no measured signal survives holdout, however combined.
