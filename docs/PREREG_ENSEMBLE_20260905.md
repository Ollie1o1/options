# Pre-registration — does a ridge combination of weak features beat any one alone?

**Frozen 2026-09-05. Immutable.**

Written before this design's fitting code has ever been run against the
2020-21 holdout window, or against any window at all beyond the synthetic
unit tests in `tests/test_alloc_ensemble.py` used to verify the arithmetic.
No real trade's outcome has been looked at while writing this document.

## 1. Why this, and why now

Every single-feature hypothesis this repo has tested has died the same way,
once `residual_ic` controlled for credit richness (`docs/HOLDOUT_20260809.md`):
`atm_iv`, `vol_of_vol`, `iv_rank`, `skew_25d` all collapse or flip sign under
the control. `term_slope` failed its holdout outright. Only `long_delta` (call
side) survives, and the strategy it sits inside is itself cost-challenged.

That is a strong, repeated result for *single* features tested *one at a
time*. It says nothing about whether several individually-weak, genuinely
independent (post-control) signals could combine into something that clears
the bar together. This registration is that question, asked the way this
repo's own methodology already answers every other one: control for credit
richness first, then measure — never after.

**The trap to name up front, because it is exactly the "run 1000 strategies"
instinct that started this line of work**: a multivariate model fit on the
same 18 features that already failed individually will, if built carelessly,
just rediscover the credit-richness identity in a new shape and call it a
discovery. The whole point of the design below is that the control is
*inside* the fit, not bolted on after a result is already in hand.

## 2. Hypothesis

**H-ENSEMBLE.** A ridge-regularized linear combination of the
credit-richness-residualized `ATTRIBUTION_FEATURES` (`src/alloc/__main__.py`,
currently an 18-feature list, minus the two controls) predicts realized
return on capital, on the 2020-21 holdout, more strongly than the single best
individually-residualized feature does on the same window.

**Estimand.** The day-clustered Spearman rank-IC of the frozen ensemble's
score against realized RoC on holdout trades (`ensemble_ic`), compared
against `max(|residual_ic(trades, f)["ic"]| for f in features)` measured
independently on the same holdout population.

**Decision rule, fixed now:**
- **REAL**: holdout `|t_clustered| ≥ 3.0` (`src/alloc/report.py::MIN_TSTAT`,
  the same Harvey hurdle every other prereg here uses) **and** the ensemble's
  `|ic|` exceeds the best single residualized feature's `|ic|` on the same
  holdout population.
- **NULL**: `|t_clustered| < 3.0`, regardless of whether it nominally beats
  the best single feature — an insignificant number beating another
  insignificant number is not a finding.
- **INVERTED / NOT A DISCOVERY**: significant but does *not* beat the best
  single feature — combining added complexity without added information.
- One look. No EXTEND state, matching `PREREG_GATE_RD_20260902.md`'s
  precedent for why that state doesn't exist here either.

## 3. The population and windows — LOCKED, matching precedent exactly

Same universe, same two windows as `docs/HOLDOUT_20260809.md`, for direct
comparability with the numbers already published there:

- **In-sample (fit window):** `--start 2022-01-07 --end 2024-12-31`.
- **Holdout (the one look):** `--start 2020-01-27 --end 2021-12-31`.
- **Universe:** `--all-names` (the full 123-symbol, COVID-inclusive cache),
  not the tight-spread MEGA subset — the holdout study ran on the full
  universe and this should be directly comparable to it.
- **Structures:** `bull_put` and `long_call`, fit and evaluated **separately**
  — feature relationships already flip sign between them (e.g. `skew_25d`:
  −0.17 on bull_put, +0.17 on long_call in the holdout study), so a pooled
  model would average away exactly the structure-conditional signal this
  design is trying to find.

## 4. The model — LOCKED

- **Candidate features:** `ATTRIBUTION_FEATURES` from `src/alloc/__main__.py`
  (18 features as of this freeze), minus `RESIDUAL_CONTROLS` (`credit_pct_width`,
  `atm_iv`). No hand-curation beyond that removal — the point of ridge over
  lasso is that it is supposed to handle the known collinearity among the
  remaining IV-level proxies (`rv`, `vol_of_vol`, `iv_rank`) itself, and
  hand-picking a "better" subset after seeing which features are collinear
  would itself be a design choice made with partial knowledge of the answer.
- **The control:** every candidate feature is residualized against
  `credit_pct_width`/`atm_iv` via `_residual_values` — the exact function
  `residual_ic` already uses, unmodified — before it is allowed into the
  design matrix. A feature `_residual_values` cannot measure (too few trades,
  collinear with the controls) is dropped from that fit, not zero-filled.
- **Combination:** ridge regression (closed-form,
  `src/alloc/ensemble.py::_ridge_fit`) on the residualized features,
  standardized to mean 0 / std 1 over the fit sample. Ridge rather than
  lasso specifically because several candidates are known-collinear proxies
  for "how volatile is this name," and ridge shrinks a correlated group
  together instead of arbitrarily keeping one and zeroing the rest.
- **Regularization strength:** chosen by blocked, non-shuffled
  cross-validation **within the in-sample window only**
  (`DEFAULT_ALPHAS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0)`,
  `DEFAULT_N_BLOCKS = 6` contiguous time blocks, never shuffled — a shuffled
  fold would put same-week trades on both sides of a split, the same leak
  `split_by_time`'s docstring already names). Ties broken toward the LARGER
  alpha (more shrinkage), the more conservative choice when the data cannot
  tell two regularization strengths apart.
- **One frozen model per structure.** The chosen alpha and the resulting
  coefficients, fit on the FULL in-sample window at that alpha, are what gets
  applied to holdout. Nothing about the fit is re-tuned after seeing holdout
  data, by construction — the holdout-scoring function (`score_ensemble`)
  takes a frozen `EnsembleModel` and does not touch fitting code at all.

## 5. The disclosed limitation — read this before citing any holdout number

`_residual_values` computes ranks and residuals **within whatever trade set
it is given** — that is how `residual_ic` has always behaved, and this design
keeps that convention rather than inventing a different one for this one
case. So scoring the holdout does **not** replay a frozen rank-transform
fitted on in-sample data; it residualizes the holdout trades against the
controls **within the holdout sample itself**. Only the ridge weights on the
standardized residual columns are frozen and carried over unmodified.

This means a positive holdout result supports "these combination weights,
learned once, still line up with return in a different regime" — a real and
useful claim — but does **not** support the stronger claim of a fully
leak-free rank-transform pipeline end to end. Flagged here so it cannot be
overclaimed later, the same way `docs/HOLDOUT_20260809.md` flags that every
number there assumes a fill at the mid, which has never been measured.

## 6. What's built now vs. what happens next

**Built and merged in this PR, before any holdout number exists:**
`src/alloc/ensemble.py` (`fit_ensemble`, `score_ensemble`, `ensemble_ic`,
`EnsembleModel`), a shared `_residual_values` extraction from
`src/alloc/attribution.py::residual_ic` (behavior-preserving refactor — the
55 pre-existing `residual_ic`/`feature_ic` tests pass unchanged), and 16 unit
tests against synthetic data verifying the ridge arithmetic, the blocked-CV
folding, the credit-richness control actually excludes a renamed control,
and that combining several weak-but-real synthetic signals beats any one of
them alone.

**Explicitly NOT done in this PR, on purpose:** no CLI wiring that runs this
against the real Dolt cache, and no holdout number of any kind, for either
structure. That is the next step, and it lands as its own separate PR/commit
once this design itself has been reviewed — the same "prereg lands as its
own PR, result lands later in a separate one" sequencing as
`PREREG_GATE_RD_20260902.md` / `GATE_RD_RESULT_20260902.md`.

## 7. Explicitly out of scope this round

- Any structure other than `bull_put`/`long_call` (matches the holdout
  study's scope).
- `term_slope_1m3m` and `entry_depth`, which the `ATTRIBUTION_FEATURES` list
  itself notes are measurable only on optionsDX / not on the Dolt cache
  respectively — they enter the candidate list and get silently dropped by
  `_design_matrix` wherever the underlying data can't support them, exactly
  like every other unmeasurable feature.
- Re-litigating any single-feature result already resolved in
  `docs/HOLDOUT_20260809.md` — this design does not re-test whether any one
  feature works alone; it only tests whether the *combination*, built with
  the same control discipline, adds something a single feature doesn't.
