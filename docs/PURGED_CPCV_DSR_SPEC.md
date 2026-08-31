# Interval-Purged CPCV and Cluster-Aware Deflated Sharpe

Date: 2026-08-31
Status: approved, not yet implemented

## Why this exists

A "Quantitative Enhancement" brief asked for a Deflated Sharpe Ratio and
Combinatorial Purged Cross-Validation to be built in `src/backtester.py` and
`src/walk_forward.py`. Reading the repo first showed that essentially all of
that math **already exists and is tested** in `src/alloc/validate.py`:

| Requested | Already present |
| --- | --- |
| Deflated Sharpe with skew, kurtosis, `sqrt(T-1)` | `deflated_sharpe()` |
| `SR*` via expected max of N trials, Euler-Mascheroni | `expected_max_sharpe()` |
| CPCV combinatorial splits `C(n_blocks, k)` | `cpcv_splits()` |
| Purge plus embargo, embargo default 5 | `purge_embargo()` |
| Probability of backtest overfitting | `probability_of_backtest_overfitting()`, `pbo_from_pairs()` |

So this is not an implementation project. It is a **correctness and reach**
project: the machinery is confined to the `alloc` sleeve, it is not applied to
the paths that publish claims, and two of its core functions count the wrong
thing.

## The two defects this fixes

### Defect 1: `purge_embargo` subtracts days from an index

`src/alloc/validate.py:60` computes:

```python
purge_from = lo - max(0, holding_days)
embargo_to = hi + max(0, embargo_days)
```

`lo` and `hi` are **sample indices**. `holding_days` and `embargo_days` are
**days**. Those are different units. The function under-purges by the
rows-per-day density of the data.

Measured on `paper_trades.db` (closed, non-paper-only, 2026-08-31):

| Strategy | rows | entry days | rows/day | avg hold | index positions purge should remove | what the code removes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Long Call | 253 | 58 | 4.4 | 7.5d | ~33 | 7.5 |
| Bull Put | 145 | 49 | 3.0 | 3.0d | ~9 | 3.0 |
| Bear Call | 135 | 41 | 3.3 | 2.1d | ~7 | 2.1 |
| Short Put | 109 | 20 | 5.5 | 4.4d | ~24 | 4.4 |
| Iron Condor | 57 | 13 | 4.4 | 23.7d | ~104 | 23.7 |
| Long Put | 49 | 12 | 4.1 | 4.0d | ~16 | 4.0 |

The embargo has the same error: `DEFAULT_EMBARGO = 5` applied as 5 index
positions is about 1.1 calendar days at 4.4 rows/day, not the 5 days intended.

`walk_forward.purge_overlapping` (landed 2026-08-31) does this correctly, using
each trade's measured `[entry_date, exit_date]` interval. Its docstring states
it avoids "days-to-index conversion" — describing exactly the bug the older
function has. The repo currently holds two purge implementations that disagree,
and the wrong one sits under a published DSR.

### Defect 2: DSR counts rows where the report counts clusters

`src/alloc/report.py` publishes `tstat_clustered` (clustered on entry day) and
`dsr` in the same dict. But `deflated_sharpe` uses `n = r.size`, the raw row
count, for both `sqrt(T-1)` and its `trial_variance = 1/n` default. One
statistic is cluster-aware, the other is not, and nothing marks the difference.

The alloc docstring estimates clustering costs "roughly a factor of two on this
data." On the main ledger it is far worse: **748 closed rows over 74 distinct
entry days, 10.1 rows/day overall**; Long Call alone is 253 rows over 58 days.
Running DSR on rows inflates `sqrt(T-1)` by about 2.1x for Long Call
(`sqrt(252/57)`).

The overall 10.1 rows/day and the per-strategy 3.0-5.5 rows/day in the table
above are both correct and not in conflict: strategies share entry days, so the
union of days is smaller than the sum of per-strategy days. Clustering is
computed **within** the cohort being measured, which is the per-strategy figure.

This is the repo's documented repeat defect — count clusters, not rows — living
inside the module being promoted.

## Decisions taken

1. **Reuse, do not rebuild.** No second copy of DSR/CPCV in `backtester.py`.
2. **Fix the units defect** rather than wiring around it.
3. **DSR goes to the backtester only.** Walk-forward measures IC (ordering);
   DSR is defined on a return series and does not speak to ordering.
4. **`n_eff` is a required argument**, no default. Precedent: the board was
   ranked by `quality_score` for months because `sort_by` had a default, and a
   default argument is invisible to AST guards.
5. **Refusal is a first-class result**, with no auto-tuning of `n_blocks`/`k`.
6. **CPCV blocks are built over entry-day clusters**, not trade rows.
7. **Move with no compatibility shim.**

## Architecture

`src/alloc/validate.py` moves to **`src/validation.py`**. It is already a
general statistics module with no alloc-specific imports; having `backtester.py`
reach into a sleeve for it would invert the layering.

Import sites to update (three):

- `src/walk_forward.py:15`
- `src/alloc/report.py:25`
- `tests/test_alloc_validate.py:14` (renamed to `tests/test_validation.py`)

`src/leverage/__main__.py:29` imports a different `.validate` (leverage's own)
and is untouched.

No compatibility shim. A shim lets a stale import path survive silently, and
there are only three call sites.

The module gains one organising idea: **the unit of observation is the
entry-day cluster, not the trade row.** Every function taking a sample count
takes clusters. Rows enter only when computing a cluster's mean.

## Components

### `ClusterIndex`

Built from `(entry_date, exit_date)` pairs. Exposes:

- ordered unique entry days
- row index to cluster id mapping
- each cluster's `[min entry, max exit]` interval

This is the single place the row/cluster distinction is resolved, so nothing
downstream can count the wrong thing.

### `effective_n(index) -> int`

Returns the cluster count. Exists so call sites do not hand-roll it.

### `cpcv_splits(index, n_blocks, k)`

Replaces the row-indexed version at `src/alloc/validate.py:33`. Blocks are
contiguous spans of **entry days**, so one block is one period of market time
regardless of how many trades were opened in it. Returns train/test as cluster
ids. Yields exactly `C(n_blocks, k)` splits.

### `purge_embargo(train, test, index, embargo_days=5)`

Replaces the index arithmetic at `src/alloc/validate.py:48`.

- Purges any training cluster whose `[entry, exit]` interval intersects the
  test window, using the same inclusive-both-ends rule as
  `walk_forward.purge_overlapping:143`.
- Embargo extends the test window forward by 5 **calendar** days before the
  intersection test.
- The `holding_days` parameter is **deleted, not defaulted**. It is the source
  of the units bug and no caller can supply it correctly.

`walk_forward.purge_overlapping` becomes a thin wrapper over this, so one purge
rule exists rather than two that disagree.

### `deflated_sharpe(returns, n_trials, n_eff, trial_variance=None)`

`n_eff` is **required and positional**. `sqrt(T-1)` uses `n_eff`;
`trial_variance` defaults to `1/n_eff`. This is a deliberate breaking change to
a tested function; both existing call sites pass a real cluster count.

The formula itself is unchanged and already correct, including the documented
`trial_variance` units fix at `src/alloc/validate.py:105-108`.

## Data flow

### Backtester

`run_backtest` gains DSR on its per-trade return series.

`n_trials` counts the threshold sweep at `src/backtester.py:569`:
`np.arange(0.3, 0.9, 0.05)` is 12 trials whose in-sample maximum is published
as `optimal_threshold` with no deflation at all. This is precisely what DSR
exists to correct.

Summary gains: `dsr`, `dsr_undeflated`, `n_trials`, `n_eff`, `n_rows`.

The two conflicting Sharpe definitions in one function are separated:

- `src/backtester.py:539` computes `mean/std * sqrt(252)` — becomes
  `sharpe_annualised`
- `src/backtester.py:578` computes `mean/std * sqrt(n)` — becomes `tstat`,
  because that is what it computes

### Walk-forward

Keeps IC as its metric. Gains CPCV over entry-day clusters using the shared
purge.

`search_bar_sharpe` at `src/walk_forward.py:406` is **removed**. Walk-forward
produces no Sharpe, so the bar has no counterpart in the summary and can only
mislead. The `n_trials` field (`TRIALS_PER_FOLD * len(per_fold)`) is correct and
stays.

### Alloc

`report.summarise` passes `effective_n` into `deflated_sharpe`, so `dsr` and
`tstat_clustered` finally cluster the same way.

## Refusal

A first-class result, mirroring the existing `_refused_summary` path at
`src/walk_forward.py:235`.

The refusal decision **stays in `src/walk_forward.py`**, which already owns
`MIN_FOLDS` and the refusal-writing path. `src/validation.py` stays a pure
statistics module: it returns splits and counts and never decides whether a
result is reportable.

When surviving blocks fall below `MIN_FOLDS`, the summary carries
`refused: True` and a reason stating the arithmetic:

- `n_clusters` — clusters available before any removal
- `n_purged_overlap` — clusters removed because their interval intersects the
  test window
- `n_purged_embargo` — clusters removed **only** because of the 5-day embargo
  extension, i.e. they did not intersect the raw test window. Reported
  separately so the two removals never double-count.
- `n_blocks_surviving`

No auto-tuning of `n_blocks` or `k`. That search would itself be an undeflated
trial count — a new overfitting surface introduced by the tool meant to detect
overfitting.

**Refusal is the expected outcome.** Long Call has 58 clusters; a 7.5-day purge
plus a 5-day embargo eats about 12.5 days on each side of every test block. Most
or all strategies are expected to refuse. This is consistent with the existing
finding that at `train_size=44` not one fold of any strategy survived an honest
purge. A refusal here reads as the system working, not failing.

## Testing

New `tests/test_validation.py` (replacing `tests/test_alloc_validate.py`), plus
additions to `tests/test_walk_forward.py`.

**Units regression — the headline test.** A synthetic ledger at 5 rows/day with
10-day holds. The old index arithmetic leaves overlapping trades in train;
interval purging removes all of them. This test fails against today's code and
passes after the change.

Further tests:

- Embargo is calendar days: a cluster 3 days after the test window is purged,
  one 7 days after survives — at any row density.
- `deflated_sharpe` cannot be called without `n_eff` (raises `TypeError`).
- DSR strictly decreases as `n_trials` rises and as `n_eff` falls, holding
  returns fixed.
- `cpcv_splits` yields exactly `C(n_blocks, k)` splits; no cluster appears in
  both train and test; blocks partition entry days with no gaps or overlaps.
- Refusal names its arithmetic and never emits an IC.
- All existing `tests/test_alloc_validate.py` assertions preserved under the new
  import path and signature.
- `tests/test_alloc_report.py:135` and `tests/test_strategy_evidence.py:57`
  updated for the new signature.

## Verification

Run `scripts/test.sh` and **check the exit code**, not the presence of "OK" in
the output. `8 pytest-only modules` is benign.

`mypy` is CI-only in this repo by deliberate policy, because venv creation and
`pip install` are not permitted from this environment. Type correctness is
verified by the CI ratchet, not by a local run.

The brief's suggested `pytest scripts/run_tests.py` is not a command that works
here; `scripts/test.sh` is the runner.

## Out of scope

Each of these gets its own spec, plan, and implementation cycle:

- **Phase 1.1** — SVI butterfly wing bound `b(1+|rho|) < 4`, calendar arbitrage
  check across maturities, risk-neutral skew and kurtosis. Note that SVI fitting
  (`src/iv_surface.py`) and Breeden-Litzenberger RND
  (`src/probability_lab/rnd.py`) already exist, and that
  `_enforce_constraints` at `src/iv_surface.py:60` projects parameters **after**
  the fit, so the reported `fit_quality` describes the unprojected parameters.
- **Phase 2** — execution microstructure. Note that `src/cost_calibration.py` is
  a tenor-range guard, not a cost fitter, and `src/execution/slippage.py` is a
  real-vs-paper fill recorder, not a fill model. The measured spread surface
  still binds nothing, which should be addressed before a second cost model is
  added.
- **Phase 3** — Ledoit-Wolf, cross-sleeve Greek budgeting, CVaR. Note that
  `RiskAggregator` in `src/portfolio_risk.py` already computes portfolio Greeks
  and VaR/CVaR via Monte Carlo. `sklearn` is unavailable and must not be
  installed; Ledoit-Wolf has a closed form on numpy.
- **Phase 4** — orthogonalization and probability calibration, **measurement-only,
  with no hook into `src/core/sizing.py`**. Kelly sizing stays unwired. The book
  has no measured edge (PF 1.044, CI [0.87, 1.24] on capital at risk; 0.971 on
  entry premium), entries are drawn at random among gate survivors so the cohort
  cannot validate the rule that selected it, `ev_net` already failed a
  preregistered test at n=2137, and news sentiment has measured IC around zero
  with large effects ruled out.
