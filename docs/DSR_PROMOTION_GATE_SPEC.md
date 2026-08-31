# Cluster-Aware Deflated Sharpe and an Honest Promotion Gate

Date: 2026-08-31
Status: approved, not yet implemented

## Why this exists

A "Quantitative Enhancement" brief asked for a Deflated Sharpe Ratio and
Combinatorial Purged Cross-Validation to be built in `src/backtester.py` and
`src/walk_forward.py`. Reading the repo first showed all of that math already
exists in `src/alloc/validate.py`.

A first draft of this spec then proposed fixing and wiring that machinery
broadly. **An audit against live callers cut roughly two thirds of it.** That
audit is recorded here, because what was removed is as important as what
remains.

Everything in scope below has a named production caller. Everything cut did
not.

## What the audit found

### Live defect 1: the promotion gate's DSR is inflated toward promotion

`src/alloc/report.py:120` — DSR is not merely reported, it **gates promotion**:

```python
if (result.get("dsr", 0.0) < MIN_DSR            # MIN_DSR = 0.5
        or result.get("pbo", 0.0) >= MAX_PBO
        or abs(result.get("tstat_clustered", 0.0)) < MIN_TSTAT
        or result.get("tstat_clustered", 0.0) < 0):
    return "reject"
```

This checks `tstat_clustered` (clustered on entry day) and `dsr` side by side.
But `deflated_sharpe` uses `n = r.size` — the raw row count — for both
`sqrt(T-1)` and its `trial_variance = 1/n` default. One condition is
cluster-aware, the other is not, and nothing marks the difference.

The ledger is heavily clustered: **748 closed rows over 74 distinct entry
days**. Per strategy:

| Strategy | rows | entry days | rows/day | avg hold |
| --- | ---: | ---: | ---: | ---: |
| Long Call | 253 | 58 | 4.4 | 7.5d |
| Bull Put | 145 | 49 | 3.0 | 3.0d |
| Bear Call | 135 | 41 | 3.3 | 2.1d |
| Short Put | 109 | 20 | 5.5 | 4.4d |
| Iron Condor | 57 | 13 | 4.4 | 23.7d |
| Long Put | 49 | 12 | 4.1 | 4.0d |

Measured impact on a 253-row / 58-cluster cohort at `n_trials=200`:

| per-trade SR | DSR (rows=253) | DSR (clusters=58) | verdict |
| ---: | ---: | ---: | --- |
| 0.201 | 0.662 | 0.117 | promote -> reject |
| 0.271 | 0.932 | 0.251 | promote -> reject |
| 0.354 | 0.997 | 0.474 | promote -> reject |

At SR 0.354 the gate reads 0.997, near-certainty, where the honest value is
0.474 — a coin flip. The gate is systematically too permissive across the whole
plausible range.

The overall 10.1 rows/day and the per-strategy 3.0-5.5 rows/day are both
correct and not in conflict: strategies share entry days, so the union of days
is smaller than the sum of per-strategy days. Clustering is computed **within**
the cohort being measured, which is the per-strategy figure.

### Live defect 2: the gate's PBO condition can never fire

`summarise()` never sets a `"pbo"` key. Every assignment was checked. So
`result.get("pbo", 0.0)` always returns `0.0`, and `0.0 >= 0.5` is always
`False`.

It is green in CI because `tests/test_alloc_report.py:31` builds its fixture by
hand:

```python
def _res(dsr=0.9, tc=3.5, broad_pnl=50.0, pbo=0.2, n=200):
    return {"n": n, "dsr": dsr, "pbo": pbo, ...}
```

`test_high_pbo_rejects` asserts on a dict shape `summarise` never produces.

Of the gate's four conditions, one is inflated toward promotion and one is
dead. `n`, `tstat_clustered`, and the `broad` stratum check work as written.

### Live defect 3: two different quantities named "sharpe"

- `src/backtester.py:539` computes `mean/std * sqrt(252)` — an annualised Sharpe
- `src/backtester.py:578` computes `mean/std * sqrt(n)` — a t-statistic

Both are called `sharpe` inside `run_backtest`.

### Live defect 4: an undeflated in-sample maximum is published

`src/backtester.py:569` sweeps `np.arange(0.3, 0.9, 0.05)` — 12 trials — and
keeps the best. That in-sample maximum is published as `optimal_threshold` and
consumed at `src/backtester.py:1114`, with no deflation of any kind. This is
precisely what DSR exists to correct.

### Live defect 5: an orphaned field

`search_bar_sharpe` is set at `src/walk_forward.py:276` and `:406` and read
nowhere. Walk-forward reports IC, not Sharpe, so the bar has no counterpart in
its own summary and can only mislead.

## What was cut, and why

These were in the first draft and are **not** being done:

- **Fixing `purge_embargo`'s units bug.** The function subtracts a day count
  from a sample index, under-purging by the rows-per-day density (3.0-5.5x
  here). The bug is real, but `purge_embargo` and `cpcv_splits` have **zero
  production callers** — tests only. The alloc DSR comes from `replay` trades
  passed straight to `summarise` and never touches a CPCV split. The first
  draft claimed "the wrong purge sits under a published DSR." That was false.
- **Moving `src/alloc/validate.py` to `src/validation.py`.** Pure churn: no
  behaviour change, three import sites plus a test rename, layering tidiness
  only.
- **Wiring CPCV into `walk_forward.py`.** Expected to refuse on every strategy
  (Long Call has 58 clusters; a 7.5-day purge plus 5-day embargo eats ~12.5 days
  either side of each test block). Building a maintained code path that produces
  nothing today is not an improvement.

## Scope

### 1. `deflated_sharpe` takes a required `n_eff`

Signature becomes `deflated_sharpe(returns, n_trials, n_eff, trial_variance=None)`.

`n_eff` is **required and positional**, no default. `sqrt(T-1)` uses `n_eff`;
`trial_variance` defaults to `1/n_eff`. Precedent for refusing a default: the
board was ranked by `quality_score` for months because `sort_by` had one, and a
default argument is invisible to AST guards.

The formula is otherwise unchanged and already correct, including the
documented `trial_variance` units fix at `src/alloc/validate.py:105-108`.

Add `effective_n(entry_dates) -> int` returning the distinct-entry-day count, so
call sites do not hand-roll it. Entry-day clustering matches the convention
`clustered_tstat` already uses in the same module.

**Direction of effect is one-way.** Both `1/n_eff > 1/n` (raising the bar `sr0`)
and `sqrt(n_eff - 1) < sqrt(n - 1)` (shrinking the z-score) push DSR down. The
gate can only turn `promote` into `reject`, never the reverse.

This is not a signal being flattened. No return value changes and no signal is
damped; only a sample count that was wrong — 253 independent observations
claimed where 58 exist — is corrected. The signal is untouched; the error bar
becomes honest.

### 2. `summarise` passes the real cluster count

`src/alloc/report.py:92-93` passes `effective_n` of the closed trades into both
`deflated_sharpe` calls. `dsr` and `tstat_clustered` then cluster identically.

Report `n_eff` alongside `n` in the summary dict so a reader can see the gap.

### 3. Remove the dead PBO condition

Delete the `pbo` clause from `promotion_verdict` and record in the docstring
that PBO is not measured, and why: it requires in-sample/out-of-sample pairs
across CPCV paths, which do not exist.

**Zero behaviour change** — the condition never fired. What is removed is a
false impression of rigor: the gate reads as four checks and is three.

`pbo_from_pairs` and `probability_of_backtest_overfitting` are kept. They are
clean pure functions with no units bug and remain available if CPCV is ever
built.

### 4. Backtester: deflate the threshold sweep

`run_backtest` gains DSR over its per-trade return series, with `n_trials = 12`
matching the sweep at `src/backtester.py:569`.

Summary gains `dsr`, `dsr_undeflated`, `n_trials`, `n_eff`, `n_rows`. Purely
additive — no existing key changes meaning.

### 5. Backtester: separate the two "sharpe" quantities

Rename the **local** variable `s_sharpe` at `src/backtester.py:578` to
`s_tstat`, because `mean/std * sqrt(n)` is a t-statistic.

The published `"sharpe"` dict key at `:607` and `:668` is **left alone** — it
has consumers at `:992`, `:707`, and in `src/dolt_blend.py`. Renaming a
published key is out of scope; this is a local variable only.

### 6. Remove `search_bar_sharpe`

Delete from `src/walk_forward.py:276` and `:406`, and drop the now-unused
`expected_max_sharpe` import at `:15`. Confirmed read nowhere.

### 7. Delete the dead, broken CPCV primitives

Delete `cpcv_splits` and `purge_embargo` from `src/alloc/validate.py`, along
with the now-unused `DEFAULT_BLOCKS`, `DEFAULT_K`, `DEFAULT_EMBARGO` constants
and the `itertools` import if it becomes unused.

Leaving a plausible-looking but silently wrong purge in the codebase is a trap:
whoever wires CPCV next would call it and get a wrong split with no signal that
anything is off. `walk_forward.purge_overlapping` is the correct interval-based
primitive and should be what anyone reaches for.

This deletes a hazard, not a capability — nothing calls these.

## Testing

### The structural guard, most important test

`promotion_verdict` reads only keys that `summarise` actually produces. Assert
this by introspecting a real `summarise` output rather than a hand-built dict.

This is the root-cause fix. The dead PBO condition survived because
`tests/test_alloc_report.py:31` constructs its own fixture, so the gate was
never tested against reality. The guard catches this class of defect for all
conditions, not just the one found.

### Other tests

- `deflated_sharpe` cannot be called without `n_eff` (raises `TypeError`).
- DSR strictly decreases as `n_trials` rises and as `n_eff` falls, holding
  returns fixed.
- Regression, gate direction: a cohort at rows=253 / clusters=58 with per-trade
  SR ~0.27 yields `promote` on row count and `reject` on cluster count. Pins the
  table above.
- `effective_n` counts distinct entry days, not rows, and is insensitive to how
  many trades share a day.
- `summarise` reports `n_eff <= n`, with equality only when every trade has a
  distinct entry day.
- Backtester DSR present in the summary, and `dsr <= dsr_undeflated`.
- Existing `tests/test_alloc_validate.py` assertions preserved for the surviving
  functions; the `cpcv_splits` and `purge_embargo` cases are removed with the
  functions.
- `tests/test_alloc_report.py:135` and `tests/test_strategy_evidence.py:57`
  updated for the new `deflated_sharpe` signature.
- `test_high_pbo_rejects` is deleted with the condition it tested, replaced by
  the structural guard above.

## Verification

Run `scripts/test.sh` and **check the exit code**, not the presence of "OK" in
the output. `8 pytest-only modules` is benign.

Beyond the suite, run the alloc report on real data and confirm the printed DSR
and verdicts change as predicted. Green tests are not sufficient evidence here:
the whole reason the PBO gate was dead is that the suite was green.

`mypy` is CI-only in this repo by deliberate policy, because venv creation and
`pip install` are not permitted from this environment. Type correctness is
verified by the CI ratchet, not a local run.

The brief's suggested `pytest scripts/run_tests.py` is not a command that works
here; `scripts/test.sh` is the runner.

## Expected outcome

Strategies currently promoted may flip to rejected. That is the finding, not a
side effect: the gate has been reading a DSR built on 4.4x more independent
observations than exist, while a dead condition sat beside it.

No number gets better as a result of this work. What improves is that the
numbers stop overstating their own confidence.

## Out of scope

Each gets its own spec, plan, and implementation cycle:

- **Phase 1.1** — SVI butterfly wing bound `b(1+|rho|) < 4`, calendar arbitrage
  across maturities, risk-neutral skew and kurtosis. SVI fitting
  (`src/iv_surface.py`) and Breeden-Litzenberger RND
  (`src/probability_lab/rnd.py`) already exist. Note that `_enforce_constraints`
  at `src/iv_surface.py:60` projects parameters **after** the fit, so the
  reported `fit_quality` describes the unprojected parameters.
- **Phase 2** — execution microstructure. `src/cost_calibration.py` is a
  tenor-range guard, not a cost fitter; `src/execution/slippage.py` is a
  real-vs-paper fill recorder, not a fill model. The measured spread surface
  still binds nothing, which should be addressed before a second cost model is
  added.
- **Phase 3** — Ledoit-Wolf, cross-sleeve Greek budgeting, CVaR. `RiskAggregator`
  in `src/portfolio_risk.py` already computes portfolio Greeks and VaR/CVaR via
  Monte Carlo. `sklearn` is unavailable and must not be installed; Ledoit-Wolf
  has a closed form on numpy.
- **Phase 4** — orthogonalization and probability calibration,
  **measurement-only, with no hook into `src/core/sizing.py`**. Kelly sizing
  stays unwired. The book has no measured edge (PF 1.044, CI [0.87, 1.24] on
  capital at risk; 0.971 on entry premium), entries are drawn at random among
  gate survivors so the cohort cannot validate the rule that selected it,
  `ev_net` already failed a preregistered test at n=2137, and news sentiment has
  measured IC around zero with large effects ruled out.
