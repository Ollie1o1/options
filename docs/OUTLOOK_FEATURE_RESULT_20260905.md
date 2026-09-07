# Outlook-composite transfer test — Result (2026-09-05)

Does `outlook_composite` — `src/outlook`'s already-validated (66-72% right
on bullish calls, relative IC +0.05-0.08) construction, transferred
point-in-time onto single-name `bull_put` — have holdout skill? One look,
run exactly as pre-registered in `docs/PREREG_OUTLOOK_FEATURE_20260905.md`,
via `scripts/outlook_feature_test.py`. Reporting only — no scoring, gating,
or allocation-path changes.

## Verdict

**NULL.** `t_clustered = 0.446` on the residualized (credit-richness
controlled) holdout measure, nowhere near Harvey's hurdle (`|t| ≥ 3.0`) in
either direction. The already-validated `outlook` construction does **not**
carry measurable skill onto single-name `bull_put` timing, once credit
richness is controlled for.

Per the frozen decision rule (§5 of the prereg), this is the honest read of
a transfer test that failed: the factor formulas and cross-sectional
z-scoring that work on 16 broad, comparatively low-idiosyncratic-vol
sector/asset ETFs did not transfer their skill onto a noisier,
single-name population. That is a real, informative answer, not a failure
to find something that was always there — see §1 of the prereg for why this
was framed as a transfer test rather than an assumed-valid import from the
start.

## The primary numbers

| | in-sample (2022-01-07→2024-12-31) | holdout (2020-01-27→2021-12-31) |
|---|---:|---:|
| n (closed bull_put trades) | 102 | 64 |
| raw IC | +0.2696 | +0.1708 |
| raw t_clustered | 2.349 | 1.381 |
| **residualized IC** (credit_pct_width, atm_iv controlled) | +0.2028 | **+0.0523** |
| **residualized t_clustered — the decision statistic** | 1.767 | **0.446** |

The sign stayed positive in both windows, matching §2's predicted direction
(more bullish → better bull_put outcome) — but the magnitude collapsed by
roughly 4x from in-sample to holdout (residualized IC 0.20 → 0.05), and even
the in-sample residualized figure (`t_clustered = 1.767`) never cleared the
pre-registered hurdle on its own. This is the same shape every other feature
in `docs/HOLDOUT_20260809.md` showed before being marked dead: a number that
looks interesting raw, and shrinks hard once credit richness and holdout
discipline are both applied.

**Note on the raw (uncontrolled) numbers:** they look more encouraging
(`t_clustered = 2.349` in-sample) and would have crossed a naive significance
bar. That gap between "raw looks real" and "residualized does not" is
exactly the trap `residual_ic` exists to catch, and exactly why the prereg
locked the residualized measure as the decision statistic before this script
was ever run.

## Coverage, honestly

- 108 candidate entries opened in-sample, 105 closed; 72 opened in holdout,
  69 closed. 44/36 respectively were `skipped_missing` (no fillable chain
  that day) — this is the same friction every other feature in this universe
  measures against, not something specific to this feature.
- `outlook_composite` coverage: 135,960 (symbol, date) scores computed
  across the full universe, from 117 of 120 symbols with cached close
  history (`data/dolt_options.db::stocks_close`) — 3 sub-industry SPDR ETFs
  (`XHE`/`XME`/`XPH`) were never fetched and were skipped rather than
  triggering a live network call from this script.
- **n=64 holdout trades is thin.** The prereg did not set a formal minimum-N
  floor for this design (unlike the gate-RD prereg's 30-symbol-day-cluster
  floor) — worth naming as a limitation of this specific registration's
  design, not something that changes the verdict: the point estimate itself
  (residualized IC +0.05) is small in absolute terms, not merely
  imprecisely measured around a larger one.

## What this does not show

- **This is a `bull_put`-only result**, per the prereg's own scope (§3): the
  mechanism argument for why short premium fits this signal does not extend
  to `long_call`, and no `long_call` number was computed here.
- **This does not re-open `src/outlook`'s own validation** on its native
  16-ETF universe — that result (66-72% right, IC +0.05-0.08) stands
  unchanged; this is a statement about transfer to a different, harder
  population, not a re-measurement of the original.
- **A re-tuned version of this construction, fit directly on single names,**
  is explicitly out of scope per the prereg's §7 — that would be a different
  hypothesis requiring its own registration, not a patch to this one.

## Next step

`outlook_composite` is documented as not transferring to single-name
`bull_put` timing. Per the prereg's decision rule, this specific
registration is closed — re-running it without a materially different
design (a different structure, a re-fit construction, or a different
population) would not be a fresh test, it would be re-litigating a closed
result. `src/outlook/` itself is unaffected: it remains validated and
display-only on its original 16-ETF universe.
