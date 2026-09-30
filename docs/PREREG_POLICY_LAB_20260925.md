# Preregistration — Counterfactual Policy Lab, first sweep

**Written:** 2026-09-25, before any sweep has been run.
**Spec:** `docs/superpowers/specs/2026-09-22-counterfactual-policy-lab-design.md`
**Code:** `src/policy_lab/`, branch `feat/policy-lab`

`docs/RESEARCH_PROCESS.md` step 1 requires this document to land before any test
runs. Everything below was fixed in advance. Anything discovered afterwards goes
in the RESULT document, not here.

## Hypothesis

**H1.** At least one exit-policy setting in the preregistered grid produces a
positive paired mean difference against the baseline policy, replayed on the same
recorded price paths under `cross` costs, with a family-wise permutation
`p < 0.05` across the whole grid and a ticker-day-clustered 95% CI excluding
zero.

## Grid (frozen)

`src/policy_lab/policies.py::SHORT_PREMIUM_GRID` (**750** cells) and
`LONG_PREMIUM_GRID` (**320**), as of the commit this document lands in. The grid's
cardinality is the `n_policies` dimension of the permutation test. Expanding
either grid after the first run invalidates the run and requires a new
preregistration.

## Baselines

- **Corpus A** (`data/candidates.db`): `ExitPolicy("corpusA_recorder", 0.50, 2.0,
  None, None)`. Recovered from the data, not assumed: 3,817 recorded take-profits
  have exit/entry p95 = 0.495, and 1,831 recorded stops have exit/entry min =
  2.010. The live book's baseline is deliberately NOT used here — it arms
  `time_exit_dte=21` while Corpus A's median DTE at entry is 17, which would make
  the baseline exit on its first interior point.
- **Corpus B** (`paper_trades.db` ∩ `data/chain_archive.db`):
  `LIVE_BASELINE_SHORT` / `LIVE_BASELINE_LONG`.

## Clustering unit

`(symbol, entry_date)`. Row counts are used for no interval and no statistic.
Corpus A yields 1,618 such clusters for Bull Put; Corpus B yields 215 paths total
(Bull Put 44, Iron Condor 58, Long Call 61, Bear Call 22, Short Put 20, Long Put
10).

## Decision rule

`src/policy_lab/stats.py::policy_verdict`. A knob is promoted only on
`"promote"`. `"underpowered"`, `"insufficient"` and `"reject"` are all
non-promoting and are reported distinctly, because they mean different things.

The multiple-testing guard is a **max-statistic sign-flip permutation test**
(`family_wise_p`), not a deflated Sharpe. The deflated Sharpe is unusable on this
corpus and that is a measured fact, not a preference: `effective_n` — its required
count of non-overlapping holding intervals — is **9** for every strategy
regardless of row count, being fixed by a 26-day window and a 3-11 day holding
period. At n_eff=9 it returns 0.917 with `n_trials=1`, failing the conventional
0.95 bar before any multiple-testing penalty is applied. `dsr` and `pbo` are still
computed and reported as diagnostics so that limitation stays visible.

## Declared in advance

**The expected result is null.** Roughly 25 prior experiments in this repository
have returned null. Nothing about this one changes that prior.

**The detection floor is ~+0.012 of capital at risk per trade** at this grid size
and cluster count, measured before the sweep: a planted +0.008 effect returns
family-wise p = 0.474, a planted +0.012 returns p = 0.000, and a pure-noise grid
returns p = 0.788. **A null result therefore means "no exit-policy knob worth more
than roughly 1.2% of capital at risk per trade survives correction across 750
candidates" — it does not mean "no effect".**

**A single uncorrected comparison already measured during development gave
+0.00445** (recorder baseline vs take-profit at 35%, cluster-bootstrap CI
[+0.00197, +0.00717]). It is recorded here so it cannot later be presented as a
discovery: it is below the detection floor, uncorrected for the grid, and not a
finding.

**Barrier geometry is expected to contribute.** Option prices are bounded below by
zero, and near that barrier a price can only rise, so any early-exit policy gains a
real advantage unrelated to edge. Verified: on an unfloored martingale the
bootstrap gives 0/20 false positives at every noise level; with a floor it gives
11/20 at sd=0.45. A synthetic bounded-walk benchmark calibrated to this corpus
attributed ~8% of the raw effect to geometry. **Every promoted policy must be
reported as a difference-of-differences against that benchmark**, and a raw
take-profit advantage alone is not evidence of anything specific to this book.

**Corpus A's recorded outcomes reproduce at only ~78%** under mid costs (41.6%
under the recorder policy with imputed costs). Causes are structural: `entry_price`
matches the entry-day mark mid in 1 of 1,432 cases, ~10% of exits are priced off
something the marks do not contain, and stop labelling ran on a different cadence.
The paired comparison does not read recorded P&L — both arms replay the same path,
so the basis error cancels — but **Corpus A's own `pnl_pct` must never be quoted as
ground truth**, and the per-exit-reason breakdown must be reported because
stop-loss positions fail reproduction at 46.1%, more than double other reasons.
If that irreproducible slice correlates with outcome, the paired difference could
still be biased; Corpus B is the independent check.

**Every Corpus A cost figure rests on a modelled spread.** 100% of Corpus A's Bull
Put marks carry NULL bid/ask, so crossing cost is imputed at a half-spread of
0.095 — measured on the 46,460 real single-leg marks, and independently
corroborated at 0.0981 by the chain archive's real single-name quotes. It is a
single-leg figure applied to a two-leg round trip, so it likely UNDER-states
friction. A `mid`-cost run may never produce a promotion verdict.

**The `time_exit_dte` cells of 21 and 28 are degenerate on Corpus A** — 94.1% of
positions reach DTE <= 21 by exit and median entry DTE is 17 — so those cells
measure "exit at the first available mark", a real policy but not a time-decay
result.

## Scope limits fixed in advance

- 26 trading days of a single low-volatility regime. SPY's worst drawdown across
  the whole book's life is **-4.49%** against a long-run **-55.2%**. Nothing this
  sweep produces speaks to a volatility event.
- Median path depth 5.75 marks: coarse thresholds are answerable, intraday timing
  is not.
- Short Put has zero terminal priced rows in Corpus A and 20 paths in Corpus B.
- A positive verdict is evidence that a **policy knob** paid within one benign
  regime. It is **not** evidence that the system has a selection edge, and must
  not be reported as such.

## What promotion would authorise

A config-only PR changing `exit_rules` thresholds. Nothing here authorises real
money, changes the auto-log path, or alters any selection rule.
