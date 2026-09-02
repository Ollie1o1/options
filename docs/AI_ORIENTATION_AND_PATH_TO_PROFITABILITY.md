# Orientation for the next AI session, and where profit could actually come from

Written 2026-09-02, after a session that merged PRs #80–#89.

Read this before proposing anything. It exists so you do not spend a session
re-deriving what is already known, re-running tests that are already resolved,
or building machinery whose input data does not exist.

Everything numeric here is reproducible with the snippets given. If a number
here disagrees with what you measure, **trust your measurement and update this
file** — but first check you are applying the same filter, because most
disagreements in this repo have turned out to be different denominators rather
than different facts.

---

## 1. The one-paragraph truth

This system has **no measured edge**. It is a well-instrumented measurement
apparatus attached to a book that cannot yet be distinguished from noise. The
binding constraint is **statistical power**, not features, not signals, and not
code quality. Almost everything that has been tested has come back "no
evidence", and the honest response has been to refuse rather than to rank. Your
instinct will be to add a signal. Resist it. The highest-value work available is
either (a) reducing variance so existing questions become answerable, or (b)
answering a question the existing data already supports and nobody has asked.

---

## 2. Measured state, with exact filters

### The book

```python
import sqlite3, numpy as np, collections
c = sqlite3.connect('paper_trades.db')
rows = c.execute("""SELECT pnl_usd, substr(date,1,10) FROM trades
  WHERE status='CLOSED' AND pnl_usd IS NOT NULL AND capital_at_risk>0
  AND COALESCE(paper_only,0)=0""").fetchall()
```

| quantity | value |
| --- | --- |
| closed trades (all) | 927 |
| closed, `paper_only=0` | 752 |
| entry-day clusters | **75** |
| win rate | 50.3% |
| profit factor (dollar) | **1.111** |
| PF 95% CI, bootstrapped on entry-day clusters | **[0.785, 1.571]** |
| total P&L | +$12,065 |

**The CI contains 1.0.** With 75 clusters this book cannot distinguish a 25% edge
from a 25% bleed. That single fact should govern your priorities.

Per strategy (closed, non-`paper_only`):

| strategy | n | P&L |
| --- | ---: | ---: |
| Long Call | 256 | +$7,606 |
| Bull Put | 146 | +$5,290 |
| Iron Condor | 57 | +$2,758 |
| Long Put | 49 | +$4,581 |
| Bear Call | 135 | +$108 |
| Short Put | 109 | **−$8,278** |

Short Put's −$8,278 **is not a verdict**: 93 of 109 exceed the $4,000 capital
cap; inside the cap n=16 and +$173. It is off for *absence of evidence*, not
proven failure.

### A number discrepancy you must not paper over

Project memory records **PF 1.044, CI [0.87, 1.24] on 900 closed**. This file
measures **PF 1.111, CI [0.785, 1.571] on 752**. Both may be right — they use
different filters, and my CI is **clustered on entry day** (75 clusters) while
the older one appears not to be, which is why mine is much wider. Clustering is
the honest choice here; see §6. **Always state the filter and the clustering
with any P&L number in this repo.** Most of its historical errors have been a
ratio quoted without its denominator.

### Open positions

**4.** Three Long Put, one Bull Put. Any "portfolio risk engine" proposal must
survive the question: *what does it constrain, on four positions?*

### Real execution data

**Zero.** No fills database exists. `src/execution/slippage.py` records
real-vs-paper fills and **has never been used**. This single absence blocks all
honest cost calibration (§5).

---

## 3. What has been ruled out — do not redo this work

Each of these consumed real effort. Re-running them is not neutral: it burns
looks at the data and invites the multiple-testing problem the DSR machinery
exists to correct.

| Question | Verdict |
| --- | --- |
| Does the composite score order outcomes? | **No.** Purged walk-forward, every strategy p ≥ 0.46 |
| Does `ev_net` order gate survivors? | **No.** Pre-registered test, failed at n=2137. **Single look spent. Do not re-run.** |
| Does news sentiment predict? | **No.** IC ≈ 0, *large effects ruled out* |
| Any single-name sector edge? | **No** |
| Catalyst columns predict anything? | **No evidence** on all 3 testable hypotheses; H4 not computable |
| Long Call gate? | **STOP.** n=92, rho −0.132, posterior 4%. Resolved |
| Any entry feature predicts outcome? | **No** (attribution study) |
| `iv_mispricing` (weight 0.05)? | Clustered IC **+0.117, CI [−0.120, +0.341]**. No evidence — but the CI is wide, so *large effects are not ruled out*. Sign flips between row-level and cluster-level. See `docs/IV_MISPRICING_MEASUREMENT_20260831.md` |

Note the distinction the repo draws and you must preserve: **"no evidence of an
effect" ≠ "evidence of no effect."** Only news sentiment has the stronger claim
attached.

---

## 4. What this session changed (PRs #80–#89)

Four live defects, all the same shape — *a number describing something other
than its label*:

1. **The promotion gate's DSR was inflated toward promotion.** It counted rows
   where the neighbouring `tstat_clustered` counted clusters. On 253 rows over
   58 non-overlapping intervals it read **0.997 where the honest value is
   0.474**. `deflated_sharpe` now requires `n_eff` with no default.
2. **A gate condition that could never fire.** `promotion_verdict` gated on
   `result.get("pbo", 0.0) >= 0.5`, but `summarise` never set `pbo`. Green in CI
   only because the test built its own fixture. Removed; replaced with a
   structural guard asserting the gate reads only keys `summarise` produces.
3. **The SVI fitter reported quality for parameters it did not return.**
   `fit_quality` came from the optimiser's iterate while the *projected* params
   were returned — different points, in ~60% of fits. Worst case: **reported
   0.9905 for a fit actually scoring 0.0000.**
4. **A convergence flag discarded 62% of good fits.** `res.success` was a hard
   gate; every rejected fit scored >0.95 against its own data. SVI fit rate went
   **38% → 100%** on realistic slices.

Plus: butterfly wing bound + calendar arbitrage + Q-moments (#85), the measured
spread surface made reachable from `friction()` (#87), and three CI time bombs
(#88, #89) — tests reading a live clock or calendar and calling it a fixture.

**`main` is green and now branch-protected** (3 required checks, enforced on
admins, force-push and deletion blocked). Direct pushes to `main` are blocked;
everything goes through a PR.

### Honest accounting of that work

**Not one trading output changed.** All four alloc structures rejected before
and after. What improved is that the numbers stopped overstating their own
confidence, and one broken component (the SVI fitter) now works. That is real
but narrow. **Do not describe this session as having improved returns.**

---

## 5. What is blocked, and by what

The remaining phases of the original brief are blocked by **missing data, not
missing code**. This is the most useful thing in this document.

| Wanted | Blocker |
| --- | --- |
| Fit market-impact params (κ₀, κ₁) from executed logs | **Zero real fills.** Fitting cost params on *modelled* paper fills recovers the model you assumed — circular, and it would produce a confident number describing nothing |
| Price spreads off the measured surface | **29% of the book is unpriceable.** For spreads `entry_price` is a **net credit, not a leg mid**. The surface returns a *relative* half-spread and multiplies by the mid you hand it. I made this exact error mid-session and produced a confident, meaningless table before catching it |
| Portfolio Greek ceilings, CVaR, Ledoit-Wolf | **4 open positions.** The machinery would bind on nothing |
| Combinatorial purged CV | Measured to **refuse on every strategy** at honest purge + embargo. Builds a maintained path that outputs nothing |
| Kelly sizing | Requires an honest `p̂`. None exists. Kelly on an edge whose CI contains 1.0 sizes noise, and sizes it *up* |

**Two small data-capture investments unblock most of this.** Neither improves
anything today; both are the precondition for the rest meaning anything:

1. **Record per-leg mids at entry.** Unblocks spread cost measurement — 29% of
   the book.
2. **Wire `src/execution/slippage.py`.** Unblocks all cost calibration forever.

---

## 6. How this codebase fails — the recurring shapes

Every significant defect found here has been one of these five. Check your own
work against them before claiming anything.

1. **A number describing something other than its label.** DSR counting rows,
   `fit_quality` describing unprojected params, a t-stat named "sharpe", a
   quality score for a surface that fits nothing. **Ask: what exactly is this
   number *of*?**
2. **Counting rows where the unit is a cluster.** This hit three times in one
   day historically, and twice more this session. Trades sharing an entry day
   share that day's move; overlapping holds share a price path. **Ask what ONE
   OBSERVATION is before counting them.** 752 rows are 75 clusters.
3. **A ratio without its denominator.** PF 1.044 on capital at risk is 0.971 on
   entry premium — the same trades. The apparent edge is partly a *sizing
   artifact*.
4. **A test that constructs the object it claims to check.** The dead PBO gate
   survived for its whole life because the test hand-built a dict `summarise`
   never produces. **Assert against real output.**
5. **A fixture that reads a live clock or calendar.** Three separate instances
   broke `main` in one night: a chain dated off wall-clock against a pinned
   `as_of`; `date.today()` (local) vs `datetime.now(utc)`; and a test that read
   the real macro calendar, so an NFP window opening at 00:00 UTC turned it red
   with no code change. **Anchoring to *a* clock is not enough — it must be the
   same clock the code under test reads.**

Related traps, all real here:
- A **default argument is invisible to AST guards** (the board was ranked by
  `quality_score` for months because `sort_by` had one).
- **A green suite proves less than you think.** Run the app.
- **`from .x import y` inside a function** means patching the importing module
  does nothing — patch the source module. Two of my guards were vacuous for
  exactly this reason before I tested the guards themselves.
- A **guard you have not seen fail** is not a guard. Verify it fails on the
  defect.

---

## 7. Working discipline that is actually enforced

- Run `scripts/test.sh` and **check the exit code**, not the presence of "OK".
- `mypy` is **CI-only** by deliberate policy (venv provenance). Never
  `pip install`, never create a venv.
- venv is `~/.venvs/options/bin/python`, never the project `venv/`.
- `rm`, `git push --force`, `git push --delete`, `git branch -D` are **denied**.
  Hand them to the user.
- `main` is branch-protected. Verify a PR's **run conclusion**, not the
  `gh pr checks` summary — a PR once merged six seconds after `mypy` finished
  while both test jobs were still running.

---

## 8. Where profit could plausibly still come from

This is the generative section. Everything above is constraint; this is where to
think. Ranked by expected value per unit of effort, with a falsification test for
each so you cannot fool yourself.

### 8.1 The gate has a control group and nobody has used it — **start here**

`data/candidates.db` records **every candidate the system considered**, with
forward marks:

| | count |
| --- | ---: |
| candidates recorded | 147,086 |
| **refused** (`gate_passed=0`) | **110,552** |
| **passed** (`gate_passed=1`) | **33,001** |
| contracts with forward marks | 11,898 (2026-08-18 → 2026-09-01) |

The standing objection to every ranking test has been: *"entries are drawn at
random among gate survivors, so a cohort selected by rule X cannot test rule X."*
**That objection does not apply here.** The refused population is a genuine
control group, and it is 110k rows.

**The question nobody has asked: does the gate add value at all?** Compare
forward outcomes of passed vs refused contracts. This needs no new trades, no
new signal, and no new data collection.

- **Falsification:** if refused candidates perform indistinguishably from
  survivors, the gate is decoration and its complexity is a liability.
- **Watch for:** clustering (one observation = one symbol-day, not one
  contract), survivorship in which contracts get marked, and the fact that marks
  span only ~2 weeks. Power may be insufficient — **say so if it is** rather
  than reporting a number.

This is the highest-value untested question in the repository.

### 8.2 Cost reduction is a deterministic edge — no prediction required

You cannot reliably predict which contract wins. You *can* reliably pay less to
trade it. Measured facts:

- Crossing the market costs **~27% of credit** (execution truth study).
- The per-strategy constant **undercharges friction 2.28×–3.04×** on single-leg
  trades vs the measured surface.
- 29% of the book is unpriceable, so the true figure is unknown for spreads.

If gross PF is ~1.11 and friction is materially understated, **the book may
already be net-negative and the measurement is hiding it.** Conversely, every
basis point of friction removed is an edge that requires no forecast.

- **Concrete levers:** limit-order placement inside the spread, avoiding the
  first 15 minutes, contract selection biased toward tight-spread cells of the
  measured surface, fewer legs.
- **Falsification:** reprice the closed book under the surface (the machinery
  exists — `reprice_pnl_pct`) and report PF with honest costs. If PF drops below
  1.0, that is the single most important number in the system.

### 8.3 Exits, not entries

Every ranking study here has tested *selection*. Exits are rule-based
(`time_exit_dte`, take-profit, stop-loss) and have never, to my knowledge, been
optimised out-of-sample. **Same entries, different exits** is a large unexplored
axis, and it does not require predicting anything at entry.

- **Falsification:** replay the closed book under alternative exit rules using
  existing marks. Deflate by the number of rules tried (`deflated_sharpe` with
  `n_eff` — the machinery is there and now honest). If the best rule's DSR is
  below 0.5, the sweep found noise.
- **Trap:** this is a large search space over one dataset. It is exactly the
  setting DSR exists for. Count your trials honestly.

### 8.4 Reduce variance to make questions answerable

With 75 clusters and a PF CI of [0.785, 1.571], **nothing can be concluded**.
More signals will not fix this; more *independent observations* will.

Levers that raise effective n without needing any edge:
- more symbols (lower cross-sectional correlation)
- more distinct entry days (75 days over ~4 months is sparse — the launchd fix
  already helped; verify actual vs scheduled runs)
- smaller size per trade, more trades
- lower concurrency correlation (`max_concurrent=3` starves the sample at
  ~36 trades/yr; the alloc report already flags this)

**This is the highest-leverage structural change available**, because it makes
every other question in this document answerable sooner.

### 8.5 Survival is worth more than edge when edge is unproven

Phase 3's real value is not profit — it is **not dying**. A book with no edge and
controlled tails bleeds slowly and stays alive to be measured. One with
uncontrolled tails ends the experiment. CVaR and Greek ceilings do not require
predictive power to be worth having.

Caveat honestly: with **4 open positions**, this constrains almost nothing
*today*. It becomes valuable in tandem with §8.4 (more, smaller positions).

### 8.6 Untested axes, lower confidence

- **Regime conditioning.** VIX regimes exist and are recorded. Has edge been
  tested *conditional* on regime with proper clustering and deflation? Beware:
  slicing multiplies trials.
- **Entry timing within the day.** Only 39% of entry windows fired before the
  launchd fix. Is there a measurable outcome difference by window?
- **Tenor and universe coverage.** Cost thresholds are calibrated on DTE 10–67
  only (`src/cost_calibration.py`); past 250 DTE the friction gate silently
  becomes the dominant filter. Edge outside the sampled band is *unmeasured*,
  not absent.
- **Prediction-market archive.** Kalshi data is being collected
  (`src/predmarkets/`), archive-only, never cross-referenced. Only 36% of
  contracts have a usable mid. Speculative, but genuinely untouched.

### 8.7 The uncomfortable possibility

Take it seriously: **there may be no retail-accessible edge in this universe at
this cost level.** The evidence so far is consistent with that. If true, the
correct outcome is not a better model — it is knowing, cheaply and quickly, so
the capital and attention go elsewhere. This system's real product may be a
*fast, honest verdict*. §8.4 is how you get one sooner. Treat "we measured it and
there is nothing" as a successful outcome, not a failed session.

---

## 9. Suggested order of work

1. **§8.1** — test the gate against its refused control group. Uses existing
   data. Highest value.
2. **§8.2** — reprice the closed book under measured costs and publish honest
   PF. May be the most important number here.
3. **§5** — the two data-capture items (per-leg mids, real fills). Small; unblocks
   everything else. Payoff is months out — say so.
4. **§8.3** — exit-rule sweep, deflated.
5. **§8.4** — variance reduction as a standing structural goal.

Do **not** start with Phase 3 or Phase 4 of the original brief. Both are blocked
(§5), and both would produce machinery that constrains or predicts nothing today.

---

## 10. How to not waste this

- **Measure before you build.** Every genuine finding this session came from
  measuring first; every mistake came from assuming.
- **Report refusals as results.** "Underpowered" is a finding. "No evidence" is
  a finding. A number with an interval that contains zero is not a result to
  quote.
- **State the denominator, the filter, and the clustering** with every number.
- **When you correct yourself, do it plainly and continue.** This document
  contains several of my own corrections because they are load-bearing.
- Read `docs/DSR_PROMOTION_GATE_SPEC.md` and
  `docs/IV_MISPRICING_MEASUREMENT_20260831.md` before touching validation or
  scoring.
