# The adjustment stack — 2026-08-07

`docs/ABSOLUTE_SCORES_20260807.md` §4 found that the ~20 hand-set constants
applied after the weighted composite carry the whole negative IC of
`quality_score`, while the 27-component composite is flat. This is the follow-up
that asks what the stack is made of, whether that result survives the obvious
confounds, and which half of it is doing the damage.

Reproduce: `scripts/measure_adjustment_stack.py` (read-only, writes nothing).
`--strategies "Short Put"` runs the replication in §5.

---

## 1. What the stack is

Applied in `options_screener.py:2074-2227`, after the composite and before the
display scale.

| condition | effect | |
|---|---|---|
| earnings_nearby (buyer / seller) | −0.02..−0.15 / +0.01..+0.08 | scaled by avg IV crush |
| EarningsPlay & is_underpriced | +0.08 | |
| Trend_Aligned | +0.05 | |
| decay_warning | **−0.20** | also a risk-multiplier flag |
| gamma_ramp | −0.15 | also a risk-multiplier flag |
| sr_warning | −0.10 | also a risk-multiplier flag |
| seasonal ≥0.8 / ≤0.2 | +0.10 / −0.10 | |
| oi_wall_warning | −0.10 | |
| squeeze confirmed / bare | +0.10 / +0.04 | |
| macro MACRO RISK | −0.10 | also a risk-multiplier flag |
| short_interest >20% & long call | +0.05 | |
| div_warning | −0.08 | |
| EarningsPlay & iv_cheap / not | +0.06 / −0.06 | |
| charm (dte<7) / vanna | −0.05 / +0.03 | config |
| macro event (sector-aware) | config | |
| spread 10-15% / >15% | −0.04 / −0.08 | spread is **also** a composite component (6.8%) |
| stale quote | −0.05 | |
| *clip(0,1)* | | |
| prob_profit <0.25 | **×0.60** | multiplicative |
| residual crush (EarningsPlay) | −0.00..−0.06 | additive, *after* the clip |
| risk_flag_count 3 / 4 / 5+ | **×0.85 / ×0.70 / ×0.50** | multiplicative, plus a 0.40 cap |

**Maximum additive swing −1.22 to +0.47 — a span of 1.69 — against a composite
whose observed standard deviation is 0.065.** A single `decay_warning`
outweighs any component; two penalties outweigh all 27 together.

Three structural problems, independent of any measurement:

- **The double count.** All five flags counted by `risk_flag_count` —
  gamma_ramp, decay_warning, earnings_nearby, macro_risk, sr_warning — *also*
  fire as additive penalties. Every one is charged twice, once additively and
  once multiplicatively.
- **Earnings reaches the score from five places**: the nearby penalty, the
  underpriced bonus, the iv_cheap bonus/penalty, the residual crush penalty, and
  as one of the five risk flags.
- **Spread is charged twice**: as a weighted composite component (6.8%) and
  again as a tiered additive penalty. The code comment at the penalty says
  "reduced — spread_score now weighted in composite", so this was known and
  halved rather than removed.

## 2. Two confounds, both corrected

**`pnl_pct` is priced at the mid with no crossing cost.** It is computed against
`entry_price`, and only 35 of 335 long-premium rows have a real fill price
(`fill_policy='limit'`); the other 300 are `fill_source='unknown'`. Wide-spread
trades are therefore flattered by exactly the cost the spread penalty exists to
punish — the confound points the same way as the hypothesis, so it had to be
removed before anything could be claimed.

Restating returns by charging half the spread each way (and charging entry only
where the position expired worthless, since it was never sold) moves the cohort
mean from **−0.95% to −15.62%** and flips `spearman(spread_pct, return)` from
**+0.0152 to −0.0262**. Every number below is friction-adjusted.

**The composite is rebuilt with today's weights**, not the ones in force at
entry. It does not matter: every component is bounded [0,1] and the weights sum
to 1, so reweighting cannot move the composite far.

| weighting | composite sd | stack sd |
|---|---|---|
| live IC-blended | 0.0645 | **0.1004** |
| raw config | 0.0679 | **0.1016** |
| uniform | 0.0490 | **0.0995** |
| theta-heavy (the pre-fix artifact) | 0.0661 | **0.1024** |

The stack moves the score ~1.55× more than every weighted component combined,
under all four.

## 3. The bonuses are the problem

Rank IC against friction-adjusted return, n = 335, plus the mean over the five
expanding walk-forward windows:

| variant | full-cohort IC | mean of windows | windows negative |
|---|---|---|---|
| as shipped (composite + full stack) | −0.0995 | −0.1687 | 5 / 5 |
| **composite only (stack OFF)** | **+0.0038** | **−0.0261** | 4 / 5 |
| composite + penalties only | −0.0291 | −0.0960 | 5 / 5 |
| composite + bonuses only | −0.1029 | −0.1537 | 5 / 5 |

**Bonuses-only is as bad as the whole stack; penalties-only is a third as bad.**
The bonuses touch 57% of rows, the penalties 43%.

Sorting rows by what the stack did to them:

| | n | mean return | median | win rate |
|---|---|---|---|---|
| net penalised (< −0.02) | 118 | −0.1525 | −0.3008 | 33.1% |
| ~neutral | 46 | **−0.0232** | −0.2257 | **39.1%** |
| net bonused (> +0.02) | 171 | **−0.1945** | −0.3761 | **29.8%** |

The more the stack likes a trade, the worse it does. Rows it left alone are the
best of the three.

## 4. The recoverable firing conditions

`score_adjustments` has no history, but three conditions can be recovered from
stored component columns — `catalyst_score` is 0.8 iff earnings_nearby fired,
`pop_score` is `prob_profit`, and `spread_score` inverts to `spread_pct`.

| condition | fired | mean return | win rate |
|---|---|---|---|
| spread >15% (−0.08) | 213 (63.6%) | **−0.1353** | **36.6%** |
| spread 10–15% (−0.04) | 111 (33.1%) | −0.1791 | 25.2% |
| spread ≤10% (no penalty) | 11 (3.3%) | −0.3298 | 18.2% |
| earnings_nearby (−0.02..−0.15) | 11 (3.3%) | −0.0374 | 45.5% |
| *not* earnings_nearby | 324 | −0.1602 | — |
| **prob_profit <0.25 (×0.60)** | **0 (0.0%)** | — | — |

- **The spread penalty is pointed backwards on this sample**, even after being
  charged its own friction: the most-penalised tier has the best mean and the
  best win rate, and the unpenalised tier is the worst. `spearman = −0.0262,
  p 0.632` — no evidence it discriminates in either direction, but certainly
  none that it discriminates the way it assumes. It fires on **96.7%** of logged
  rows, so it is close to a constant shift with one −0.04 step inside it.
- **earnings_nearby is pointed backwards too**, on n = 11. Nothing can be
  concluded from eleven rows; recorded so it is checked again when there are
  more.
- **The `×0.60` low-PoP multiplier has never fired once.** Minimum `pop_score`
  in the cohort is 0.252 — the filters upstream already exclude everything it
  targets. It is dead code that reads as an active risk control.

## 5. Replication on a different weight branch

`Short Put` scores through `premium_selling_weights` (8 components, not 27) but
runs the same adjustment stack. n = 109:

| | rank IC vs friction-adjusted return |
|---|---|
| shipped `quality_score` | −0.0970 (p 0.316) |
| PS composite | −0.0088 (p 0.928) |
| adjustment stack | −0.0692 (p 0.475) |

Residual sd 0.0973 against composite sd 0.0442 — the stack dominates variance
2.2× here too. Same sign, same shape, an independent cohort and a different
weight branch. Individually insignificant; as corroboration, it is the right
sign in the right place.

## 6. What this does and does not license

**It does not license retuning twenty constants by taste.** The evidence is
n = 335 in one cohort, p ≈ 0.07, on the strategy already known to be the worst
thing the screener does, with the stack inferred as a residual rather than
recorded. `score_adjustments` (schema 20) is the instrument that will settle
this properly and it currently holds 0 of 947 rows.

**It does separate the stack into two halves with very different standing:**

- The **bonuses** are pure alpha claims — "this setup is better, add 0.10". They
  are anti-predictive in 5 of 5 windows and account for essentially all of the
  damage. Nothing defends them except that nobody has measured them.
- The **penalties** are a mix. Some are genuine risk controls (stale quote,
  dividend early-exercise, decay) whose job is to keep you out of structurally
  bad trades, not to rank the survivors — **IC is the wrong test for those**, and
  a weak IC is not grounds to delete them.

**Three items need no calibration decision at all**, because they are structural
rather than a question of what a constant should be:

1. The double count — five conditions charged both additively and through the
   risk multiplier. Whichever is right, both is not.
2. Spread charged twice, once in the composite and once additively.
3. The dead `×0.60` low-PoP multiplier.

---

## 7. What was changed as a result

The execution-truth work replaced `sort_values("quality_score")` with
`rank_by_verdict` on the **display** paths. It was never applied to the
**auto-log** path — so the composite still chose which leg per symbol survived
the per-symbol dedup, and which symbols reached the top-N. **Every row in the
ledger was selected by the score this document measures at −0.10.** The cohort
that shows the negative was picked by the thing that shows it.

`rank_single_legs_by_verdict(df, mode)` now orders the single-leg auto-log
path, the single-pick auto-log fallback, and the Premium Selling display path.
Ordering only — every input row is returned, because the allowlist and budget
filters downstream do the dropping and removing candidates here would starve
the forward cohort. Pinned by `tests/test_autolog_ordering.py`.

**A second bug, found on the way.** `candidate_verdict._legs_of` reads the
buy/sell side off `strategy_name`, and scan rows carry only `type` at these
call sites. A Premium Selling short put was therefore priced as a debit **buy**,
which flips `is_credit` and skips *both* the "credit disappears once the spread
is crossed" check and the breakeven-vs-history check — the two gates that matter
most for short premium. The helper labels before ranking. The buyer-mode display
paths (Budget scan, Discovery) were unaffected: `_legs_of` already defaults to
`buy`, which is correct there.

### The spreads and condors auto-log path — now ranked too

Initially left alone, because **iron condor rows carried no per-leg quotes at
all**: `find_iron_condors` emitted strikes, credits and `return_on_risk` and
nothing else, so `_legs_of` refused every condor. Routing them through the gate
would have sunk the whole cohort rather than ranked it. That was a data gap, so
it was closed rather than worked around.

- `find_iron_condors` now carries all four legs' `bid`/`ask` as
  `short_put_*`, `long_put_*`, `short_call_*`, `long_call_*`, plus
  `spread_width` — the wider wing, which is what `max_risk = width − credit` is
  actually measured against and therefore what the breakeven win rate needs.
- `candidate_verdict._FOUR_LEG` / `_legs_of` price a condor as two sells and
  two buys, refusing the whole structure if any one leg lacks a two-sided
  quote — the same rule the two-leg path already used, and for the same reason:
  a structure priced from three real quotes and one guess is not a price.
- `rank_structures_by_verdict` labels each row (`structure_strategy_name`,
  hoisted out of the auto-log block so it is testable) and orders the path.

**Friction is the whole point here.** Four crossings against one credit runs
roughly twice the two-leg burden, which already measured ~33% of the credit on
the logged Bull Puts against 1–4% for a single leg. On the test fixture, a
condor whose friction is **92.3% of its credit** — refused — was being logged
*ahead* of one at **6.2%**, purely because its `quality_score` was higher.

Sizing is unaffected: `capital_at_risk_for_pick` already resolved condors
through the stored `max_loss` path, and returns the identical figure with and
without `spread_width` (checked: 740.0 both ways, matching the row's own
`max_risk`).

**The caveat that remains.** Multi-leg rows score through `spread_scoring` and
`credit_spread_weights` — a different composite that has *not* been measured
against outcomes. So the case for this ordering rests on the cost argument,
which is measured and large, not on the composite being bad here, which is
untested.

### What this is not

**It is not measured.** The change alters *selection*, and the existing ledger
was selected by the rule being replaced, so it cannot be evaluated on it — and
the inputs the new ordering needs (`bid`, `ask`, `ev_per_contract`) are not
stored per trade anyway. Its justification is that `quality_score` is measured
anti-predictive here, and that `rank_by_verdict` is the ordering this repo
already adopted everywhere it had looked. Whether it helps will be visible only
in trades logged from now on.
