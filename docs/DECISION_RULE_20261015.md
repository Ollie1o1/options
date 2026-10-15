# Decision rule for the screener's trading role

**Fixed 2026-10-15, before any data after this date exists.** Task 5 of the 10-15
strategic reset. The user agreed the thresholds on 2026-10-15.

Amendments are allowed only as dated notes in this file, made **before**
2027-01-15, and only in the stricter direction. Nothing here may be re-tuned
after the evaluation is run.

## What is already known

| source | result |
|---|---|
| Task 1 (EV gate) | The spread EV prices both legs at one flat realised vol, so put skew counts against the seller; index Bull Puts pass 5.2% vs 13.2% for single names. |
| Task 2 (shadow corpus) | Refused index vs passed single-name: inconclusive, underpowered (CI ±10% of capital at risk). |
| Task 3 (live book) | +$878 book vs +$1,553 from holding its own delta. Excess −4.02% of capital at risk per trade, CI [−9.61, +1.64], inconclusive. |
| Task 4 (optionsDX SPY 2010-23, the live rule) | Excess **−2.10% [−2.72, −1.59]**, 927 trades, 114 months → **costly beta**. PF on capital at risk 0.958 [0.74, 1.27]. The gate doesn't separate (−0.20% [−1.29, +1.07]). Refunding both crossings gives excess +0.31% [+0.01, +0.62]; holding to expiry gives +0.03% [−2.91, +2.90]. `docs/RESULT_LIVE_BULL_PUT_ODX_STRESS_20261015.md` |

## The rule

### 1. The current live Bull Put rule is VETOED for real money

The live rule cannot go to a pilot, whatever its paper record shows. "Live rule"
means the configuration on `main` at 2026-10-15:
- the spread board, plus the friction ceiling 0.25 and the negative-EV gate in
  `refuse`;
- exits TP +50% / stop −100% / time exit at ≤ 21 DTE after 3 days held.

Reason: on 14 years of index data it earns less than its own delta, with the
whole CI below zero. Task 3 shows the same sign on the single-name book. A
paper record from one rising-market quarter cannot outweigh that.

### 2. Evaluation on 2027-01-15 — run, recorded, binding on entries

**Population:** `paper_trades.db` (repo root), with
- `strategy_name='Bull Put'`, `status='CLOSED'`, `date >= '2026-08-20'`,
- `exit_date <= '2027-01-15'`.

| condition | estimator | bar |
|---|---|---|
| **A. Beats its own delta** | `scripts/delta_matched_benchmark.py`, unchanged from its registration (`docs/PREREG_DELTA_MATCHED_BENCHMARK_20261015.md`). Mean excess per trade on capital at risk, entry-week cluster bootstrap. | CI lower bound **> 0** |
| **B. Profitable on capital at risk** | PF = Σ max(rᵢ,0) / Σ max(−rᵢ,0), where rᵢ = `pnl_usd / capital_at_risk`. Same week-cluster bootstrap, 10,000 resamples, seed 20261015, percentile 95%. | CI lower bound **> 1** |

Both are reported whatever the outcome.

**Outcome:**
- **Fail (A or B misses), the expected case:** Bull Put **auto-log entries stop**
  on the first trading day after 2027-01-15. Open positions run to their
  normal exits.
  - The **shadow recorder keeps running**: `data/candidates.db` candidates,
    `candidate_marks`, simulated positions. Any future rule can then be tested
    forward without restarting the corpus.
  - Capital that would back the book is held in a broad index fund. That's a
    user-side decision; no system action.
- **Pass (A and B):** still no pilot of the live rule (section 1). A pass is
  recorded as evidence that the single-name book differs from the index result,
  and becomes the motivation for section 3. Entries continue on paper only.

### 3. The only path to real money

All three, in order:

1. **A new rule R, pre-registered** in its own `docs/PREREG_*.md` before it is
   run anywhere.
2. **R passes on optionsDX SPY 2010-2023**, using the task-4 estimator
   (`scripts/live_bull_put_odx_stress.py`, re-pointed at R's parameters):
   - H-A excess over delta: month-clustered CI lower bound > 0;
   - PF on capital at risk: CI lower bound > 1.
3. **R then passes A and B** (section 2) on its own forward paper record of at
   least 8 entry weeks, evaluated on a date fixed in its registration.

Only then a **pilot**:

| | |
|---|---|
| size | 1 contract per trade; **≤ $1,500 concurrent max loss** |
| length | 30 trades or 3 calendar months, whichever comes first |
| purpose | measure real fills against paper fills on the same trades (paper execution truth ≈ 27% of credit, `project_execution_truth`) |
| hard stop | cumulative realised loss ≥ $1,500 ends the pilot |
| after | the pilot authorises nothing larger by itself. Scaling needs its own rule, written before the pilot ends. |

## What this does NOT change today

No config or code change ships with this document. Auto-logging continues
until 2027-01-15. Commitments already scheduled still run:
- ~11-01 forward gate re-run;
- mid-Nov time-exit-to-expiry check;
- mid-Jan 2027 estimate revisions;
- 10-19 AXTX corporate action.

Their results inform section 3; they cannot amend sections 1-2.

## Prior, stated now

Section 2 fails. At ~150 trades by January, the CI on A is roughly ±3.5 points
of capital at risk, and both point estimates so far are negative. The section-3
path is open but has no candidate rule. On task 4's sensitivities, the only place a premium shows is at mid fills,
where it's +0.31% of capital at risk. That makes cost control (one crossing,
not two) the first thing any section-3 rule has to solve.

## Amendments

**2026-10-15 — Bull Put auto-logging stopped NOW, not on 2027-01-15.** The
user's decision, made the same day: §1 already vetoes the live rule for real
money, so the January evaluation could not change what happens to it.
Stricter than the original, as amendments must be.

Implementation: `auto_log.blocked_strategies: ["Bull Put"]`.
`apply_auto_log_allowlist` drops a blocked strategy before the allocation is
consulted.
- Long Call, Long Put and Short Put admissions are unchanged. Checked on the
  real config: 400 sampled keys each, identical decisions before and after.
- Removing Bull Put from `allocation.eligible_strategies` would instead have
  moved Short Put from 6% to 73% of admissions.
- Open Bull Puts run to their normal exits.
- `data/candidates.db` keeps recording the board and the auto-log queue.
- Revert: `config.bak.20261015-block-bull-put.json`, or remove the entry.

§2 is still run on 2027-01-15, **for the record only**, on the trades closed
by then.
