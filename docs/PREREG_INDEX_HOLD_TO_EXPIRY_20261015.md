# Index put spread, one crossing, held to expiry — REGISTRATION

**Frozen 2026-10-15, before this rule was run.** This is the first candidate
rule R under `docs/DECISION_RULE_20261015.md` §3, step 2 (optionsDX).

## Provenance — what has already been seen

Two related numbers on the same data are known. They constrain how much this
test can claim:

| run | rule | result |
|---|---|---|
| `docs/OPTIONSDX_RESULTS_20260811.md` H1 | 25-60 DTE, 25Δ, $5 wide, to expiry | +2.30% RoC/trade, n=1,591, **not delta-adjusted** |
| Task 4 sensitivity (`docs/RESULT_LIVE_BULL_PUT_ODX_STRESS_20261015.md`) | 14-31 DTE, 32Δ, 1.5% wide, credit floor 0.20, to expiry | excess over delta **+0.03% [−2.91, +2.90]** |

Task 4 also showed where the live rule's deficit comes from: two crossings on a
~3-day hold. Refunding them moved the excess to +0.31% [+0.01, +0.62]. That
finding motivates this rule: cross once, then let the option decay. The
parameters below come from convention and from that mechanism. None were chosen
by searching outcomes. The sensitivities are listed now, so they can't be
picked afterwards.

## Rule R

| parameter | value | why |
|---|---|---|
| underlying | SPY, optionsDX 2010-01-04 .. 2023-12-29 | the only index chain history here |
| expiry | nearest listed with DTE in **[30, 45]** | the conventional monthly premium-selling tenor; inside the window where H3 found friction stationary (< 120 DTE) |
| short leg | put nearest **0.25** delta, ±0.10 | conventional; not the live 0.32 |
| long leg | nearest strike **2% of spot** below | scales with price; wider than the live 1.5% to suit a longer tenor |
| credit floor | **none** | task 4's 0.20 floor selected high-vol regimes (2,306 skips, zero trades in 2017). R should trade every regime. |
| friction ceiling | crossing ≤ **25%** of credit | cost control is the point of R |
| exit | **hold to expiry**, settled at intrinsic (`settle`, no exit crossing) | the single crossing |
| fills | entry at the touch | conservative |
| entry | **every trading day** | per-trade question; overlap is handled by clustering |

`StrategySpec`:
- entry `{"dte":[30,45], "short_delta":0.25, "width":5.0, "width_pct_of_spot":0.02, "max_friction_pct_of_credit":0.25}`
- exit `{"hold_to_expiry": true}`
- `max_concurrent` 1000

## Outcomes

These are the same definitions as the task-4 registration, so the two are
comparable:
- RoC = pnl / capital at risk.
- H = (|Δshort| − |Δlong|)·100·(S_exit − S_entry), from the chain's own entry
  deltas and parity-implied spot.
- x = (pnl − H) / capital at risk.
- PF on capital at risk = Σ max(RoC,0) / Σ max(−RoC,0).

**Clustering, a departure from task 4, stated here:** clusters are by
**expiration month**, not entry month. Every trade sharing an expiry rides
nearly the same path to settlement, so they are one observation. Adjacent
expiry months still overlap partly (~35-day holds), so that sensitivity is
reported too. Bootstrap: 10,000 resamples, seed 20261015, percentile 95%.

## Decision (from the decision rule §3, step 2)

R **passes optionsDX** iff BOTH:

- **A′:** mean x, CI lower bound **> 0**;
- **B′:** PF on capital at risk, CI lower bound **> 1**.

If R passes, it may enter a forward paper test (§3 step 3) under its own
registration. If it fails, R is closed, and **no variant of it may be promoted
from the sensitivities below.** A variant would need its own registration, with
this result disclosed as provenance.

## Reported with it (no authority)

- Win rate, mean RoC, per-year table (n, RoC, x, PF, SPY), stress years 2011 /
  2015 / 2018 / 2020 / 2022.
- Fixed-fractional book: 2% of $100k risked per trade, not compounded, with max
  drawdown and recovery.
- **Non-overlapping series:** one trade per monthly expiration, the first entry
  that lands in it. Same A′/B′ statistics. Reported in parallel; it does not
  decide.

## Sensitivities (no authority)

1. Cluster by entry month (task-4 convention).
2. Short delta 0.16 and 0.32.
3. DTE [45, 60].
4. Width 1% and 3% of spot.
5. Credit floor 0.20 restored.
6. Mid fill at entry (half the entry crossing refunded; there is no exit
   crossing).

## Prior

- Mean RoC positive: the window is one long bull market and this is a
  long-delta position.
- x near zero, with A′ more likely inconclusive than passing. Task 4's
  hold-to-expiry sensitivity read +0.03% ± 2.9.
- B′ likely passes on the point estimate but not on the CI.
- Expect R to fail §3 step 2.

Estimator: `scripts/index_hold_to_expiry_odx.py`.
