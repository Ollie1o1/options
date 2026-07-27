# Profitability Findings — 2026-07-27

Evidence snapshot taken while investigating why the Phase-1 gate had not fired.
All P&L below is **net** of the exit model's slippage + commissions.

## 1. The gate is statistically unreachable as written

`src/phase1_checkpoint.py` fires READY on `IC >= 0.08 AND p < 0.05`. Those two
arms disagree by an order of magnitude:

| true IC | closed LCs needed for p<0.05 |
|---:|---:|
| 0.076 (observed) | 666 |
| 0.080 (the threshold) | 601 |
| 0.100 | 385 |
| 0.150 | 171 |
| 0.200 | 96 |

At the `n >= 50` trigger, the `p < 0.05` arm actually demands an **observed IC of
≈0.286** — 3.6× the stated threshold. The 0.08 number is decorative; significance
is the binding constraint. Reaching n≈600 is a year-plus of accrual to prove a
statistic that currently reads as zero.

## 2. The gate measures the worst strategy in the book

Closed trades since `phase1_start` (2026-05-27), n=393:

| strategy | n | total USD | win% | scorer IC (p) |
|---|---:|---:|---:|---|
| Long Put | 37 | **+4,787** | 35% | +0.107 (0.53) |
| Bull Put | 68 | **+3,630** | 66% | +0.036 (0.77) |
| Bear Call | 91 | +547 | 66% | **+0.152 (0.15)** |
| Iron Condor | 23 | +206 | 43% | −0.269 (0.22) |
| Short Put | 76 | −2,083 | 50% | −0.160 (0.17) |
| **Long Call** | **98** | **−17,620** | **33%** | **−0.020 (0.85)** |

**Book total: −$10,531.**

Phase 1 gates **Long Call only** — the worst line by an order of magnitude, and
the one where the scorer shows no signal whatsoever. The credit structures
carrying the book are ungated and unmeasured.

**No strategy reaches p<0.05 on scorer IC.** P&L is driven by *which structure
family is deployed*, not by *contract ranking within a family* — and ranking
within a family is the only thing `quality_score` measures.

## 3. The cohort itself is clean

Worth recording so it is not re-litigated. Cohort n=57:

- **DTE contamination is a non-issue.** Only 6 of 57 trades predate the
  `--min-dte 30` floor; excluding them moves IC 0.076 → 0.073.
- **Ticker diversity is healthy** — 38 distinct names across 57 trades.
- **Clustering is mild.** 57 trades on 18 entry days, ICC +0.108, design effect
  1.23 → **effective n ≈ 46** vs nominal 57. Two days (2026-07-09, 2026-07-16)
  have zero return variance — every trade hit the identical exit.
- The `n >= 50` trigger counts **nominal** trades, not effective ones.

Because the cohort is clean, the weak signal is a **real result, not an
artifact**. IC survives no slicing: all 57 → +0.076 (p=0.57); DTE≥30 → +0.073;
excluding time-exits → +0.098 (p=0.52, best case); time-exits only → −0.089.

One open thread: collapsing to day-means gives IC **+0.249** (n=18, p=0.32).
Not significant, but it hints the scorer may rank *days* better than it ranks
*trades within a day*.

## 4. The through-line is the cost wall

Today's NFLX exit: **+56% gross → −1.5% net**. The 50%-of-credit take-profit on
a $1-wide spread does not clear slippage + commissions.

Every negative result in this repo has the same shape:

- long-premium negative-EV outside low-VIX (`docs/INTEL_BACKTEST_FINDINGS.md`)
- leverage signals: cost is the wall, 22× short vs edge
- single-name equity VRP: no edge — but **index** VRP positive
- crypto perp signals: 4 candidates, all dead net-of-cost
- lottery sleeve: negative-EV
- breakout engine: honest no-edge

Everything that has **ever** cleared the wall is short premium on liquid
underlyings. Long premium has now failed in backtest, in VRP research, and in
98 live paper trades.

## 5. Cherry-picking by quality score makes results WORSE

Tested directly (2026-07-27): if instead of logging everything you took only the
top-scored trade(s) each day, how would you have done? Simulated over 32 entry
days, 393 closed trades.

| rule | n | win% | avg USD | total | 95% CI on avg USD |
|---|---:|---:|---:|---:|---|
| top-1 per day | 32 | 43.8 | **−126** | −4,031 | [−308, +35] |
| top-2 per day | 64 | 43.8 | −87 | −5,574 | [−251, +102] |
| top-3 per day | 95 | 45.3 | −79 | −7,526 | [−221, +71] |
| top-5 per day | 155 | 47.7 | −67 | −10,396 | [−180, +50] |
| **all logged** | 393 | **50.4** | **−27** | −10,532 | [−89, +36] |

**Monotonic degradation.** The more selective you are by `quality_score`, the
worse the outcome — being maximally selective (top-1) is 4.7× worse per trade
than taking everything. The same ordering holds on the long-premium side
(top-3 = −$208/trade vs −$95 unselected).

By score bucket, the top of the range is the worst place to be:

| score bucket | n | win% | avg USD |
|---|---:|---:|---:|
| [0.85, 1.00] | 18 | **33.3** | **−165** |
| [0.75, 0.85) | 47 | 42.6 | −112 |
| [0.65, 0.75) | 151 | 52.3 | −66 |
| [0.55, 0.65) | 121 | **57.0** | **+109** |
| [0.00, 0.55) | 56 | 42.9 | −99 |

Long premium at 0.85+ is the single worst cell in the book: **23% win rate,
−$270/trade** (n=13).

**Caveat, stated plainly:** every CI above straddles zero, and n=32 days is
small. No individual cell is significant. What carries the weight is that the
degradation is *monotone across all three panels* (all / long / short) — that is
much harder to produce by chance than any one number.

**Do not chase the [0.55, 0.65) bucket.** It is the best-looking cell out of ~15
slices, which is exactly where a spurious winner appears. It is not a strategy.

The one genuinely positive cell in the entire analysis is short premium taken
**unselected**: 59.3% win, +$9/trade, +$2,301 total.

## 6. There is no better metric — and `pop_score` is a decoy

Scanned all 35 stored metrics (component scores + entry Greeks) against
`pnl_pct`, n=393. Two survived Benjamini-Hochberg FDR correction at alpha=0.05
(Bonferroni threshold p<0.00161):

| metric | n | Spearman IC | p |
|---|---:|---:|---:|
| `pop_score` | 393 | **+0.176** | 0.00046 |
| `entry_vega` | 393 | +0.148 | 0.0032 |
| `quality_score` | 393 | −0.077 | 0.127 |

**`pop_score` does not survive the confound test.** Its mean is rank-ordered by
structure family — Long Call 0.287, Long Put 0.323, IC 0.374, Short Put 0.458,
Bull Put 0.539, Bear Call 0.544 — so aggregate POP is just re-discovering
"credit beats debit". Within strategy it predicts nothing, and in Short Put it
is significantly **backwards**:

| strategy | n | pop IC (p) | quality IC (p) |
|---|---:|---|---|
| Bear Call | 91 | +0.055 (0.61) | +0.201 (0.056) |
| Bull Put | 68 | −0.103 (0.40) | −0.031 (0.80) |
| Iron Condor | 23 | −0.190 (0.39) | −0.299 (0.17) |
| Long Call | 98 | +0.144 (0.16) | −0.077 (0.45) |
| Long Put | 37 | +0.180 (0.29) | +0.173 (0.31) |
| Short Put | 76 | **−0.305 (0.007)** | −0.231 (0.045) |

`entry_vega` is the same artifact (vega differs systematically by structure and
DTE). `corr(pop_score, quality_score) = 0.071` — the composite barely reflects
probability of profit at all.

**Conclusion: nothing in this system predicts within a strategy.** The only
reliable signal in the database is which structure family is being traded.
`quality_score` is usable as a hygiene *filter* (exclude illiquid/wide/garbage);
it is not usable as a *ranker*.

## 7. The 700 CAD constraint decides the strategy

700 CAD ≈ **511 USD**. Capital required per single unit, from logged trades:

| strategy | median capital | affordable at 511 USD |
|---|---:|---:|
| Bear Call | $54 | 100% |
| Bull Put | $128 | 96% |
| Short Put | $488 | 51% |
| Long Put | $580 | 43% |
| Long Call | **$710** | 31% |
| Iron Condor | **$1,380** | 13% |

The Long Call median position exceeds the entire account. At this size the book
is effectively restricted to **Bull Put and Bear Call verticals**.

That is also where fixed costs bite hardest. Round-trip cost (0.05/share slip x2
+ 0.65/contract comm x2) is **$22.60 for a 2-leg vertical** regardless of size:

| structure | median credit | cost if closed at target | cost if held to expiry |
|---|---:|---:|---:|
| Bear Call | $47.50 | **48% of max profit** | 24% |
| Bull Put | $122.50 | **18% of max profit** | 9% |

Two consequences that follow directly:

1. **Bull Put dominates Bear Call at this account size** — the same $22.60 cost
   is 18% of a Bull Put's credit vs 48% of a Bear Call's. Nothing to do with
   edge; purely credit-per-trade vs a fixed toll.
2. **Holding to expiry instead of closing at the 50% take-profit halves the
   cost**, because only the opening legs are paid. This is the cheapest
   available improvement and requires no new signal.

This is the mechanism behind the NFLX +56% gross to -1.5% net exit.

## Bottom line

Real money stays OFF — that call remains correct. The binding problem is not a
weak scorer to be tuned; it is that the measurement budget is pointed at
contract selection in the one structure family that loses money.
