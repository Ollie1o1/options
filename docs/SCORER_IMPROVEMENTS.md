# Scorer improvements the backtest evidence supports

**Date:** 2026-08-06
**Basis:** `src/alloc` backtest over 115 symbols x real EOD chains, plus the
ledger re-derivations. Every item names the measurement behind it.

**None of these are applied.** Changing what the screener flags changes the
cohort every gate reads, so each is a proposal with its evidence attached.

---

## The headline mismatch

The scorer's weights and the measured drivers of P&L disagree sharply:

| input | current weight | what the data says |
|---|---:|---|
| `spread` | **0.0768** (6th) | **the single largest driver measured.** −6.76% vs −0.64% RoC, t=−17.84, n=10,363 |
| `iv_rank` | **0.0260** (11th) | the **only** signal showing a monotone relationship: −0.97% → +3.00% across its range |
| trend / downtrend | **absent** | −5.92% selling into a downtrend vs +2.44% overall |
| `vrp` | 0.1755 (1st) | right idea, but single-name equity VRP measured **no edge**; index VRP positive |
| `pop` | 0.0354 | correctly low — `pop_score` is a structure-family artifact, not a predictor |

The two inputs with the strongest measured effect sit 6th and 11th by weight.

---

## 1. Filter on friction as a share of CREDIT, not spread as a share of mid

**Highest expected impact. This is a filter change, not a weight change.**

`config.filters.max_bid_ask_spread_pct = 0.15` caps each contract's spread at
15% of its own mid. That misses the thing that matters, in two ways:

- It is **per contract**, but you cross the spread on *every leg*. A two-leg
  spread pays twice, an iron condor four times.
- It is **relative to mid**, not to the credit received. The credit is what the
  friction has to come out of.

Measured: median per-leg bid-ask **$0.35**, median credit at mid **$90**,
credit actually received **$45**. **Crossing costs 50% of the credit** while
every leg individually passed a 15% test.

**Proposal:** add `max_friction_pct_of_credit` (~0.25) computed on the assembled
structure — `sum(half-spread over legs) / credit`. Reject above it.
`src/alloc/engine.py:crossing_cost()` already implements the calculation.

Sanity check on the current filter: a 15% cap admits most of the liquid stratum
(16.4% median spread) and much of the broad stratum (20.9%), which together
returned **−6.76%**. The tight-spread names sit at **3.6%**.

## 2. Raise the `spread` weight substantially

0.0768 → **~0.20**, making it the largest or second-largest input.

Nothing else measured moves returns as much. Same strategy, same period, only
the universe changed: **−6.76% → −0.64%**, purely on spread tightness.

## 3. Raise `iv_rank`, and gate credit structures on it

0.0260 → **~0.10**, plus a hard `iv_rank_min` of ~50 for credit strategies.

IV rank produced the only monotone series in the study:

| IV rank | RoC/trade |
|---|---:|
| ≤ 30 | −0.97% |
| unconditional | −0.64% |
| ≥ 50 | +2.02% |
| ≥ 70 | +3.00% |

Ordered across a graded parameter, in the direction theory predicts. Not
statistically established (clustered t=0.83), which is why the proposal is a
weight increase and a soft gate rather than a hard dependency.

Note `config.filters.min_iv_percentile = 25` already exists but is far too
permissive — it excludes only the cheapest quartile, where the measurement says
the interesting boundary is around the median.

## 4. Add a trend gate for short puts

**Not currently represented at all**, and it is the largest single avoidance
effect measured: **−5.92%** selling puts into a downtrend against **+2.44%**
overall, surviving split correction.

**Proposal:** refuse (or heavily penalise) bull puts and cash-secured puts on
underlyings trading below their 50-day average. Mirror for bear calls above it.

This is also the cheapest to implement — the screener already computes moving
averages for other purposes.

## 5. Reconsider the 50% take-profit on spreads

`exit_rules.spread.take_profit = 0.5` closes at half the credit, which means
crossing the spread a **second** time.

Measured, holding to expiry roughly halves the total toll — you pay the opening
legs only. Per structure, round-trip vs held:

| structure | n | friction/share | median credit | round trip | held to expiry |
|---|---:|---:|---:|---:|---:|
| Bull Put | 30 | $0.350 | $102.50 | **68% of credit** | 34% |
| Bear Call | 41 | $0.050 | $44 | 23% | 11% |
| Iron Condor | 59 | $0.175 | $964.50 | 4% | 2% |
| Short Put | 25 | $0.100 | $634.50 | 3% | 2% |
| Long Call | 34 | $0.100 | $838.75 | 2% | 1% |

**Method, because an earlier version of this table said 53% for Bull Put.** The
figures are the median of `|entry_price_cross - entry_price_mid|` and the median
of `|entry_price_mid|` per structure, over the 194 ledger trades that recorded
both prices, with **no quote filtering**. The earlier 53% dropped the 4 Bull Put
fills (13% of that bucket) whose crossed price was not a tradeable credit —
which measures the friction of the trades you would have *wanted*, not the
friction the chain offered. Those quotes are real, so they stay in.

`src/strategies/friction.py` computes this live off the ledger and the
`[6] STRATEGIES` desk shows it as a column per setup, so the number here moves
when the ledger does rather than sitting frozen in a doc. Bull Put remains
**5.8× more expensive to cross than Bear Call**, which is the part that matters
and which the flat $0.05/share assumption hid entirely.

**Caveat, and it is a real one:** holding to expiry means accepting assignment
risk and full max loss on breach, and the backtest's weekly cadence cannot
resolve intra-week stop behaviour. This is a proposal to *test on daily data*,
not to apply.

## 6. Show tail risk beside probability of profit

Every configuration measured carries skew between **−1.1 and −2.8**. An 84.9%
win rate at skew −2.80 is not a good trade; it is one whose losses have not
arrived.

The screener displays PoP prominently and nothing about the shape of the loss.
Proposal: show max-loss-to-credit and a tail flag next to PoP, so a 90%-PoP
trade risking 9× its credit reads as what it is.

## 7. Treat `quality_score` as a filter, not a ranker

Already established (`docs/PROFITABILITY_FINDINGS.md`): selectivity by
`quality_score` degraded returns **monotonically**, and the top bucket
[0.85, 1.00] was the worst cell in the book. The screener still sorts by it.

Proposal: keep it for hygiene (exclude illiquid, wide, malformed rows), but rank
the display by something with measured predictive content — friction and IV rank
are the two candidates this work produced.

---

## What NOT to do with this

**Do not apply all seven and re-measure.** That is seven more configurations on
top of 34 already tried, and the deflation bar rises with every one. Pick the
one with the largest measured effect — friction, item 1 — apply it alone, and
see whether it moves the live cohort.

**Do not treat any of this as validated.** The strongest items (1, 2) rest on a
large-sample measurement and are safe. Items 3 and 4 are suggestive
(clustered t below 1.5). Items 5, 6, 7 are reasoning from measurement rather
than direct results.

**Remember the period.** Every positive number above comes from a 2022-2026
window that was a bull market; `bull_put` +2.44% against `bear_call` −9.43% on
identical names says directional exposure is doing work that a variance premium
is not. The *avoidance* items survive that critique; the positive ones may not.
