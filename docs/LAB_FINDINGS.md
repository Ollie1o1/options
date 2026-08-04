# The idea lab, and what it says about single-leg options — 2026-08-04

`src/lab/` exists so an idea can be tested in a few lines instead of a few days
of logging. This records what it is, the two bugs it caught in its own results,
and the answer it gives to the question it was built for: **can single-leg
options beat buying the index?**

The answer is no, on the evidence available. Details below, including the
places where the answer nearly came out "yes" for the wrong reasons.

## The harness

```python
idea = lambda ctx: core.Entry("long_call", delta=0.40, dte=180, hold_days=21)
res  = engine.run(idea, universe, every_n=10)
```

`ctx` carries spot, realized vol (30d, 252d), 21d/63d momentum and drawdown —
price-derived only. `iv_cache.db` holds 83 days across 251 tickers, which
cannot support a multi-year test, so nothing in an idea may depend on it.

Exit is a fixed hold: no stop, no target, one parameter. It is the least
overfittable exit there is and the cleanest read on whether the ENTRY has edge.

Single leg only, deliberately. Measured on real archived quotes, one crossing
costs **0.7-1.7% of a single leg's premium** against **33% of a two-leg credit
spread's credit** — a spread's friction is denominated in its legs while its
reward is denominated in their difference. See `docs/EXECUTION_TRUTH.md`.

### Three data tiers, and the tier travels with every trade

| tier | source | coverage | IV |
|---|---|---|---|
| real_marks | `data/dolt_options.db` | 9 symbols, <=67 DTE, 2020-2026 | **real** |
| real_marks | `data/chain_archive.db` | 15 symbols, <=120 DTE, 22 days | real |
| modeled | `data/squeeze_prices.db` | 21,244 symbols, 2017-2026 | **modelled** |

Only tier 3 reaches past 120 DTE, and only tier 3 carries an assumption.

## Two calibrations

**Black-Scholes is not the problem.** Repriced 800k real DoltHub contracts
using each contract's OWN real IV: median error -0.3% to -0.5%, and 94-95%
land within 10% of the real mid. Given the right vol, the model is accurate.

**The IV assumption is the whole error budget.** Real ATM implied over trailing
30d realized, n=6,515: median **1.02**, but p10-p90 spans **0.70x to 1.52x**.
Option prices are near-linear in vol across that range, so a single modelled
number is not a result. `sweep_iv` runs the band.

## Two bugs this caught in its own output

**1. Short-side sign error.** The engine reported short puts winning 21% of the
time and losing 100% at every risk level. A short put on a dead-flat series
came out at -77%. Two errors: P&L was computed as debit minus credit rather
than credit minus debit, and the return was divided by the closing debit
instead of the collateral tied up. Fixed; short puts now return ~0% CAGR at
69-82% win rates, consistent with the repo's existing short-premium findings.

**2. `data/squeeze_prices.db` stores RAW, unadjusted closes.**

```
NVDA 2024-06-10  -89.9%     AMZN 2022-06-06  -94.9%
GOOG 2022-07-18  -95.1%     AAPL 2020-08-31  -74.2%
TSLA 2020-08-31  -77.5%,  2022-08-25  -66.8%
```

Splits, not crashes. Unadjusted they destroy long calls and manufacture long
puts — which is almost certainly why long puts looked like the best thing on
the board in-sample (+11.9%/yr). `data.adjust_splits` back-adjusts by ratio
detection (no corporate-actions table exists for a 21k-symbol universe), with
the threshold set at -45% so genuine crashes are left alone. Before the fix
NVDA read -2.4%/yr over 2020-2026; after, +74.2%/yr.

`data/equity_ohlcv.db` raises `disk I/O error` on every read including
`.backup`, and is treated as absent.

## What the search found

### Long puts: the holdout earned its keep

Best in-sample thing on the board, negative out-of-sample in **every** config.

| | TRAIN 2017-2022 | TEST 2023-2026 |
|---|---:|---:|
| long_put 180 DTE / 21d | **+11.9%/yr** | **-4.5%/yr** |
| long_put 365 DTE / 45d | +7.0%/yr | -2.7%/yr |

Train contains COVID and the 2022 bear; test is a bull market. Without a time
split this would have been reported as the answer. (Both periods also carry the
split bug, which inflates puts further.)

### Long calls: real per-trade, but it is leverage, not alpha

Three findings that each looked like an edge and were not:

1. **Universe selection.** 15 hand-picked megacaps gave a 53.8% win rate at 365
   DTE. A random 400-symbol draw from the 6,058 names with enough history gave
   **34.7%**. Nineteen points was hindsight in the symbol list.
2. **Beta.** Leveraged long beats unleveraged long in any bull market. Against
   leverage-matched buy-and-hold with drawdown, the margin mostly disappears.
3. **The delta tell.** Returns rise monotonically as delta falls (0.25 ->
   +14.1%/+39.2%; 0.70 -> +4.6%/+19.0%). An edge that grows monotonically with
   leverage is beta wearing a costume.

The one configuration that beat leveraged SPY risk-adjusted in both windows —
180 DTE calls held 21 days — does not survive its own robustness checks. It
wins on 3 of 5 underlyings (MSFT loses both windows), and at the p90 end of the
measured IV band its TRAIN CAGR is **-1.1%** against SPY's +9.0%.

### The decisive test: real marks, real IV, no model

DTE <=67, 8 megacaps, 2020-2026, bought at the ask and sold at the bid.

| config | n | win% | mean | median | ann @10% risk |
|---|---:|---:|---:|---:|---:|
| 45 DTE / 21d hold | 196 | 43.9% | +9.3% | -22.0% | **+6.1%** |
| 60 DTE / 21d hold | 236 | 43.2% | +8.3% | -25.7% | +4.7% |
| 45 DTE / 30d hold | 218 | 38.5% | +14.0% | -52.6% | +6.3% |

Against the same window, split-adjusted:

| | CAGR |
|---|---:|
| Those 8 names, equal-weight buy-and-hold | **+30.1%/yr** |
| SPY alone | **+13.9%/yr** |
| Best long-call config, real marks | **+6.3%/yr** |

**Long calls lose to SPY by 7.6 points a year and to their own underlyings by
23.8, on real quotes with no modelling anywhere.** The modelled version of
approximately this trade said +30.7%/yr — the model was flattering it by ~5x,
the same failure mode as the mid-fill assumption in `docs/EXECUTION_TRUTH.md`,
caught this time by the tier separation.

## Bottom line

Nothing tested here beats buying the index. The single-leg friction advantage
is real and large, longer-dated genuinely does reduce the friction burden, and
both are worth having — but neither creates edge on its own, and no entry
filter tried (realized vol high or low, 3-month momentum either way, drawdown,
delta) changed that.

What has NOT been tested, and is where the remaining hope lives: the tier-1
corpus stops at 67 DTE, so **every claim about genuinely long-dated options
rests on modelled IV**, and the modelled tier has now been caught overstating a
result by 5x. Until there is a source of real long-dated marks, a LEAPS
strategy cannot be honestly validated in this repo — it can only be measured
forward.
