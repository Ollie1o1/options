# Delta-matched benchmark for the Bull Put book — RESULT

Registration: `docs/PREREG_DELTA_MATCHED_BENCHMARK_20261015.md`, frozen in
`1cec5a5` and amended in `f4c73c0` before computation. Estimator:
`scripts/delta_matched_benchmark.py`, run once as registered. The population is
the 59 closed sized-era Bull Puts (2026-08-20 → 10-15): 28 symbols, 9 entry
weeks, 18 using the S₀ fallback (16 with no leg quotes, 2 that failed the 3%
check).

## The dollars

| | |
|---|---|
| book realised P&L | **+$877.50** |
| holding each trade's entry delta in its underlying, same window | **+$1,552.95** |
| excess (the premium-selling component) | **−$675.45** |
| capital at risk, summed | $30,532 |

**Every dollar the book made, and more, is explained by being long delta in a
rising market.** The part the strategy exists to earn — theta and vega, net of
short gamma and costs — lost $675.

## Per trade, on capital at risk (week-clustered bootstrap, 9 clusters)

| | mean [95% CI] |
|---|---|
| book | +0.84% [−4.20%, +6.91%] |
| delta-equivalent | +4.86% [+0.40%, +9.25%] |
| **excess** | **−4.02% [−9.61%, +1.64%]** |

**Primary verdict: INCONCLUSIVE.** The CI on the excess contains zero.

Sensitivities (no authority):
- clustered by symbol (28): −4.02% [−8.74%, +0.59%], inconclusive
- dropping S₀-fallback rows (n=41, 6 week-clusters): −3.69% [−6.54%, −1.38%],
  reads **costly beta**

Pairing works here, unlike the SPY benchmark: corr(book, delta-equivalent) =
0.76 per trade, sd 0.247 → 0.160. Netting out the trade's own delta removes
about a third of the noise.

## Reading

- The point estimate says the book is a costly way to be long. The
  premium-selling residual is about −4% of capital at risk per trade, roughly
  −$11 per trade.
- That fits short gamma: holds are ~3 days, and the underlying moves over them
  were large (NFLX +6.7%, COIN +10.8%, INTC −6.9%) relative to 3 days of theta
  on single names.
- It is not yet significant, so it must not be quoted as "the book loses to its
  delta".
- Caveats:
  - The hedge is static at the entry delta, and exits are at the ledger's
    recorded prices.
  - The delta-equivalent stock position needs far more capital than the
    spread's max loss. This compares P&L for the same market exposure, not
    return on the same cash.
  - One trade (1053, AMD, +$340 with the stock down 1.7%) supplies most of the
    fallback group's offset.

## For the 2027-01-15 decision rule (task 5)

"Beats the delta-matched benchmark" now has a definition and an estimator. On
today's data the book does not meet it: the point estimate is negative and the
CI upper bound is +1.6%. Re-run the same script on the same registration at the
decision date; do not redefine it.
