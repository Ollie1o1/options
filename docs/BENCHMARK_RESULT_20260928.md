# Benchmark Result

- git: `3bc6b4bcfd87bea4a0e9aaaa6d1c051553da251e`
- generated: 2026-09-28T19:54:21.554988+00:00
- seed: 0
- clustering unit: (symbol, entry_date)
- corpus fingerprint: max_exit_date=2026-09-28 10:30:52, terminal_count=968, loaded=968
- rows dropped: {}
- SPY date match: 827 exact, 141 within 4d tolerance, 0 missing (of 968 matched trades)
- SPY stitch validation: 28 common dates, median diff +0.000%, mean diff +0.053%, max abs diff +1.211%, median abs diff +0.076% (threshold +0.500%, OK)

## Headline: cumulative dollar comparison

**One realisation of one path. No confidence interval attaches to this section.**

- matched trades: 968
- book total realised P&L: $1,629.58
- total capital-at-risk (primary convention, sum over trades): $3,371,946.15
- SPY-equivalent P&L, primary convention (same dollars, same entry->exit windows, per trade): $2,701.62
- mean concurrent capital deployed (secondary convention): $113,958.42
- peak concurrent capital deployed: $628,310.00
- SPY return over the full book span: +8.222%
- SPY-equivalent P&L, secondary convention (mean concurrent deployment held in SPY for the whole span): $9,369.19

## Per-trade excess (descriptive; CI shown)

| vs. null | n trades | n clusters | mean book | mean null | mean excess | 95% CI | CI contains zero | variance ratio | corr(book, null) | n needed (80% power) |
|---|---|---|---|---|---|---|---|---|---|---|
| `cash` | 968 | 766 | -0.220% | +0.000% | -0.220% | [-4.149%, +3.676%] | YES | 1.000 | n/a | 520,955 |
| `spy_buy_hold` | 968 | 766 | -0.220% | +0.263% | -0.483% | [-4.404%, +3.421%] | YES | 0.998 | +0.0721 | 107,485 |

`mean excess` = mean(book_ret - null_ret) per trade, `(symbol, entry_date)`-clustered bootstrap CI (`policy_lab.stats.cluster_bootstrap_mean_ci`, n_boot=10000). `variance ratio` near 1.0 means pairing against this null removed essentially no variance (contrast with `policy_lab`, where a working pairing drives this well below 1.0).

## Daily portfolio series (descriptive; CI shown)

- aligned daily observations: 105
- mean book daily return (on deployed capital): -0.348% (sd +10.283%)
- mean SPY daily return over the same days: +0.080% (sd +0.779%)
- mean daily excess: -0.428% (sd +10.178%)
- t-statistic: -0.431
- 95% CI: [-2.398%, +1.542%] (contains zero: YES)
- variance ratio (sd(excess)/sd(book)): 0.990
- n needed to resolve at 80% power: 4,440 trading days

## What this engine cannot tell you

This engine **cannot establish statistically that the book under- or
out-performs a passive alternative.** The per-trade excess confidence
interval contains zero and would need on the order of 100,000+ trades to
resolve at the observed effect size and noise; the daily series would need
on the order of thousands of trading days (years). What it provides instead
is an honest, reproducible point comparison and a permanent fixture for
accumulating forward observations — not a hypothesis test with a pass/fail
bar.

Why pairing does not rescue this the way it does in `policy_lab`: there,
both arms replay the SAME position on the SAME price path, so the
underlying's move cancels exactly in the paired difference. Here the book's
trade and SPY are different instruments on different paths — there is no
shared path to cancel, and the book's per-trade return noise is many times
larger than SPY's return over the same window (see the `variance ratio`
figures below, which stay near 1.0 rather than falling the way a working
pairing would).

The cumulative dollar comparison is different in kind: it is **one
realisation of one path and carries no confidence interval.** It says what
actually happened, not what would happen on average — a description, not
an inference.

Null 3 (a mechanical SPY put spread) is **out of scope for this report.**
The options chain that would price it (`chain_archive.db`) only covers 52
snapshot dates, a small fraction of the closed book's date range, and
building it as though it covered the whole book is the failure this project
exists to avoid.
