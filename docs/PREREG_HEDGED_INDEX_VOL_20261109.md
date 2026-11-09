# Delta-hedged index premium — REGISTRATION

**Frozen 2026-11-09, before any of these rules was run.** New candidates for
`docs/DECISION_RULE_20261015.md` §3 step 2 (optionsDX).

## Why a new mechanism

Four measurements (live book, live rule on SPY, rule R, R's best variant) all
landed at or below their delta-equivalent: on this data a put spread is a way to
hold delta and adds no premium beyond it
(`docs/RESULT_INDEX_HOLD_TO_EXPIRY_20261015.md`). The tests below remove the
delta instead of benchmarking against it. If a variance risk premium (VRP)
exists in SPY options after costs, a delta-hedged short-vol position is where
it shows up on its own. An earlier Dolt result (SPY delta-hedged short straddle,
Sharpe +1.26) is known and caveated (Dolt data defects pre-08-14); it is the
only prior seen and motivated this test. No optionsDX outcome of any hedged
structure has been looked at.

## Common setup

- Data: SPY, optionsDX 2010-01-04 .. 2023-12-29.
- **One trade per monthly expiry** (third-Friday expirations, day 15-21). Entry
  = the first trading day on which that expiry has DTE <= 35 (and >= 25). Trades
  do not overlap, so each trade is one observation.
- Options: entry crosses the touch (sell at bid, buy at ask), held to expiry,
  settled at intrinsic using the underlying on the last trading day <= expiry.
  One crossing.
- **Hedge (H-variants):** at each trading-day close t, hold −(position delta)
  shares of SPY, position delta from the chain's own leg deltas (missing quote
  -> carry the leg's last delta; on the entry day use entry deltas). P&L of the
  hedge from close t to close t+1. Cost **$0.01 per share traded** (SPY trades
  at a 1-cent spread; this charges a full cent). No hedge on the expiry day.
- Capital at risk (CAR) = 100 × (max wing width − credit), the defined-risk
  maximum loss of the options. Same denominator for hedged and unhedged.
- r = pnl / CAR. PF = Σ max(r,0) / Σ max(−r,0).
- Bootstrap: 10,000 resamples of trades (iid; trades don't overlap), seed
  20261109, percentile 95%.

## Tests

**T1 — hedged iron condor (PRIMARY).** Short call and put nearest 0.25 |delta|,
long call and put nearest 0.10 |delta|, same expiry; daily delta hedge.
Passes iff mean r CI lower > 0 AND PF CI lower > 1. Delta is hedged, so the
mean is already the excess over delta; no delta benchmark is subtracted.

**T2 — VRP-timed T1.** Signal at entry = ATM IV (mean of call and put IV at the
strike nearest spot, the trade's expiry) − realised vol (std of the prior 21
daily log returns of the chain's underlying, ×√252). Trade only when the signal
is above the median of the signal over all PRIOR entries (needs >= 12 prior;
before that, no trade). Passes iff the filtered set passes T1's test.
Reported alongside: the above-minus-below difference in mean r with its CI.

**T3 — hedged ATM iron fly.** As T1 but short legs at the strike nearest spot
(call and put), long wings nearest 0.10 |delta|. More gamma, more premium,
more hedging. Same pass rule.

Family: three claimants. A pass needs to survive Bonferroni: report each CI also
at 98.33% (= 1 − 0.05/3); **a claim requires the 98.33% lower bounds** to clear.

## Sensitivities (no authority; listed now so they can't be chosen later)

- T1 unhedged (shows what the hedge removes).
- T1 at mid fills (cost ceiling of the mechanism).
- T1 hedge cost $0.02/share.
- T1 short legs 0.16 |delta|.
- By-year table and the stress years 2011, 2015, 2018, 2020, 2022.

## What a pass does and doesn't mean

A pass means a §3 rule exists and moves to the forward step (≥ 8 forward
weeks). It is not authority for real money. A fail on all three closes index
short-vol on this data. Real-world caveats noted in advance: a hedged condor
needs daily share trades and margin for the share hedge; paper fills ≠ real
fills (execution truth ≈ 27% of credit).
