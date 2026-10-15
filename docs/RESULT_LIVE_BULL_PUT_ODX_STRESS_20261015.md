# The live Bull Put rule on optionsDX SPY 2010-2023 — RESULT

Registration: `docs/PREREG_LIVE_BULL_PUT_ODX_STRESS_20261015.md` (frozen in
`724965f`). Estimator: `scripts/live_bull_put_odx_stress.py`.

Run log:
- The first run completed H-A and then crashed in H-B on a column-access bug
  (`df.take` is a pandas method). Fixed in `36de351`.
- The full rerun matched H-A to every printed digit.
- `data/optionsdx.db` sha256 begins `c28c463b44434e6a`; 3,500 dates.

## Rule as replayed

- 927 closed trades over 114 entry months.
- Median DTE 15, median hold 3 days, short delta 0.323, width 1.51% of spot.
- Exits: 612 time, 220 take-profit, 95 stop.
- Engine skips: 2,306 below the 0.20 credit floor, 206 with no legs, 24 on
  friction, 37 missing or crossed quotes.

**The credit floor decides when this rule trades.** At 32 delta and 1.5% wide,
SPY credit clears 20% of width mostly when vol is elevated. So 2017 has no
trades, 2014 has 3 and 2021 has 8, while 2022 has 211. "Index under the live
rule" means index in higher-vol regimes.

## H-A (primary): COSTLY BETA

| per trade, on capital at risk | mean [95% CI, month-clustered] |
|---|---|
| **excess over own entry delta** | **−2.10% [−2.72%, −1.59%]** |
| raw RoC | −0.30% [−2.20%, +1.54%] |
| delta-equivalent | +1.81% [+0.33%, +3.31%] |
| PF on capital at risk | 0.958 [0.740, 1.265] |
| win rate | 59.9% |

**Fixed-fractional book** (2% of $100k risked per trade, not compounded):
- −5.5% over 14 years;
- max drawdown 19.6% of notional, from a peak on 2018-12-03 to a trough on
  2022-10-10, **never recovered** by 2023-12-29.

| year | n | win | RoC | excess | PF | intra-year DD | SPY |
|---|---|---|---|---|---|---|---|
| 2010 | 101 | 62% | −0.6% | −1.9% | 0.89 | 4.0% | +11.0% |
| **2011** | 115 | 48% | −3.8% | −2.8% | 0.58 | 11.6% | −1.1% |
| 2012 | 97 | 65% | +3.1% | −1.2% | 1.84 | 2.0% | +11.6% |
| 2013 | 36 | 81% | +6.6% | −0.6% | 3.72 | 1.3% | +26.3% |
| 2014 | 3 | 67% | −2.9% | −5.3% | 0.75 | 0.7% | +12.4% |
| **2015** | 42 | 71% | +6.2% | −1.3% | 3.36 | 0.7% | −0.7% |
| 2016 | 57 | 61% | −0.6% | −2.1% | 0.92 | 2.6% | +11.2% |
| 2017 | 0 | — | — | — | — | — | — |
| **2018** | 29 | 59% | −4.3% | −3.6% | 0.60 | 5.7% | −7.3% |
| 2019 | 22 | 73% | +4.8% | −1.8% | 1.96 | 1.6% | +28.8% |
| **2020** | 106 | 60% | −2.2% | −4.0% | 0.76 | 7.2% | +15.3% |
| 2021 | 8 | 88% | +10.7% | +1.4% | — | 0.0% | +28.8% |
| **2022** | 211 | 53% | −2.7% | −2.1% | 0.71 | 12.4% | −20.0% |
| 2023 | 100 | 62% | +1.9% | −1.2% | 1.32 | 3.5% | +24.7% |

Excess is negative in 12 of 13 trading years. The exception, 2021, has 8 trades.

## H-B: NO SEPARATION

The gate's flat-RV EV verdict, recomputed on each trade's entry chain:
- 919 gated rows; TAKE 27.7%; gross ≤ 0 on 58.3%.

| arm | n | RoC | excess |
|---|---|---|---|
| TAKE | 255 | −1.03% [−4.74, +2.35] | −1.93% [−3.31, −0.78] |
| refused | 664 | +0.17% [−1.64, +2.01] | −2.12% [−2.59, −1.68] |

**refused − TAKE (excess): −0.20% [−1.29%, +1.07%] → no separation.** Both arms
lose to their delta by about the same amount. The gate neither rescues nor
condemns index Bull Puts under this rule.

**Mechanism cut** (no authority). Mean excess by skew-penalty quintile:

| quintile | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---|---|---|---|---|
| mean excess | −2.25% | −1.21% | −1.78% | −1.97% | −3.13% |

It is not monotone. The highest-penalty quintile is the worst, if anything.
**Task 1's suspicion that the skew penalty is spurious is NOT supported.** The
penalty doesn't separate, but where it is largest the trade is not better.

## Sensitivities — where the loss comes from

| variant | n | RoC | excess | reads |
|---|---|---|---|---|
| **mid fills** (both crossings refunded) | 927 | +2.12% [+0.44, +3.79] | **+0.31% [+0.01, +0.62]** | premium exists |
| DTE [8, 31] | 812 | +0.39% [−1.80, +2.60] | −1.77% [−2.35, −1.21] | costly beta |
| **hold to expiry** | 927 | +5.99% [−0.51, +12.04] | **+0.03% [−2.91, +2.90]** | inconclusive |
| 25 delta | 28 | +2.53% [−8.05, +10.02] | −0.49% [−3.67, +1.61] | inconclusive (credit floor leaves 28 trades) |

Read together:

1. **The deficit is the round trip.** Refunding the two spread crossings moves
   excess from −2.10% to +0.31%, i.e. 2.4 points of capital at risk per trade.
   A 3-day hold pays the spread twice for three days of theta. Mid fills are not
   achievable; the live `limit` policy recovers some of it, not all (paper
   execution truth ≈ 27% of credit). So the premium that exists at mid is
   smaller than the cost of harvesting it this way.
2. **Holding to expiry removes the exit crossing but finds no premium either.**
   Excess is +0.03% with a CI of ±2.9. Its raw +6.0% RoC is delta: SPY rose
   about 4x over the window.
3. **Tenor, not delta, is the live rule's problem.** Shorter DTE is just as
   bad. The 25-delta variant can't be judged, because the credit floor leaves
   only 28 trades.

## What this does and does not say

- **Says:** the live Bull Put rule, run on the index, is a costly way to be long
  SPY, and the cost is the exit crossing on a ~3-day hold. That matches task 3's
  sign on the single-name book (−4.0%, inconclusive). It is the basis of the
  live-rule veto in `docs/DECISION_RULE_20261015.md` §1.
- **Does not say:** that index premium selling has no edge. At mid there is a
  small one (+0.31%, CI barely above 0). Held to expiry the estimate is ~0 with
  wide error. Any new rule must be registered before it is run (decision rule
  §3). The obvious candidate is longer holds with a single crossing; it has NOT
  been tested here beyond this sensitivity.
- **H-B:** the negative-EV gate is not what kills index Bull Puts' value. Even
  the trades it would TAKE lose to their delta.

No config change.
