# Index put spread, one crossing, held to expiry — RESULT

Registration: `docs/PREREG_INDEX_HOLD_TO_EXPIRY_20261015.md` (frozen in
`9155069`). Estimator: `scripts/index_hold_to_expiry_odx.py` (`3034397`), run
once as registered on optionsDX SPY 2010-01-04 .. 2023-12-29.

## Rule as replayed

- 3,157 closed trades in 167 expiration months, all settled at expiry. 20 more
  were closed `ticker_ended` at the end of the data and are excluded, as
  registered.
- Median: DTE 31, hold 31 days, short delta 0.250, width 2.00% of spot,
  credit/width 0.134.
- Engine skips: 230 no legs, 49 friction, 43 missing or crossed quotes.
- With no credit floor, R trades every year, including 2017 (247 trades).

## Verdict: R FAILS decision rule §3 step 2

Clustered by expiration month:

| | estimate [95% CI] | bar | |
|---|---|---|---|
| **A′ excess over own delta** | **−0.82% [−2.98%, +1.32%]** | lower > 0 | **fail** |
| **B′ PF on capital at risk** | **1.338 [0.963, 2.001]** | lower > 1 | **fail** |
| mean RoC | +3.46% [−0.50%, +7.16%] | | |
| delta-equivalent | +4.28% [+1.51%, +6.85%] | | |
| win rate | 87.1% | | |

**R is closed.** Per the registration, no variant below may be promoted from
this run.

## What it says

- **The +3.46% per trade is the delta.** Holding the same delta in SPY over the
  same 31 days earned +4.28%. The option structure added nothing on top, and
  the point estimate says it subtracted 0.8 points.
- **The deficit is real but no longer significant.** Crossing once instead of
  twice took the live rule's −2.10% [−2.72, −1.59] to −0.82% [−2.98, +1.32],
  which is about the size of the second crossing. What's left is a premium of
  roughly zero beyond delta.
- **The left tail is the price of the win rate.** Per year:

| year | n | win | RoC | excess | PF | SPY |
|---|---|---|---|---|---|---|
| 2010 | 154 | 92% | +9.3% | +2.9% | 2.19 | +11.0% |
| **2011** | 167 | 83% | +2.6% | +5.3% | 1.21 | −1.1% |
| 2012 | 175 | 87% | +4.5% | −1.5% | 1.42 | +11.6% |
| 2013 | 239 | 94% | +12.0% | +0.5% | 5.39 | +26.3% |
| 2014 | 223 | 93% | +8.3% | +3.2% | 2.80 | +12.4% |
| **2015** | 246 | 81% | −4.9% | −1.3% | 0.72 | −0.7% |
| 2016 | 247 | 98% | +14.3% | +6.4% | 14.75 | +11.2% |
| 2017 | 247 | 100% | +10.6% | −0.9% | ∞ | +18.6% |
| **2018** | 246 | 72% | −12.4% | −8.3% | 0.44 | −7.3% |
| 2019 | 245 | 91% | +6.2% | −4.0% | 1.86 | +28.8% |
| **2020** | 240 | 87% | +4.1% | +2.6% | 1.35 | +15.3% |
| 2021 | 247 | 95% | +8.2% | +1.3% | 2.77 | +28.8% |
| **2022** | 253 | 66% | −13.4% | −8.2% | 0.51 | −20.0% |
| 2023 | 228 | 83% | +2.3% | −6.1% | 1.18 | +24.7% |

  Excess is positive in 7 years and negative in 7. The two worst years (2018,
  2022) are each −8 points, larger than any good year's gain except 2016.

- **On the drawdown figure.** The registered fixed-fractional book reads
  +218% total and a −108.7% max drawdown (2022-01 → 2022-10). That is an
  artifact of entering **every trading day** with no concurrency cap. The 31-day
  holds overlap to a median of **21 open positions** (max 30), so ~40-60% of
  notional is at risk at once. The live sizing rules (2% per trade, 10%
  concurrent) would allow ~5. The number shows that 2022 wipes out a fully
  stacked short-put book. It is not the drawdown a sized book would see, and it
  should not be quoted as one.

## Parallel and sensitivities (no authority)

| variant | n | RoC | A′ excess | B′ PF |
|---|---|---|---|---|
| non-overlapping (first entry per expiry month) | 167 | +1.71% | −2.83% [−6.56, +0.70] fail | 1.144 [0.755, 1.943] fail |
| cluster by entry month | 3,157 | +3.46% | −0.82% [−2.95, +1.29] fail | 1.338 [0.968, 1.990] fail |
| mid fill at entry | 3,157 | +4.33% | +0.06% [−2.11, +2.20] fail | 1.428 [1.027, 2.138] pass |
| short delta 0.16 | 3,055 | +2.77% | +0.32% [−1.38, +1.92] fail | 1.513 [1.003, 2.609] pass |
| short delta 0.32 | 3,183 | +4.05% | −1.53% [−3.92, +0.84] fail | 1.286 [0.955, 1.816] fail |
| DTE [45, 60] | 2,682 | +4.36% | −0.47% [−3.35, +2.26] fail | 1.409 [0.962, 2.261] fail |
| width 1% of spot | 2,965 | +2.54% | −2.00% [−4.35, +0.33] fail | 1.210 [0.881, 1.790] fail |
| width 3% of spot | 3,176 | +3.84% | −0.08% [−2.11, +1.87] fail | 1.447 [1.024, 2.216] pass |
| credit floor 0.20 | 22 | +26.8% | (10 months, unjudgeable) | — |

**No variant clears A′.** Every excess CI contains zero, and the point
estimates run from −2.8% to +0.3%. B′ "passes" three times. That is just a
long-delta position in a bull market: PF above 1 with no excess over delta is
what holding SPY gives you. And none of them could be promoted anyway.

## Consequence

Across four measurements in this reset, the live single-name book, the live
rule on index, and this index rule each come out **at or below their own delta,
never above it**:

| measurement | excess over delta |
|---|---|
| task 3, live book (n=59) | −4.02% [−9.61, +1.64] |
| task 4, live rule on SPY (n=927) | −2.10% [−2.72, −1.59] |
| R, SPY held to expiry (n=3,157) | −0.82% [−2.98, +1.32] |
| R's best variant (0.16Δ) | +0.32% [−1.38, +1.92] |

A put spread here is a way to hold delta. Done well, it costs nothing beyond
holding the delta; done badly, it costs the crossings. Section 3 of the decision
rule has no candidate rule left. The honest end state is the decision rule's
fail branch: hold the index directly, and keep the screener as a research tool
with its shadow recorder.

No config change.
