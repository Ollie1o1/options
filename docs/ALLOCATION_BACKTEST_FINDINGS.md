# Allocation Backtest — First Findings

**Date:** 2026-08-06
**Engine:** `src/alloc/` on 115 symbols x 225 weekly dates of real DoltHub EOD
chains (bid/ask/IV/Greeks), 2022-01-07 to 2026-06-12.
**Status:** engine mechanically validated; statistics layer (CPCV/DSR/PBO) not
yet built, so nothing here is deflated for multiple testing.

> **CORRECTION 2026-08-08 — every number below was computed with splits
> unhandled.** `replay()` accepts `splits=` and the CLI never passed it, so the
> split guard was dead code and `split_closed` was structurally always 0 —
> against this repo's own warning (`src/alloc/splits.py`) that leaving splits
> unhandled makes a short put book a catastrophic loss that never happened and
> makes trend signals read a split as a 95% crash for a year afterwards. Six of
> the fourteen tight-spread names split inside the sample. Fixed and wired; the
> §4d signal results in particular should be re-run before being relied on.
> See `docs/TAIL_OBSERVED_20260808.md` §4.

---

## 1. The engine is mechanically credible

Held to expiry, settling at intrinsic value:

| structure | n | win rate | worst loss |
|---|---:|---:|---:|
| bull put 25-delta | 206 | 70.9% | -$495 |
| bear call 25-delta | 220 | 80.5% | -$495 |
| iron condor 16-delta | 171 | 74.9% | -$490 |
| **long call 40-delta [CONTROL]** | 6 | **16.7%** | -$2,798 |

These are the rates the structures must mechanically produce, the worst loss
respects the $500 structural maximum of a $5-wide spread, and the
known-negative control loses as designed. Three bugs were caught getting here,
each of which produced *plausible-looking* wrong answers — see §4.

## 2. The finding: crossing the spread costs HALF the credit

Measured over 400 real trades, comparing what the same legs would have paid at
mid against what they actually fill at:

| | credit | as % of width |
|---|---:|---:|
| at MID | $90 | 18% |
| **CROSSED (real fill)** | **$45** | **9%** |
| **given up** | **$45** | **50% of the mid credit** |

Median per-leg bid-ask spread: **$0.35**.

This is the cost wall, quantified on real chains rather than assumed. It is
larger than any previous estimate in this repo — `PROFITABILITY_FINDINGS.md` §7
assumed a flat $0.05/share, and the ledger re-derivation on 2026-08-06 measured
**$0.050-$0.350** depending on structure (`docs/SCORER_IMPROVEMENTS.md` §5; Bull
Put's round trip is 68% of its credit, Bear Call's 23%). On the wider
115-symbol universe the toll is **half the credit**.

### Why that is fatal at these parameters

A 25-delta short strike finishes ITM roughly 25% of the time. Selling a $5-wide
spread for $45 against $455 of risk:

```
EV = 0.709 x $45  -  0.291 x $455  =  $32 - $132  =  -$100
```

The observed average is **-$79**. The arithmetic and the measurement agree.

At mid ($90 credit) the same trade is about **-$55** — still negative. So at
25-delta on a $5 width, this structure does not work *even before* costs, and
crossing roughly doubles the loss.

## 3. Selling further out of the money does not rescue it

| short delta | n | win rate | avg P&L | credit / width |
|---|---:|---:|---:|---:|
| 0.10 | 208 | 93.3% | **-$2.45** | 2% |
| 0.16 | 227 | 85.5% | -$27.71 | 4% |
| 0.25 | 206 | 70.9% | -$79.27 | 8% |
| 0.35 | 159 | 59.1% | -$118.14 | 10% |

Monotone: the further out you sell, the closer to breakeven — because you are
crossing a smaller absolute spread. **10-delta is nearly free money and nearly
zero money**, at $10 of credit per trade.

The pattern says the loss is being driven by *friction*, not by direction.
Nothing here is a timing signal; it is the toll.

## 4. Three bugs, each of which produced a believable wrong answer

Recorded because none would have been caught by "does the code run", and all
three were caught by comparing against what the structure must mechanically do.

1. **Quotes keyed on (strike, type)** while a chain spans many expirations, so a
   March 100-put collided with a June 100-put. Produced a 17% win rate.
2. **A zero bid treated as missing data.** An option expiring worthless has a
   zero bid — that is a price. Winners could never be closed while losers
   always could. Produced a 13% win rate.
3. **Expiry settled off quotes.** An illiquid long leg with a zero bid could be
   "sold" for nothing, giving a **-$620 exit on a $5-wide spread** — a loss the
   structure cannot physically sustain. Expiry now settles at intrinsic value,
   with the underlying recovered by put-call parity.

## 4b. Nothing survives deflation — including the result that looked like a find

Attacking friction three ways (a crossing-cost filter, wider spreads, and the
two combined) produced **two positive configurations out of eighteen tried**.
Deflated by the search that produced them:

| configuration | n | win | avg RoC | Sharpe | t | skew | DSR alone | **DSR deflated** |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| width $5 (baseline) | 206 | 70.9% | -17.15% | -0.389 | -5.58 | -1.14 | 0.000 | **0.000** |
| width $10 | 237 | 75.5% | -12.72% | -0.320 | -4.93 | -1.48 | 0.000 | **0.000** |
| width $20 | 243 | 71.6% | -7.69% | -0.290 | -4.52 | -1.96 | 0.000 | **0.000** |
| **width $25** | 199 | **84.9%** | **+1.76%** | 0.117 | 1.65 | -2.80 | **0.921** | **0.432** |
| width $25 + friction<=15% | 76 | 78.9% | +0.51% | 0.025 | 0.22 | -1.92 | 0.583 | **0.056** |

The `width $25` row is the whole lesson. In isolation it reads **DSR 0.921** —
85% wins, positive expectancy, a discovery. Deflated by the **18 configurations
actually tried**, it reads **0.432**: below the 0.5 line, with t=1.65 against
Harvey's hurdle of 3.0, and a skew of **-2.80**.

It is what an 18-way search produces from noise. The width series is also
non-monotone ($10 worse than $5, $20 worse than $10, then $25 positive), which
is the shape of an artifact rather than an effect.

**Verdict: no configuration tested so far has an edge.** That is a real result,
not a failure to find one.

### Why the skew matters

Every configuration shows skew between **-1.1 and -2.8**. That is the
short-premium signature — many small wins, rare large losses — and it is why a
high win rate here means very little on its own. An 84.9% win rate with skew
-2.80 is not a good strategy; it is a strategy whose losses have not arrived
yet. The same caveat the short-premium gate already carries about its
unobserved tail applies with equal force to everything in this table.

## 4c. Index/mega-cap universe — friction confirmed as the driver

Three further bugs had to be fixed before this test was even possible, and each
had silently excluded the high-priced names — i.e. **every result above §4c was
measured on a biased subset that omitted the tightest-spread underlyings**:

1. **Wings required an exact `short - width` strike.** The dataset lists ~150-200
   contracts per symbol-day, so on a $500 name the strike five dollars away is
   often not listed. Wings now snap to the nearest listed strike on the
   protective side, and risk is priced off the width actually obtained.
2. **Fills required both a bid and an ask on every leg.** You only need a bid to
   sell and an ask to buy; far-OTM protective wings legitimately quote bid=0 and
   are bought at the ask. This alone rejected ~1,000 entries.
3. **Settlement could not find the underlying on expiry day**, because the
   expiring contracts have usually already left the chain. Positions therefore
   never closed — they accumulated to the end of the sample and were written off
   as `ticker_ended`, discarding almost every mega-cap trade.

With those fixed the sample went from 276 trades to **10,363**:

| universe | n | win | RoC/trade | t | skew | DSR (26 trials) |
|---|---:|---:|---:|---:|---:|---:|
| ALL 115 names, bull put | 10,363 | 76.8% | **-6.76%** | -17.84 | -1.66 | 0.000 |
| **MEGA/index, bull put** | 1,442 | **80.5%** | **-0.64%** | **-0.58** | -1.71 | 0.004 |
| MEGA/index, bear call | 1,424 | 71.8% | -10.66% | -8.40 | -1.17 | 0.000 |
| MEGA/index, iron condor | 1,276 | 71.3% | -6.44% | -5.09 | -1.28 | 0.000 |

**The friction hypothesis is confirmed.** The identical strategy moves from
decisively negative (-6.76%, t=-17.84) to statistically indistinguishable from
zero (-0.64%, t=-0.58) purely by restricting the universe to names whose spreads
are tighter — 3.6% of mid on the mega-caps and ETFs, against 16-21% on the
liquid and broad strata.

> **PARTIALLY REPLICATED, and narrower than stated (2026-08-08).** Re-derived
> uncapped with splits wired over 2022-2024: the ALL-names bull put row
> reproduces closely (-6.40% vs -6.76%, n=6,917), and the mega restriction does
> move bull puts to indistinguishable-from-zero (-1.15%, clustered t -1.57).
> **But the bear call is WORSE on the tight-spread universe** (-12.54% vs
> -7.92% broad), and tighter spreads cannot make a strategy worse. The MEGA
> cohort is SPY plus mega-cap tech over a rally, so the call side is dominated
> by a bullish tilt, not by friction. Friction is the driver for the PUT side
> only. See `docs/ATTRIBUTION_20260808.md` 7b.

**But breakeven is not an edge.** DSR = 0.004. At the tightest spreads available
in this dataset, selling put spreads pays for its own friction and no more. The
bear call and iron condor stay clearly negative even there.

## 4d. Signals on top of the breakeven baseline

Signals are computed strictly causally from the chain itself — spot by put-call
parity, ATM IV from the strike nearest spot, ranked within a trailing 52-week
window. A condition whose feature cannot be computed **fails**, so an
insufficient-history day is never silently treated as unconditional.

Universe: mega-cap + index ETFs. Structure: bull put, 25-delta, held to expiry.

| condition | n | win | RoC/trade | t | DSR (34 trials) |
|---|---:|---:|---:|---:|---:|
| **BASELINE — no signal** | 1,442 | 80.5% | **-0.64%** | -0.58 | 0.003 |
| IV rank <= 30 | 600 | 80.2% | -0.97% | -0.59 | 0.003 |
| IV rank >= 50 | 691 | 82.3% | **+2.02%** | 1.27 | 0.208 |
| **IV rank >= 70** | 440 | 82.3% | **+3.00%** | 1.51 | **0.282** |
| uptrend (spot > avg) | 919 | 81.5% | +0.33% | 0.25 | 0.031 |
| **downtrend (spot < avg)** | 504 | 75.8% | **-4.65%** | **-2.30** | 0.000 |
| after 4w drop > 5% | 391 | 78.3% | -2.38% | -1.03 | 0.001 |
| after 4w rally > 5% | 496 | 79.8% | -1.19% | -0.59 | 0.003 |
| IVR>=50 AND uptrend | 405 | 82.5% | +2.06% | 0.98 | 0.138 |

> **DOES NOT REPRODUCE (2026-08-08).** Measured as a rank IC on 2020-2024 with
> splits wired, `iv_rank` is flat for every structure: **-0.010** (long_call),
> **-0.029** (bull_put), **+0.006** (bear_call), all p > 0.69. The monotone
> pattern above was computed on a different window, with splits unhandled, and
> as a conditional mean rather than a correlation. Treat it as unconfirmed and
> do NOT build an IV-rank entry filter on it. Separately, nothing tested
> predicts the max-loss trades at all (best AUC 0.636 over an 11-way search).
> See `docs/ATTRIBUTION_20260808.md` §4-5.

### The shape is the evidence, not any single number

**IV rank is monotone**: -0.97% at IVR<=30, -0.64% unconditional, +2.02% at
IVR>=50, +3.00% at IVR>=70. Ordered across a graded parameter, in the direction
theory predicts — sell premium when premium is rich. Monotonicity across a range
is much harder to produce by chance than one good cell, and it is exactly what
the earlier width series (non-monotone) lacked.

**Selling puts into a downtrend is the clearest single effect**: -4.65% against
a -0.64% baseline, the largest deviation in the table. Also the most intuitive —
a put you sold is a bet the fall stops.

**The "sell fear after a drop" hypothesis is contradicted.** Selling after a 4-week
drop of more than 5% returned -2.38%, worse than baseline. The IV is higher
because the risk is higher, and here the compensation did not cover it.

### None of it is established

| condition | t (naive) | t (clustered by entry day) |
|---|---:|---:|
| BASELINE | -0.58 | -1.06 |
| IV rank >= 50 | 1.27 | 0.55 |
| IV rank >= 70 | 1.51 | 0.83 |
| downtrend | -2.30 | -1.42 |

Positions opened on the same day share that day's market move, so the naive
t-stat treats correlated trades as independent. Clustering by entry day roughly
**halves every t-statistic**. The best result, IVR>=70, falls to **t=0.83** —
against Harvey's hurdle of 3.0 — and its deflated Sharpe is 0.282 against a 0.5
line.

**Verdict: suggestive, coherent, and not established.** The monotone IV-rank
pattern and the downtrend penalty are the two things worth carrying forward;
neither is yet evidence a real-money gate should act on.

## 4d. The wider index spread, run 2026-08-07 — and the instrument bug it found

`index_put_spread_w25` (§ the STRATEGIES desk) was replayed against its own
controlled comparison: identical universe, entry, delta and exit, with **width
the only difference**.

| arm | n | win | RoC/trade | t (clustered) | skew | DSR (35 trials) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| **$25 wide, SPY** | 28 | 85.7% | **+5.00%** | 1.26 | -2.49 | **0.255** | reject |
| $5 wide, SPY | 52 | 84.6% | +3.20% | 0.65 | -2.23 | 0.090 | reject |

**The mechanism is visible and the verdict is still reject.** Width lifted
return on capital by 1.8 points on the same 225 Fridays, in the direction the
friction argument predicts — credit scales with width while the two crossings
do not. It did not lift it far enough: DSR 0.255 against the 0.5 line, t=1.26
against Harvey's hurdle of 3.0, and **skew -2.49**, which is the short-premium
signature of losses that have not arrived. Capacity is 6.6 trades/year at one
concurrent position, because $2,682 of peak deployed capital is 67% of the book.

Undeflated it reads DSR 0.829. That gap — 0.829 alone, 0.255 after 35
configurations — is the same lesson as §4b, arriving for the second time.

**This is IN-SAMPLE.** The window is the one the 18-configuration friction
search already ran on, so the result can falsify the hypothesis but cannot
confirm it.

### The instrument bug: a DTE window narrower than the expiration ladder

The first run of this setup returned **n=3**. The cause was not the strategy:
78 of 96 otherwise-eligible SPY dates could assemble no legs, because the window
`dte [30,45]` contained no listed expiration at all.

Audited across **25 randomly sampled symbols** — share of symbol-days carrying
any expiry inside the window:

| dte window | median | worst | symbols below 95% |
|---|---:|---:|---:|
| `[30, 45]` (was in the library) | **52.4%** | 34.4% | 25/25 |
| `[7, 21]` (was the earnings setup) | **55.4%** | 53.4% | 25/25 |
| `[25, 45]` (the alloc CLI default) | **69.6%** | 69.0% | 25/25 |
| `[7, 35]` | 95.3% | 94.5% | 4/25 |
| `[25, 60]` | **99.8%** | 99.3% | 0/25 |
| `[7, 45]` / `[5, 45]` | **100%** | 99.8% | 0/25 |
| `[30, 70]` (dolt_vol, dolt_earnings_sell) | **100%** | 99.6% | 0/25 |

The backfill carries a **median of 2.2 expirations per symbol-day**. What
matters is span, not placement: every window spanning 35 days or more cleared
99%, every window under 30 days failed. Windows are now `[25,60]` for the
monthly setups and `[7,45]` for the earnings setup, and the engine picks the
NEAREST expiry inside the window, so widening only adds dates on which nothing
closer was listed — it does not change which expiry is chosen when one exists.

**A window validated on SPY is not validated for the universe.** SPY carries 3.2
expirations/day against the universe median of 2.2, so `[25,45]` reaches 99.8%
on SPY and only 69.6% universe-wide. That error was made once in this very
section before the audit was run.

#### What this does to the earlier results

- **§4c mega/index rows are unaffected.** Measured per symbol, `[25,45]` reaches
  99.8% on SPY, 99.5% on AAPL, 99.6% on NVDA, 99.4% on XLV.
- **§2, §3 and the ALL-115-names row of §4c are subject to a coverage caveat.**
  Those runs used `[25,45]` over the wide universe, which reaches only ~70% of
  symbol-days. The samples were large (10,363 trades), so this is not a starved
  n — but entries were confined to the ~70% of dates where the monthly ladder
  happened to fall in range, which is a calendar-structured subsample rather than
  a random one. Direction of bias unknown; magnitude presumed small.
- **The Dolt VRP and earnings research is unaffected.** `src/dolt_vol.py` and
  `src/dolt_earnings_sell.py` use `[30,70]` (100%), and every other Dolt module
  picks the nearest expiry past a floor (`>= min_dte`) rather than bounding a
  window, which cannot starve.

The general lesson is the one worth keeping: **the starved run did not fail
loudly. It reported DSR 0.996 and a verdict of `liquid_only` on three trades.**
`promotion_verdict` now refuses any verdict under 20 closed trades and returns
`insufficient` — deliberately NOT `reject`, because "could not measure" and
"measured and failed" are different claims and only one of them tells you to go
and get more data. Re-graded under that floor, the starved run reads
`insufficient`. The deeper guard is still to check coverage before believing a
result rather than after.

## 5. What this does NOT yet say

- **No statistics.** CPCV, Deflated Sharpe and PBO are not built. No number here
  is corrected for multiple testing.
- **No signal has been tested.** Every result above is unconditional selling.
  Whether a directional or IV-rank condition beats unconditional selling is the
  open question, and the benchmark to beat is now measured.
- **The stratum split is thin.** liquid n=34, broad n=172, legacy n=0 — the
  legacy names produced no fills at a $5 width. Not yet a fair comparison.
- **Weekly sampling.** Entries and exits observe Fridays only, so managed exits
  (profit targets, stops) fire late. Held-to-expiry is the honest read on this
  cadence.

## 6. What it implies for the next test

The binding constraint is **not when to trade — it is the spread paid to get
in.** No timing signal recovers 50% of the credit. The next tests should attack
friction directly:

1. **Liquidity-filtered universe.** Restrict to names whose bid-ask is a small
   fraction of the credit. The $0.35 median per-leg spread is the enemy.
2. **Wider spreads.** More credit per unit of fixed friction.
3. **Index and mega-cap only.** Where spreads are tightest — and where prior VRP
   research already found the premium positive.

Only after friction is controlled does testing a directional signal make sense.
