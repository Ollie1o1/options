# Allocation Backtest — First Findings

**Date:** 2026-08-06
**Engine:** `src/alloc/` on 115 symbols x 225 weekly dates of real DoltHub EOD
chains (bid/ask/IV/Greeks), 2022-01-07 to 2026-06-12.
**Status:** engine mechanically validated; statistics layer (CPCV/DSR/PBO) not
yet built, so nothing here is deflated for multiple testing.

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
$0.050-$0.288 depending on structure. On the wider 115-symbol universe the toll
is **half the credit**.

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

**But breakeven is not an edge.** DSR = 0.004. At the tightest spreads available
in this dataset, selling put spreads pays for its own friction and no more. The
bear call and iron condor stay clearly negative even there.

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
