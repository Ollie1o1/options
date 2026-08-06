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
