# What Not To Do — measured avoidance rules

**Last updated:** 2026-08-06

Every rule here is something the data says **costs money**, with the measurement
behind it and an honest confidence level.

Why this document exists separately from the findings: an avoidance rule is
usually *more useful and safer than an edge*. It tends to be a larger effect
(the downtrend penalty is a bigger deviation from baseline than the best
positive signal found), it needs no edge to act on — only the discipline to skip
a known-bad cell — and acting on it cannot lose money you were not already going
to lose. A "no edge found" result that discards its negative cells has thrown
away most of what it learned.

Confidence is stated plainly. **Strong** = large effect, large sample, survives
clustering. **Moderate** = consistent and sensible, but not statistically
established. **Inherited** = from earlier repo research, not re-measured here.

---

## 1. Do not sell puts into a downtrend

**−4.65% return on capital per trade, against a −0.64% baseline.**
Condition: spot below its own 52-week average, mega-cap universe, n=504.

The single largest deviation measured. Also the most intuitive: a put you sold
is a bet that the fall stops, and a name already trending down is the worst
place to make it. The naive t is −2.30; clustered by entry day it is −1.42.

**Confidence: moderate-to-strong.** Large effect, mechanically sensible, and the
sign is not in doubt even if the magnitude is.

## 2. Do not sell premium into a name that just dropped hard

**−2.38% per trade** after a 4-week fall of more than 5%. n=391.

This one is worth stating because it **contradicts a popular idea** — "IV spikes
when price falls, so sell the fear." It was tested directly and came back worse
than doing nothing. The implied vol is higher because the risk is genuinely
higher, and here the extra premium did not cover it.

**Confidence: moderate.** Contradicts intuition, which is precisely why it is
worth recording rather than quietly dropping.

## 3. Do not sell premium when it is cheap

**−0.97% at IV rank ≤ 30**, worse than selling indiscriminately (−0.64%).

Part of a monotone series: IVR≤30 −0.97%, no filter −0.64%, IVR≥50 +2.02%,
IVR≥70 +3.00%. The negative end is as informative as the positive end — if you
are going to sell premium at all, selling it cheap is the worst version.

**Confidence: moderate.** The monotone ordering across a graded parameter is the
evidence; no single cell reaches significance.

## 4. Do not trade options on wide-spread underlyings

**−6.76% per trade across all 115 names, versus −0.64% on the mega-caps and
index ETFs.** Same strategy, same period; only the universe changed.

Median bid-ask as a share of mid: **3.6%** on mega-caps and index ETFs, **16.4%**
on the liquid stratum, **20.9%** on the broad stratum.

This is the largest single driver found anywhere in the study. n=10,363.

**Confidence: strong.** Huge sample, huge effect, t=−17.84, and the mechanism is
directly measured rather than inferred.

## 5. Do not cross the spread more than you must

**Crossing costs 50% of the mid credit.** Median credit $90 at mid, $45 crossed,
over 400 trades. Median per-leg bid-ask $0.35.

Consequences that follow:
- **Hold to expiry rather than managing to a profit target** where the strategy
  allows it — you pay the opening legs only, roughly halving the toll.
- A $5-wide spread collecting 9% of width after crossing cannot work at
  25-delta; the arithmetic is `0.709 × $45 − 0.291 × $455 = −$100`.

**Confidence: strong.** Direct measurement, not a fitted result.

## 6. Do not buy premium to express a directional view

**Long call control: 16.7% win rate, −$934 per trade** in this backtest.
The live ledger agrees: 243 long calls, 38% win, −147.9% total.

The same bullish view sold as a put spread won 66% in the live book. Buying
needs direction *and* magnitude *and* timing; selling needs only "not sharply
down".

**Confidence: strong.** Two independent datasets, same conclusion.

## 7. Do not sell cash-secured puts on a $4,000 account

Only **15 of 109** universe symbols trade at a strike level at or below $40, and
that subset skews to beaten-down, high-volatility names (APA, CZR, NCLH, PENN,
VTRS) — the worst place to be short puts. Median capital for a cash-secured put
in the live ledger was **$21,855**, with only **15% affordable**.

Defined-risk spreads reach the same premium for roughly **$500** instead.

**Confidence: strong.** Arithmetic, not inference.

## 8. Do not cherry-pick by quality_score

Selectivity degraded results **monotonically**: top-1-per-day was **4.7× worse
per trade** than taking everything, and the pattern held across all three panels.
The highest score bucket [0.85, 1.00] was the worst cell in the book at 33% win
and −$165 per trade.

**Confidence: inherited, strong.** `docs/PROFITABILITY_FINDINGS.md`, n=393.

## 9. Do not trust a high win rate on its own

Every configuration tested shows skew between **−1.1 and −2.8**. A short-premium
strategy with an 84.9% win rate and skew −2.80 is not a good strategy; it is one
whose losses have not arrived yet.

This is the same unobserved-tail caveat the live short-premium gate already
carries, now confirmed independently.

**Confidence: strong.** Present in every configuration measured.

## 10. Do not believe a backtest result without deflating it

A $25-wide spread showed 84.9% wins and +1.76% return on capital, reading
**DSR 0.921 in isolation**. Deflated by the 18 configurations actually tried, it
read **0.432**. It was what an 18-way search produces from noise.

The width series was also non-monotone ($10 worse than $5, $20 worse than $10,
$25 positive) — the shape of an artifact rather than an effect.

**Confidence: strong.** This is a methodological rule, not an empirical one.

---

## How to use this

These rules compose into a negative filter that requires no edge:

> Trade tight-spread underlyings only. Do not sell when implied vol is cheap.
> Do not sell into a downtrend or straight after a sharp fall. Hold to expiry
> rather than paying the exit toll. Do not buy premium to express a view. Do not
> size a cash-secured put you cannot afford.

Applying all of it still does not produce a demonstrated edge — the best
signalled configuration reaches t=0.83 clustered, against Harvey's hurdle of
3.0. But it removes most of the cells where money was measurably lost, and that
is worth more than a marginal positive result that has not survived deflation.
