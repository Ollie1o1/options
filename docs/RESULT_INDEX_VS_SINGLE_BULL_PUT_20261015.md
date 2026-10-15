# Index vs single-name Bull Puts — RESULT

Registration: `docs/PREREG_INDEX_VS_SINGLE_BULL_PUT_20261015.md` (frozen in
`b17cf48`, before any outcome was queried). Estimator:
`scripts/index_vs_single_bull_put.py`, run once as registered. DB sha256
`247b714dbea31d9f`; 19,150 primary rows; 4 clipped.

## Primary: INCONCLUSIVE

`D = R̄(index, refused) − R̄(single, passed)`, paired by day, with a
week-cluster bootstrap:

```
D = -0.0036  [-0.1098, +0.0951]   32 days, 8 weeks, 4 / 45 symbols, 4,822 / 1,232 rows
```

R is P&L per dollar of max loss over a ~3-trading-day hold, at mid.

| arm | rows | symbols | mean R [95% CI] |
|---|---|---|---|
| index refused | 4,822 | 4 | −0.005 [−0.085, +0.071] |
| index passed | 193 | 3 | +0.047 [−0.194, +0.203] |
| single refused | 11,531 | 85 | −0.006 [−0.030, +0.012] |
| single passed | 1,232 | 45 | +0.014 [−0.045, +0.086] |
| other ETF refused | 1,029 | 12 | −0.093 [−0.208, −0.004] |
| other ETF passed | 343 | 4 | −0.044 [−0.199, +0.087] |

All five sensitivities are inconclusive on the primary: drop the 10-12 batch,
row-weighted, include 08-29/30, mature ≤ 10-01. The exception is the other-ETF
substitution, which comes out "gate right" (−0.112 [−0.231, −0.003]). That cut
has no authority. It is driven by the other-ETF refused arm's poor level
(GLD/SLV/TLT/XL*), which has nothing to do with the index question.

## What this does and does not say

- **No evidence either way that the gate is pointed at the wrong population.**
  Point estimates of the refused-index and passed-single arms are within half a
  percent of capital at risk.
- **The test has no power for the question.** The CI half-width is ~10% of
  capital at risk per 3-day hold. Any edge worth having is ~1-2% per hold, so
  the interval would have to shrink ~5-10x. That needs ~25-100x more weekly
  clusters. A forward re-run in November will not change the verdict.
- The index arm is **4 symbols** that move together. The binding constraint is
  market regimes, not rows; 4,822 rows is about 8 independent observations.

## S1, the skew mechanism: UNTESTABLE here

The skew penalty is almost perfectly confounded with being an index:
- Q5 of the penalty is 99% index.
- Single-name rows barely reach Q5 (32 refused rows).

Refused index+single Q5 − Q1 = −0.024 [−0.084, +0.033]. Single-only
Q5 − Q1 = −0.172 [−0.409, +0.040], on 12 days. Neither resolves. The corpus
cannot separate "skew penalty" from "index".

## Consequence for the plan

The shadow corpus cannot answer task 2's question, now or by the 2027-01-15
decision date. The venue that can is **optionsDX SPY 2010-23 (task 4)**: 14 years
of index chains, where the gate's own EV can be recomputed historically from
trailing realised vol. That allows a direct test of whether spreads the gate
would refuse for skew actually underperform those it passes. Only fold that
into task 4 under its own registration.

Unchanged: no gate change, no config change.
