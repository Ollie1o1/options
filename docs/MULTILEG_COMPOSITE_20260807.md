# The multi-leg composite — 2026-08-07

`docs/ADJUSTMENT_STACK_20260807.md` measured the single-leg score and left a gap
it kept flagging: spreads and condors score through `spread_scoring` and
`credit_spread_weights` / `iron_condor_weights`, an entirely different composite
that had never been measured against outcomes. It governs 405 closed trades.

Reproduce: `scripts/measure_multileg_composite.py` (read-only, writes nothing).

---

## 1. Why this one can be measured directly

The single-leg score had to be decomposed as a residual, because ~20 hand-set
constants are applied after its composite. **Multi-leg rows never touch that
stack.** `spread_scoring` recomputes `quality_score` from its own weights and
that is the number that ships. The stored score *is* the composite.

| | credit spreads | iron condors |
|---|---|---|
| pop | 0.25 | 0.30 |
| credit_to_width | 0.20 | 0.20 |
| iv_rank | 0.15 | 0.12 |
| return_on_risk | 0.10 | — |
| delta_neutral | — | 0.15 |
| liquidity | 0.10 | 0.10 |
| theta | 0.08 | 0.08 |
| spread | 0.05 | 0.05 |
| momentum | 0.04 | — |
| catalyst | 0.03 | — |

Three of these — `credit_to_width_score`, `return_on_risk_score`,
`delta_neutral_score` — are not stored per trade but are exactly reconstructable
from `net_credit`, `spread_width`, `max_profit_usd`, `max_loss_usd` and
`net_delta`. The reconstruction reproduces the stored score **exactly** (median
absolute error 0.0000) on 245 of 405 rows: every vertical that carries component
scores. Condors carry a systematic offset — most likely `max_net_delta` or the
condor weights having moved since those rows were scored — so the decomposition
below is not extended to them. It does not affect the headline, which uses the
stored score throughout.

## 2. The composite is not anti-predictive here

Rank IC against return as recorded, n = 405, 2026-04-26 → 2026-07-31:

| cohort | n | rank IC | p |
|---|---|---|---|
| ALL multi-leg | 405 | −0.0087 | 0.861 |
| Bull Put | 131 | **+0.1396** | 0.112 |
| Bear Call | 135 | **+0.1010** | 0.244 |
| Iron Condor | 139 | **−0.0965** | 0.259 |

Expanding walk-forward over all multi-leg: mean **+0.0172**, negative in **1 of
5** windows.

Nothing here is significant at n ≈ 130 — an IC standard error at that size is
roughly 0.09. But the contrast with the single-leg score is the point, and it is
not subtle: **−0.0995 and negative in 5 of 5 windows there, ≈0 and positive in 4
of 5 here.**

### This is independent evidence about the adjustment stack

Same repo, same ledger, same period, same auto-logger. The difference between
the two scoring paths is that one applies the ~20-constant stack and the other
does not. **The path without the stack is the one that is not dragged negative.**

That is a second, structurally independent line of evidence pointing where the
residual decomposition already pointed. It does not prove the stack causes the
single-leg negative — the cohorts differ in strategy as well as in scoring path,
and long premium is known to be the worse book — but it is the comparison that
was missing, and it does not contradict.

## 3. Friction is a much larger fact than the score

109 rows carry real `entry_price_mid` and `entry_price_cross`.

| | |
|---|---|
| entry crossing, share of the mid credit | median **9.2%**, mean 19.5%, p90 40.0% |
| round trip (×2) | median **18.5%**, mean **39.0%** |
| round trip exceeds the **entire** credit | **8%** of trades |
| credit **vanishes** once crossed (`cross ≤ 0`) | **4%** of trades |
| never a credit even at the mid | 1 row |

Restating at the crossed credit — receive the cross, pay the slip again to
close:

| cohort | n | mid mean | mid median | mid win | net mean | net median | net win |
|---|---|---|---|---|---|---|---|
| ALL | 105 | +0.0615 | +0.1644 | 65% | −0.5961 | **−0.1287** | 43% |
| Bull Put | 26 | +0.1867 | +0.4067 | 69% | −1.2950 | **−0.1911** | 42% |
| Bear Call | 41 | +0.0191 | +0.1149 | 63% | −0.6711 | **−0.2083** | 29% |
| Iron Condor | 38 | +0.0216 | +0.1228 | 63% | −0.0372 | **+0.0561** | 58% |

The means are ratios over a credit base that can be arbitrarily small and are
outlier-driven; the medians carry the finding and agree on direction. **A book
that looks profitable at the mid (+16.4% median, 65% win) is losing at the price
it would actually fill (−12.9% median, 43% win).**

This is the tradeability finding again, on a larger and more recent sample, and
it dwarfs anything the composite does. A ±0.14 IC reorders candidates inside a
cohort whose median trade loses 13% of its credit to the spread.

## 4. The sharp bit: the composite's worst cohort is the one that works

Iron condors are the **only** multi-leg structure with a positive median return
net of crossing (+5.6%, 58% win). Bull Puts and Bear Calls both go clearly
negative.

And the composite ranks condors **worst** — the one cohort with a negative IC
(−0.0965 over 139 rows, −0.1752 on the execution-truth subset).

So the structure that survives its own costs is the one the score likes least.
Neither figure is significant on its own, and the condor IC rests on a cohort
whose composite could not be reconstructed, so this is a lead rather than a
result. It is the most decision-relevant thing in this document and the first
thing to re-check when more data lands.

## 5. What this changes

- **The multi-leg composite does not need urgent attention.** It is not
  anti-predictive, and on verticals it is mildly (insignificantly) useful.
- **It strengthens the case for the verdict ordering shipped in `c3667ea`**,
  and on the measured ground rather than the assumed one. The gate refuses
  exactly the trades this section counts: 8% whose round trip exceeds their
  credit, 4% whose credit vanishes on entry.
- **It does not license re-weighting anything.** n ≈ 130 per cohort, nothing
  clears p 0.10, and the condor cohort's composite could not be reconstructed.
- **The condor lead in §4 is the open question**, not the vertical weights.
