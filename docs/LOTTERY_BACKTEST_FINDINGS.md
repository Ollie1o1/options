# Lottery-Ticket Sleeve — Backtest Findings

**Date:** 2026-06-08
**Code:** `src/lottery/` (selector + model-based backtest), `tests/lottery/` (31 tests)
**Status:** Built, tested, adaptable. **Not deployed.** Single-name selection did
not beat the baseline; the strategy is marginal-to-negative EV in this study.

---

## What was built

- `src/lottery/selector.py` — config-driven "Lottery Score" (vol-cheapness,
  momentum, convexity sweet-spot, liquidity, catalyst-gated-on-cheap-IV,
  explosiveness) with hard disqualifiers for the Boyer-Vorkink overpricing trap.
  Fully tunable via the `lottery` block in `config.json`.
- `src/lottery/backtest.py` — model-based simulation on **real underlying price
  paths**. Three strategies on identical draws: `smart` (evidence-based single
  pick), `naive` (buy the hottest / highest-IV-rank name), `blind` (buy one of
  every candidate = the average-lottery baseline).

## Method & honest assumptions

Historical option chains / IV are unavailable, so options are priced with
Black-Scholes at an **IV proxy** = 60d realized × (1 + vrp) with a far-OTM skew
markup; outcome = intrinsic at expiry on the actual realized path (optional
path-based take-profit). No real implied surface, no earnings calendar (catalyst
factor inert here), flat rate, European exercise. **This measures *relative*
edge between selection rules, not a promise of live P&L.** 28-name basket,
deliberately mixed with laggards to fight survivorship; weekly draws over 2y.

## Results (ROI per ~2-week cycle, $100/ticket, 5% slippage)

Realistic pricing turns the naive/blind baseline to **~−6%** (2σ, hold) —
consistent with Boyer & Vorkink (2014): buying lottery-like options is
negative-EV on average. Key cells:

| config | SMART | naive | blind | blind win% |
|---|---|---|---|---|
| dte14 σ2.0 hold | −32% | −6% | −6% | 2.7% |
| dte14 σ2.5 hold | −63% | −60% | **+18%** | 1.3% |
| dte30 σ2.0 hold | −88% | −100% | −34% | 2.2% |
| dte45+ | worse | worse | worse | — |

Take-profit (3×/5×/8×) made **every** strategy worse: capping winners destroys
the uncapped right tail that is the entire edge.

## Findings

1. **Single-name selection failed.** The evidence-based `smart` picker never
   beat `blind` in any configuration. Concentrating into one "best" ticket
   throws away the breadth that lottery payoffs depend on.
2. **The earlier +41% was a pricing artifact.** Underpricing entry made timing
   irrelevant and inflated payoffs; realistic, vol-state-dependent pricing
   removed it.
3. **The lottery edge, to the extent it exists in the data, is breadth + tail:**
   buy *many* short-dated ~2.5σ-OTM options, hold to expiry, never take profit.
   Even that (+18%) is **one fragile cell** — 1.3% win rate carried by a couple
   of outlier moves, and highly sensitive to the IV-pricing assumption.
4. **Longer DTE is worse** — theta on far-OTM options dominates the higher hit
   rate.
5. The academic "buy cheap vol" edge (Goyal-Saretto) is for vol-harvesting /
   delta-hedged trades, **not** naked directional far-OTM payoffs. The relevant
   trait for lottery payoffs is explosiveness (fat tails), which selection should
   lean *into*, not filter out.

## Update 2026-06-08 (later) — real-IV calibration (free data)

Added a provider-agnostic option-data layer (`src/lottery/data.py`): a free
`YFinanceProvider` (real current chains — IV/bid/ask/OI), a paid `PolygonProvider`
stub (same interface, for historical per-contract premiums when subscribed), and
`calibrate_from_chains()` which measures the **real** vol-risk-premium and skew.
Run with `python -m src.lottery.backtest --calibrate [--provider polygon]`.

**Measured from 9 real chains: VRP ≈ 0.093, skew ≈ 0.210/σ** — the real OTM skew
is **~2.6× steeper** than the 0.08 I had guessed. Far-OTM wings are richly
priced, exactly as Boyer-Vorkink predicts. Re-running with real pricing makes
everything materially worse:

| config | guessed | real-calibrated |
|---|---|---|
| blind σ2.5 hold | +18% | **−67%** |
| smart σ2.0 hold | −32% | −82% |
| naive σ2.0 hold | −6% | −58% |

The one "promising" cell was an artifact of underpricing the skew. With real
data the lottery sleeve is solidly negative-EV across the board. (Caveat: only
current surface shape applied to historical realized vol — true per-contract
history needs the paid provider, but that would only *add* costs like IV-crush,
not reverse the sign.)

## Recommendation

Do **not** auto-log a single-pick lottery sleeve as-is — the backtest says it
loses. If pursued, pivot the design from "pick the one best ticket" to a **small
diversified lottery basket** (breadth + tail), and re-validate with **real
historical IV** (the current proxy is the biggest source of uncertainty) before
risking money. The system is built to be re-tuned — weights, guardrails, sigma,
DTE, TP are all config knobs and sweep in seconds.
