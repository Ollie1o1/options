# Quant Research Log — real-marks cohort (DoltHub)

All numbers are all-in (real bid/ask spread + $0.65/contract commission), long
calls, canonical exit rules, weekly entries, via `src/dolt_research.py`.

## Method (overfitting-resistant)
Hypothesis-driven entry filters, one at a time, basket-wide, with time AND symbol
holdouts. A filter must show a **large, consistent effect that survives out-of-sample**
to count. We discount anything that only works in-sample or on one cut.

## Experiment 1 — entry-filter sweep (AAPL/SPY/QQQ/MSFT, 2022–2024)

| filter | n | win | avg | median | PF |
|---|---|---|---|---|---|
| baseline | 326 | 38.7% | +0.8% | −17.4% | 1.02 |
| low_vix_20 | 195 | 39.0% | +4.5% | −10.8% | 1.15 |
| **low_vix_18** | 160 | 40.6% | **+8.7%** | −9.2% | **1.29** |
| high_vix_22 | 89 | 32.6% | −6.5% | −32.3% | 0.82 |
| trend_up_50 | 187 | 39.6% | +0.3% | −14.3% | 1.01 |
| low_iv_0.30 | 248 | 39.1% | +1.7% | −14.1% | 1.05 |
| lowvix20_trend50 | 140 | 37.9% | +2.0% | −12.9% | 1.06 |

**Read:** monotonic in VIX (vix18 > vix20 > baseline > high-vix); high-VIX actively
loses. Trend and low-IV filters don't help. The VIX effect matches the earlier
independent BS-walk-forward finding (low-VIX = only profitable regime).

## Experiment 2 — validating low_vix_18 (the only promising signal)

**Time holdout:** train 2022–23 PF 1.01 → test 2024 PF 1.5.
**Symbol holdout:** AAPL+SPY PF 1.17 → **1.63**; QQQ+MSFT PF 0.84 → 0.87.

**Verdict:** the low-VIX overlay improves *every* cut (robust direction), but the
magnitude is only enough to make AAPL/SPY clearly profitable — it lifts QQQ/MSFT
to ~breakeven and temporally leans on 2024. **A real factor / overlay, NOT a
standalone edge.** The sweep's aggregate PF 1.29 was flattered by AAPL/SPY + 2024.

## Standing conclusions
- Long calls are structurally cost-challenged (negative median always; tail-driven).
- **VIX regime is the one robust factor found so far**: don't buy long calls in high
  VIX; prefer low-VIX entries. Use as an overlay / position-size gate, not a guarantee.
- No filter tested turns the cohort into a robustly profitable standalone strategy.

## Open research directions (not yet run)
1. Finer VIX buckets + more symbols (does the overlay's magnitude hold at n>4 names?).
2. A different strategy class on the same real marks: **defined-risk spreads / short
   premium**, which the cost structure may favor over long premium.
3. Underlying-character split (index-like vs high-beta) — flagged but NOT yet tested
   enough to trust (overfitting risk at n=2 pairs).
