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

## Experiment 3 — the SELL side + segmentation (the breakthrough)

Data note: QQQ and IWM are NOT in the DoltHub options dataset (verified empty), so
the real universe is AAPL, SPY, MSFT, NVDA, GOOG, AMD. Earlier "4-symbol" runs were
really 3 (QQQ contributed 0). Raw prices (split-unadjusted) make NVDA/GOOG/AMD usable.

**Long call vs short put, per segment, 2022–2024, real marks, all-in:**

| segment | symbols | LONG CALL | SHORT PUT |
|---|---|---|---|
| index | SPY | PF 1.26 | **PF 2.28** (win 69%, med +12.5%) |
| tech | AAPL/MSFT/GOOG | PF 1.02 | PF 0.95 (loses — tail risk) |
| semi | NVDA/AMD | PF 1.08 | PF 1.17 (win 66%, med +54%) |

**Findings (statistically + economically sound):**
- **Short puts beat long calls almost everywhere** — positive median (you collect the
  variance risk premium) vs long calls' always-negative median.
- **Selling index (SPY) puts is the standout (PF 2.28)** — index puts carry the richest
  VRP (crash-insurance overpay). OOS-robust: train 22-23 PF 1.26 (survives the bear),
  test 24 strongly positive. THE strategy that stuck.
- **Tech is a trap**: short puts lose (PF 0.95, avg −2.2% despite 63% wins — fat left
  tail from growth selloffs); long calls only breakeven. Avoid or use defined-risk.
- **Semis**: short puts edge long calls (PF 1.17 vs 1.08).
- Full 6-symbol short put PF 1.07 (positive but tech dilutes the index/semi edge).

**Actionable:** sell ~25-delta SPY puts, 30–45 DTE, canonical stops. Strongest, most
robust, economically-grounded edge in the whole research effort. Extend to semis;
avoid naked tech short puts (use spreads to cap the tail). Returns are on premium —
real sizing must account for margin/assignment.

## Experiment 4 — defined-risk put spreads + the long/short recommender

**Put credit spread (sell ~25Δ / buy ~10Δ wing), ret on MAX RISK, 2022–2024:**

| segment | naked short put | put credit spread |
|---|---|---|
| index (SPY) | PF 2.28 | **PF 4.29** (defined risk, even better) |
| tech | PF 0.95 | PF 0.63 |
| semi | PF 1.17 | PF 0.58 |

Defining the risk HELPS on the index (the wing is cheap relative to the rich index
VRP → bounded loss lifts PF to 4.29) but KILLS the edge on tech/semi (the wing costs
more than the thinner edge there). So: **index → put spreads (safe + best); semi →
edge only in NAKED short puts (size carefully); tech → stand down.**

**The recommender (`dolt_research --recommend --symbols ...`)** runs long calls,
short puts, and put spreads on real marks and outputs LONG / SHORT / STAND-DOWN by
PF — the verdict IS the backtest. Per-segment verdicts:
- **index → SHORT (sell put spreads)**, PF 4.29
- **semi → SHORT (sell puts)**, PF 1.17 (spread loses, so naked)
- **tech → STAND DOWN** (long call 1.02, short put 0.95, spread 0.63 — nothing clears)

Long calls remain a first-class candidate — the system will pick them whenever they
win for a given name/regime.

## Data trust
`dolt_options --audit` prints per-symbol fetched-days / chain-rows / calls / puts /
date-range with an EMPTY flag. QQQ and IWM show EMPTY (absent from the dataset) — run
it any time to confirm you're not being misled by missing data.

## Open research directions (not yet run)
1. Finer VIX buckets + more symbols (does the overlay's magnitude hold at n>4 names?).
2. A different strategy class on the same real marks: **defined-risk spreads / short
   premium**, which the cost structure may favor over long premium.
3. Underlying-character split (index-like vs high-beta) — flagged but NOT yet tested
   enough to trust (overfitting risk at n=2 pairs).
