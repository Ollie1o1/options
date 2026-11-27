# Result — V1: intraday volatility for the VM_OUTLOOK_TOP3 sleeve

**Run:** 2026-11-27 · **Registration:** `docs/PREREG_SLEEVE_INTRADAY_VOL_20261127.md` (frozen
`aa52fb5` before the 5-minute data was downloaded; estimator committed `47372dc` before it ran).
**Script:** `python -m scripts.sleeve_vol_forecast_test` · raw output
`reports/intraday/sleeve_vol_forecast_20261127T140338.json` (local, gitignored; numbers below).
**Data:** Alpaca SIP 5-minute bars, split + dividend adjusted, 16 sleeve ETFs, 2016-01 .. 2026-11
(2,720 full sessions; 99.2% of ticker-sessions valid, except XLC before its 2018-06 launch and
XLRE in 2016, 29% valid).

## Verdict: both claimants FAIL — the live sleeve keeps RV20

**Parity first:** the re-simulated sleeve with E0 reproduces the validated backtest's
`VM_OUTLOOK_TOP3` period returns exactly (max |diff| 0.0 over all 250 months; full-window
Sharpe 0.819 = recorded).

Window 2016-02-08 .. 2026-10-16, 129 rebalance months. QLIKE loss on next-month basket
variance; D = loss(E0) − loss(Ek), positive = Ek better; circular block bootstrap (3 months),
Bonferroni 97.5% CI.

| | n | mean QLIKE E0 → Ek | mean D | 97.5% CI | months better | 2016–20 | 2021–26 | verdict |
|---|---:|---|---:|---|---:|---:|---:|---|
| **E1** intraday RV20 | 129 | 0.509 → 0.438 | +0.071 | [−0.019, +0.158] | 62% | +0.013 | +0.120 | **FAIL** (CI) |
| **E2** HAR on intraday RV | 103 | 0.537 → 0.427 | +0.110 | [−0.209, +0.365] | 49% | −0.024 | +0.210 | **FAIL** (CI, halves) |
| E3 EWMA daily (diagnostic) | 129 | 0.509 → 0.427 | +0.083 | [+0.012, +0.169] | 58% | +0.060 | +0.102 | no authority |

Log-vol bias (forecast vs realised): E0 −0.04, E1 −0.02, E2 +0.15, E3 −0.00. Correlation of
log forecast with log realised: E0 0.56, E1 0.60, E2 0.51, E3 0.58.

## What each estimator would have done to the sleeve (no authority; the adoption check)

| | CAGR | vol | Sharpe | max DD | avg scale | months scaled | \|vol − 14%\| on scaled months |
|---|---:|---:|---:|---:|---:|---:|---:|
| **E0 (live)** | +11.55% | 13.13% | **0.903** | **−16.58%** | 0.83 | 74 | 4.84% |
| E1 | +11.44% | 13.32% | 0.885 | −19.41% | 0.82 | 78 | 4.86% |
| E2 (E0 fallback in 26 months) | +11.49% | 13.17% | 0.898 | −22.18% | 0.78 | 102 | 4.63% |
| E3 | +11.33% | 12.88% | 0.903 | −17.68% | 0.82 | 77 | 4.63% |

No claimant passed, so the adoption rule is not reached; even if it were, E1 and E2 would fail
its drawdown condition (−2.8 pp and −5.6 pp worse than E0).

## Reading

- Intraday data **does** measure volatility better (E1's loss is lower in 62% of months, its
  bias is smaller), but over a one-month horizon the gain is not distinguishable from noise, and
  a better σ̂ does **not** produce a better sleeve: realised-vs-target error is ≈ 4.6–4.9 pp for
  every estimator. What limits vol targeting here is how much volatility changes within the
  month after the scale is set, not how precisely the trailing month was measured.
- The best-looking forecaster is E3 — an EWMA on the daily closes the sleeve already has — and it
  too leaves Sharpe unchanged and the drawdown slightly worse. It was a diagnostic; promoting it
  would need its own registration, and the sleeve numbers give no reason to.
- **CLOSED:** the sleeve's volatility estimator is not where its performance can be improved.
  The live sleeve, `src/outlook/vm_rotation.py`, is unchanged.
