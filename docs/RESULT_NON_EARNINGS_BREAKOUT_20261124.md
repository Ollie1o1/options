# Equity Playbook: Non-Earnings Volume Breakout (Shares) — RESULT

**Date:** 2026-11-24
**Preregistration:** `docs/PREREG_NON_EARNINGS_BREAKOUT_20261124.md` (committed `39f9773` prior to running estimator)
**Estimator:** `scripts/backtest_non_earnings_breakout.py`
**Binding Verdict:** **`CLOSED`** (failed both H-A and H-B across all 3 pre-registered horizons on both strata)

---

## Summary Table — Locked Out-of-Sample Holdout (`2023-02-07` to `2026-08-07` / `2026-11-24`)

Inference: circular moving-block bootstrap (`block_len = 4` ISO weeks, `10,000` resamples, `seed = 20261124`, Bonferroni `alpha = 0.01667` = `98.33%` two-sided CI), `10 bps` round-trip shares friction, next-day close entry (`t+1`).

### 1. Primary Stratum — Broad Survivorship-Free Universe (`data/squeeze_prices.db`, `13,663` symbols scanned)

| Horizon | n trades | Weeks | Win% | Mean Net | 98.33% CI | PF | PF 98.33% CI | Excess vs SPY | Excess 98.33% CI | 5-Slot Book CAGR (SPY) | H-A | H-B |
|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|---|---|
| **5d** | 5,335 | 180 | 49.7% | +0.02% | `[-0.64%, +0.67%]` | 1.008 | `[0.796, 1.274]` | **−0.34%** | `[-0.87%, +0.19%]` | **−29.5%** (+20.1%) | FAIL | FAIL |
| **10d** | 5,156 | 180 | 51.7% | +0.64% | `[-0.44%, +1.68%]` | 1.186 | `[0.892, 1.579]` | **−0.08%** | `[-0.85%, +0.70%]` | **−17.0%** (+20.1%) | FAIL | FAIL |
| **20d** | 4,865 | 179 | 50.8% | +0.89% | `[-1.04%, +2.66%]` | 1.187 | `[0.822, 1.672]` | **−0.56%** | `[-1.82%, +0.73%]` | **−8.2%** (+20.8%) | FAIL | FAIL |

### 2. Secondary Stratum — Curated Liquid Adjusted Universe (`data/equity_ohlcv.db`, `371` symbols)

| Horizon | n trades | Weeks | Win% | Mean Net | 98.33% CI | PF | PF 98.33% CI | Excess vs SPY | Excess 98.33% CI | 5-Slot Book CAGR (SPY) | H-A | H-B |
|---:|---:|---:|---:|---:|---|---:|---|---:|---|---:|---|---|
| **5d** | 838 | 184 | 48.9% | +0.33% | `[-0.43%, +1.04%]` | 1.163 | `[0.816, 1.590]` | **−0.04%** | `[-0.74%, +0.60%]` | +7.6% (+17.9%) | FAIL | FAIL |
| **10d** | 822 | 184 | 54.3% | +1.05% | `[-0.24%, +2.19%]` | 1.417 | `[0.923, 2.106]` | +0.39% | `[-0.70%, +1.35%]` | +22.1% (+17.5%) | FAIL | FAIL |
| **20d** | 776 | 182 | 53.9% | +2.00% | `[-0.33%, +4.12%]` | 1.563 | `[0.931, 2.530]` | +0.48% | `[-1.35%, +2.26%]` | +19.7% (+17.8%) | FAIL | FAIL |

### 3. Diagnostic — Earnings-Calendar-Covered Subset Only (Holdout)
- On the smaller subset covered by `data/dolt_options.db::earnings_cal` (`n = 267–283` broad, `n = 200–216` liquid), point estimates are higher (`20d` broad PF `1.504`, liquid PF `1.889`), **but every single Bonferroni 98.33% CI still crosses zero and `PF = 1.0`** (`broad 20d PF CI [0.816, 2.834]`, `liquid 20d PF CI [0.975, 3.443]`, `excess CI [-1.73%, +4.97%]`).

---

## Key Takeaway (Why This Protects Capital)

1. **Classic survivorship-bias illusion:** On the curated 371-ticker current-survivor universe, chasing `+5% / 2× vol` breakouts superficially looks like a `PF 1.42–1.56` momentum play — yet even there it fails significance (`PF CI_lo <= 0.93`).
2. **Real-world survivorship-free failure:** Across the full 13,663-symbol survivorship-free market (`$10+`, `$20M+/day`), chasing `+5% / 2× vol` breakouts underperforms SPY at **every** horizon (`-0.08%` to `-0.56%` per trade) and produces **`-8.2%` to `-29.5%` CAGR** in a 5-slot capacity book (taking the highest relative-volume spikes first) with `-35%` to `-71%` drawdowns while SPY compounded at `+20%/yr`.
3. **Action Taken:** Registered `non_earnings_breakout` as **`CLOSED`** in `configs/playbooks.json`. Do **not** surface single-day `+5% / 2× vol` breakout chasing as an actionable buy signal on Radar; stay focused on validated Core TFSA Sector Rotation (`vol_managed_outlook`), shares-only Uptrend Pullbacks (`equity_pullback`), and limit-only 60–120d 5%-wide ETF spreads (`etf_wide_spread`).
