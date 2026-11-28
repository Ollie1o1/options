# Result — Study 5B: Single-Stock Intraday Selloff vs. Overnight Gap-Down Filter on `equity_pullback`

**Registered:** `docs/PREREG_PULLBACK_INTRADAY_VS_GAP_20261127.md` (frozen before execution)  
**Estimator:** `scripts/backtest_pullback_intraday_vs_gap.py`  
**Raw Output:** `reports/backtest_pullback_intraday_vs_gap_20261127T224357.json`  
**Data:** `10,893` `equity_pullback` trades across `375` liquid stocks (`data/equity_ohlcv.db`, `2018–2026`, `10 bps` round-trip, next-day fill), cross-verified on `8,131` trades matched to free Alpaca split-adjusted daily `Open`/`Close` bars (`$0` spend).  
**Date:** 2026-11-27

## Verdict — **`CLOSED` (Both H5B-1 & H5B-2 FAILED With the Opposite Sign on Holdout — Do NOT Filter Out Overnight Gap-Downs)**

Contrary to the ETF 5-minute `E2b` pattern, excluding overnight gap-downs from single-stock `equity_pullback` **hurt** holdout performance under both the offline `OHLCV`-native definition and exact Alpaca `Open`/`Close` bars:

| Classification | Split | `n` | Win Rate | Mean Net Ret (10 bps) | PF | Avg Loss |
|---|---|---:|---:|---:|---:|---:|
| **`ALL` Baseline** | Holdout (`2023–26`) | 5,452 | 61.5% | +0.46% | 1.34 | -3.48% |
| **`TRUE_GAP_DOWN`** (`high < prev_close`) | Train (`2018–22`) | 2,806 | 61.2% | +0.27% | 1.20 | -3.59% |
| **`TRUE_GAP_DOWN`** (`high < prev_close`) | **Holdout (`2023–26`)** | **2,893** | **62.5%** | **+0.58%** | **1.43** | -3.54% |
| `NON_GAP` (`high >= prev_close`) | Holdout (`2023–26`) | 2,559 | 60.5% | +0.32% | 1.24 | -3.42% |
| `INTRADAY_FLUSH` (`close/high <= -2.5%`) | Holdout (`2023–26`) | 1,366 | 60.6% | +0.39% | 1.25 | -3.93% |
| **`OVERNIGHT_GAP_TRAP`** (Alpaca `gap <= -1.5%`) | Train (`2018–22`) | 846 | 63.0% | +0.56% | 1.39 | -3.86% |
| **`OVERNIGHT_GAP_TRAP`** (Alpaca `gap <= -1.5%`) | **Holdout (`2023–26`)** | **1,253** | **62.0%** | **+0.80%** | **1.51** | -4.17% |
| `NO_BIG_GAP` (Alpaca `gap > -1.5%`) | Holdout (`2023–26`) | 3,646 | 61.5% | +0.36% | 1.29 | -3.29% |
| `INTRADAY_CLEAN` (Alpaca `intra <= -1.5%`) | Holdout (`2023–26`) | 2,231 | 61.1% | +0.34% | 1.23 | -3.72% |

### Binding Hypothesis Checks (Bonferroni `97.5%` ISO-Week Cluster Bootstrap CI)

* **H5B-1 (`NON_GAP - TRUE_GAP_DOWN`):** **`FAIL`** — Holdout **`-0.256% [-0.785%, +0.283%]`** (Train `+0.071% [-0.426%, +0.566%]`).
* **H5B-2 (`INTRADAY_FLUSH - TRUE_GAP_DOWN`):** **`FAIL`** — Holdout **`-0.184% [-0.891%, +0.499%]`** (Train `+0.228% [-0.317%, +0.756%]`).
* **Alpaca Exact-Open Verification:** `NO_BIG_GAP - OVERNIGHT_GAP_TRAP` Holdout **`-0.439% [-1.125%, +0.270%]`** (Train `-0.044%`); `INTRADAY_CLEAN - OVERNIGHT_GAP_TRAP` Holdout **`-0.469% [-1.211%, +0.281%]`** (Train `+0.023%`).

## Why This Saved Money

Because `equity_pullback` already removes upcoming earnings (`10d` blackout), requires strong uptrends (`Close > SMA200`, `12-1m > 0`), and enters on the **next day's close (`t+1`)** after the gap panic has already absorbed forced selling, non-earnings gap-downs in healthy uptrends actually mean-revert strongly (`PF 1.43–1.51`). Adding an intuitive gap-down exclusion filter would have thrown away the highest-PF segment of the strategy. Live `equity_pullback` rules remain unchanged.
