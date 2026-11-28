# Result — Study 5A: 11-Year Institutional OPRA Test (`2015–2026`) of `TLT`, `QQQ`, and `IWM` `60–120 DTE` Spreads

**Registered:** `docs/PREREG_OPRA_10YR_TLT_QQQ_IWM_20261127.md` (frozen before incremental `2015–2023` download)  
**Estimator:** `scripts/backtest_opra_10yr_etfs.py`  
**Raw Output:** `reports/backtest_opra_10yr_etfs_20261128T001236.json`  
**Data:** `613` weekly full-session Wednesdays (`2015-01-07 .. 2026-09-30`, `15:45 ET` `OPRA.PILLAR cbbo-1m`), `3,232` institutional spreads across `TLT`, `QQQ`, and `IWM` (`469` incremental `2015–2023` dates downloaded for **`$7.0164`**; total cumulative Databento spend **`$20.1069` / `$125.00`**, leaving **`$104.89`** unused).  
**Date:** 2026-11-28

## Executive Summary — **All 6 Non-SPY Arms `CLOSED`; `TLT` 2024–2026 Outperformance Was a Regime Illusion; `SPY` Remains the Sole `O1` Lead**

Testing across a full 11.7-year rate and equity cycle (including 2018 vol shock, 2020 COVID crash, and the 2022–2023 `-45%` bond bear market) decisively **CLOSED** all six `{TLT, QQQ, IWM} × {Bull Put, Bear Call}` `60–120 DTE` arms:

| Arm (`60–120 DTE` `O1-M`) | `n` (`effN`) | Win % `[95% CI]` | Mean RoC | PF `[99.17% CI]` | Avg Loss (Worst 5%) | Δ-Adj `x_lim` `[99.17% CI]` | `H1 (15–20)` | `H2 (21–26)` | Crossed `x_crs` | Med Fric | Verdict |
|---|---:|---|---:|---|---|---|---:|---:|---:|---:|---|
| **`TLT` Bull Put** | 487 (87) | `80.1% [76.3–83.4%]` | **-0.55%** | `0.94 [0.48, 2.51]` | **-50.2%** (-80.9%) | `+0.25% [-4.14, +4.06%]` | +1.22% | **-0.64%** | -0.84% | 10.5% | **CLOSED** |
| **`TLT` Bear Call** | 377 (83) | `85.9% [82.1–89.1%]` | **-0.53%** | `0.92 [0.37, 4.03]` | **-48.4%** (-66.8%) | `-0.52% [-4.51, +2.51%]` | -3.15% | +1.62% | -1.57% | 12.3% | **CLOSED** |
| **`QQQ` Bull Put** | 596 (100) | `89.1% [86.3–91.4%]` | +5.63% | `2.20 [1.09, 7.10]` | -42.9% (-46.9%) | `+0.31% [-2.07, +2.27%]` | +0.70% | **-0.11%** | -0.16% | 4.3% | **CLOSED** |
| **`QQQ` Bear Call** | 587 (89) | `67.5% [63.6–71.1%]` | **-8.22%** | `0.51 [0.28, 0.93]` | **-51.1%** (-84.6%) | `-0.68% [-4.74, +2.99%]` | -3.21% | +1.99% | -1.26% | 3.8% | **CLOSED** |
| **`IWM` Bull Put** | 592 (95) | `84.8% [81.7–87.5%]` | +2.02% | `1.25 [0.63, 3.43]` | **-53.7%** (-83.8%) | **`-1.01% [-4.75, +1.97%]`** | -0.68% | -1.37% | -1.68% | 5.8% | **CLOSED** |
| **`IWM` Bear Call** | 593 (98) | `76.6% [73.0–79.8%]` | **-1.34%** | `0.87 [0.48, 1.80]` | -44.3% (-66.7%) | `+0.54% [-2.36, +2.69%]` | -0.86% | +2.05% | -0.03% | 5.5% | **CLOSED** |

Monthly Bull Put return correlations (`k=0.35`): `TLT–QQQ = 0.247`, `TLT–IWM = 0.166`, `QQQ–IWM = 0.730`.

## Key Diagnostic Insights (Why This `$7.02` Study Prevented Major Tail Losses)

1. **`TLT` Bull Put 2024–2026 Outperformance Was Regime Luck:** In Study 3A (`2024–2026`), `TLT` Bull Put beat delta (`+2.09%`, `+6.20%`, `+3.24%` by year). Extending back through the 2022–2023 rate-hike cycle exposed massive bond-trend convexity losses: **`2022` Win `48.7%`, Mean RoC `-25.4%`, delta-adj `-13.25%`**; **`2023` Win `72.5%`, Mean RoC `-7.81%`, delta-adj `-4.05%`**. Over the full 11.7 years, `TLT` Bull Put lost money outright (`-0.55%` RoC, `PF 0.945`, median friction `10.5%`).
2. **All Three `60–120 DTE` Bear Calls Lost Raw Money (`PF 0.51–0.92`):** Even with `68%–86%` win rates, right-tail rallies inflicted `-44%` to `-51%` average losses (`QQQ` Bear Call lost `-8.22%` per trade, `PF 0.505`).
3. **`QQQ` & `IWM` Bull Puts Fail to Beat Their Own Linear Delta:** `IWM` Bull Put lagged delta in **both halves** (`-0.68%` in `2015–20`, `-1.37%` in `2021–26`). `QQQ` Bull Put earned `+5.63%` raw RoC (`PF 2.20`) almost entirely from NASDAQ bull beta (`x_lim = +0.31% [-2.07, +2.27%]`, slightly negative in `2021–26` at `-0.11%`, negative crossed at `-0.16%`).
4. **Operational Action:** **`SPY` (`2010–2023` `+1.59% [+0.26%, +2.67%]` at `k=0.35` limit, positive 13/14 yrs) remains the ONLY ETF with a validated `60–120 DTE` `O1` Bull Put lead.** `src/topdown/best_action.py` now strictly prioritizes `SPY` for `60–120d (O1)` and stamps `TLT`/`QQQ`/`IWM` with `STUDY5A_11YR_CLOSED`.
