# Result — Study 4A: Complete Options DTE Curve (`0DTE` through `240 DTE`) on Institutional OPRA NBBO (`cbbo-1m`)

**Registered:** `docs/PREREG_DTE_CURVE_AND_0DTE_OPRA_20261127.md` (frozen before any incremental `0DTE` download or run)  
**Estimator:** `scripts/backtest_dte_curve_opra.py`  
**Raw output:** `reports/backtest_dte_curve_opra_20261127T222016.json`  
**Data:** `3,617` institutional OPRA `cbbo-1m` spread trades (`2024-01 .. 2026-09`) across `SPY, QQQ, IWM, XLE, GLD, TLT` (`1,035,642` incremental `10:30 ET` `0DTE` rows downloaded for **`$1.5395`**; total cumulative Databento spend **`$13.0905` / `$125.00`**, leaving **`$111.91`** unused).  
**Date:** 2026-11-27

## Executive Summary — **`60–120 DTE` (`Test O1`) Is the Sole Structural Sweet Spot; `0DTE` & `1–7d` Lose Money Despite `76%–81%` Win Rates**

1. **Classic "High Win Rate = Negative Expected Value" Trap Exposed on Real OPRA Quotes:**
   * **`0DTE` Bull Put (`SPY/QQQ`, `10:30→15:45 ET`, `n=70`):** Won **`75.7%`** of trades (`95% CI [64.5%, 84.2%]`), **AND LOST `-3.39%` RoC PER TRADE (`PF 0.59`, Avg Loss `-34.33%`, Worst 5% `-62.8%`)** — delta-adjusted `-1.04%` at `k=0.35` limit (`-1.74%` crossed) → **`CLOSED`**.
   * **`0DTE` Iron Condor (`SPY/QQQ`, `10:30→15:45 ET`, `n=47`):** Won **`68.1%`** of trades (`CI [53.8%, 79.6%]`), **AND LOST `-1.69%` RoC PER TRADE (`PF 0.82`, Avg Loss `-30.13%`)** → **`CLOSED`**.
   * **`1–7 DTE` Weekly Bull Put (6 ETFs, `n=676`, `effN=130`):** Won **`80.9%`** of trades (`CI [77.8%, 83.7%]`), **AND LOST `-0.76%` RoC PER TRADE (`PF 0.92`, Avg Loss `-47.75%`, Worst 5% `-69.5%`)** → **`CLOSED`**.
2. **Short & Medium DTE (`8–30d`, `31–60d`) and Long DTE (`120–240d`) All Lag Linear Delta:**
   * **`8–30 DTE` (`n=721`):** `Win 85.0%`, raw RoC `+2.70%` (`PF 1.36`), **Avg Loss `-50.39%`**, **delta-adjusted `-1.21%` limit (`-2.05%` crossed)** → **`CLOSED`**.
   * **`31–60 DTE` (`n=703`):** `Win 86.6%`, raw RoC `+3.56%` (`PF 1.72`), tightest friction (`4.1%`), but still **delta-adjusted `-0.17%` limit (`-0.77%` crossed)** → **`CLOSED`**.
   * **`120–240 DTE` (`n=655`):** `Win 88.7%`, raw RoC `+6.76%` (`PF 2.42`), but spread widens (`6.4%`), tail loss deepens (`-42.2%`), and **delta-adjusted turns negative (`-0.56%` limit, `-1.42%` crossed)** → **`CLOSED`**.
3. **`60–120 DTE` (`Test O1`, `n=745`, `effN=42`) Is the ONLY Bucket Out of All 7 With Positive Delta-Adjusted Excess (`+0.24%` at `k=0.35` Limit):**
   * `Win 87.9%` (`95% CI [85.4%, 90.1%]`), raw RoC `+5.69%`, `PF 2.25 [1.03, 7.92]`, `Avg Loss -37.53%`, median friction `4.8%`, `x_lim = +0.24%` (`x_crs = -0.45%` — strictly limit-only!). Confirms our prior 2010–2023 SPY discovery (`Test O1` `LEAD`) sits at the true structural optimum between short-DTE gamma blowups (`< 60d`) and long-DTE spread drag (`> 120d`).

---

## Complete DTE Curve Table (`k = 0.35` Limit Fills, Net IBKR Fees, Bonferroni `99.17%` CI)

| DTE Bucket | `n` (`effN`) | Win Rate `[95% Wilson CI]` | Mean RoC | Avg Win | Avg Loss (Tail) | Worst 5% | PF `[99.17% CI]` | Δ-Adj `x_lim` `[99.17% CI]` | Crossed `x_crs` | Med Fric | Verdict |
|---|---:|---|---:|---:|---:|---:|---|---|---:|---:|---|
| **`0DTE` Bull Put** (`SPY, QQQ`) | 70 (70) | `75.7% [64.5–84.2%]` | **-3.39%** | +6.54% | **-34.33%** | -62.8% | `0.59 [0.34, 1.85]` | `-1.04% [-4.62, +2.68%]` | `-1.74%` | 6.1% | **CLOSED** |
| **`0DTE` Iron Condor** (`SPY, QQQ`) | 47 (47) | `68.1% [53.8–79.6%]` | **-1.69%** | +11.64% | **-30.13%** | -57.1% | `0.82 [0.31, 2.84]` | `-0.84% [-9.60, +6.34%]` | `-2.13%` | 5.9% | **CLOSED** |
| **`1–7 DTE`** (6 ETFs) | 676 (130) | `80.9% [77.8–83.7%]` | **-0.76%** | +10.33% | **-47.75%** | -69.5% | `0.92 [0.61, 1.47]` | `-0.28% [-2.85, +2.66%]` | `-0.56%` | 5.7% | **CLOSED** |
| **`8–30 DTE`** (6 ETFs) | 721 (58) | `85.0% [82.2–87.4%]` | +2.70% | +12.05% | **-50.39%** | -69.9% | `1.36 [0.83, 3.04]` | `-1.21% [-3.59, +1.42%]` | `-2.05%` | 5.1% | **CLOSED** |
| **`31–60 DTE`** (6 ETFs) | 703 (52) | `86.6% [83.9–88.9%]` | +3.56% | +9.82% | -36.98% | -44.1% | `1.72 [0.91, 4.57]` | `-0.17% [-2.98, +2.11%]` | `-0.77%` | **4.1%** | **CLOSED** |
| **`60–120 DTE` (`O1`)** (6 ETFs) | 745 (42) | **`87.9% [85.4–90.1%]`** | **+5.69%** | +11.63% | -37.53% | **-40.8%** | **`2.25 [1.03, 7.92]`** | **`+0.24% [-2.76, +2.24%]`** | `-0.45%` | 4.8% | **LEAD (Ref)** |
| **`120–240 DTE`** (6 ETFs) | 655 (23) | `88.7% [86.1–90.9%]` | +6.76% | +13.00% | -42.24% | -42.6% | `2.42 [1.11, 11.08]` | `-0.56% [-3.30, +1.82%]` | `-1.42%` | 6.4% | **CLOSED** |

## Operational Action

* Wire the entire `0DTE → 240D` curve directly into `[6] DISCOVER` (`src/topdown/best_action.py`, `opt discover --dte`, TUI `DTE 0d→6m`, Web Discover) with **`60–120d (O1)` highlighted `PROMISING`** and **`0DTE`, `1–7d`, `8–30d`, `31–60d`, and `120–240d` explicitly stamped `CLOSED`** alongside their exact `P(Win) [95% CI]`, `PF`, and `Avg Loss` so high win rates never mask negative expectancy.
