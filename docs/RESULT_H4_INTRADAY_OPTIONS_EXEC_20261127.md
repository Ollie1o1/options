# Result — Study H4: Intraday Options Execution Clock & Multi-ETF OPRA Liquidity (`cbbo-1m`)

**Registered:** `docs/PREREG_H4_INTRADAY_OPTIONS_EXEC_20261127.md` (frozen before any OPRA pull)  
**Estimator:** `scripts/backtest_h4_opra_intraday.py`  
**Raw output:** `reports/h4_opra_intraday_20261127T173642.json`  
**Data:** `3,723,644` Databento `OPRA.PILLAR` `cbbo-1m` consolidated NBBO rows (`SPY, QQQ, IWM, XLE, GLD, TLT` × `24` full-session first Wednesdays `2024-01..2025-12` × `5` intraday clocks; total billed cost **`$5.5449`** against the `$125` credit).  
**Date:** 2026-11-27

## Executive Summary — **H4-A & H4-B PASS** (Resolves Open H4 from `OPTIONSDX_RESULTS_20260811.md`)

1. **H4-A (`PASS_TIGHTER`): Early-session `09:45 ET` options quotes impose a massive, statistically crushing spread tax.**
   * On `60–120 DTE` `5%`-wide `Test O1` ETF Bull Puts (`692` obs), mean round-trip friction drops from **`14.31%`** at `09:45 ET` to **`6.28%`** at `12:00 ET` (**`-8.03 pp`**, `98.75% CI [-11.67pp, -4.72pp]`, both `2024` & `2025`) and **`6.57%`** at `15:45 ET` (**`-7.74 pp`**, `CI [-11.42pp, -4.24pp]`).
   * On `20–50 DTE` near-ATM single-leg options (`720` obs), mean relative bid-ask drops **nearly 4×**, from **`8.15%`** at `09:45 ET` to **`2.43%`** at `12:00 ET` and **`2.06%`** at `15:45 ET` (`-6.09 pp`, `CI [-9.85pp, -2.87pp]`).
2. **H4-B (`PASS_WIDER`): Even `10:30 ET` is still measurably wider than `15:45 ET`, while `12:00 ET` and `15:45 ET` are tied for peak liquidity — and `14:00 ET` exhibits severe FOMC/Minutes tail blowouts.**
   * `10:30` vs `15:45`: still **`+2.23 pp` wider on O1** (`98.33% CI [+0.61pp, +3.97pp]`) and **`+1.65 pp` wider on ATM** (`CI [+0.84pp, +2.59pp]`) in both years.
   * `12:00` vs `15:45`: statistically equivalent (`O1: -0.29 pp [-1.21, +0.55]`; `ATM: +0.37 pp [-0.08, +0.93]`) with deepest resting books (`87` contracts median min-depth at `12:00`, `78` at `15:45`, vs only `51` at `09:45`).
   * `14:00` vs `15:45`: median O1 friction is normal (`5.23%`), **but mean O1 friction spikes to `12.83%` (`+6.26 pp`, `CI [+0.43pp, +15.02pp]`, `PASS_WIDER` in both years)** and valid spreads drop from `144` to `131` because first-Wednesday `14:00 ET` slots coincide with scheduled **FOMC rate decisions (`2024-05-01`, `2025-05-07`) and FOMC Minutes (`2024-01-03`)**, when options market makers widen/pull quotes.
3. **H4-C (Multi-ETF OPRA `O1` Liquidity Map at `15:45 ET`):**
   * **Tier 1 (SPY-grade institutional liquidity, 100% O1 gate pass rate):** **`SPY`** (`2.33%` med friction, `100.5` depth), **`QQQ`** (`3.80%`, `41.5`), **`GLD`** (`4.93%`, **`251.5` depth — deepest book!**), **`IWM`** (`5.01%`, `67.0`), and **`TLT`** (`7.15%`, `91.5`).
   * **Tier 2 (Wide / Caution):** **`XLE`** (`12.86%` median crossed friction, `p90 = 25.97%` breaching the `25%` O1 cap, `83.3%` pass rate, `37.0` depth — `5.5×` wider than SPY).

---

## H4-A & H4-B: Intraday Execution Clock Tables (`n = 24` Full-Session First Wednesdays, `2024–2025`)

### Overall Spread & Book Depth by Clock (`Test O1` `60–120 DTE` Bull Puts, 6 ETFs)

| Clock (ET) | Valid O1 Spreads (of 144) | Mean O1 Friction | Median O1 Friction | Median Min Depth (Contracts) | Mean Near-ATM Rel Spread |
|---|---:|---:|---:|---:|---:|
| **`09:45`** | 130 | 14.31% | 7.60% | 51.0 | 8.15% |
| **`10:30`** | 143 | 8.80% | 5.07% | 65.0 | 3.71% |
| **`12:00`** | **144** | **6.28%** | **4.36%** | **87.0** | 2.43% |
| **`14:00`** *(incl. FOMC/Min)* | 131 | 12.83% | 5.23% | 70.0 | 9.12% |
| **`15:45`** | **144** | **6.57%** | 5.01% | **78.0** | **2.06%** |

### Paired Day-Level Bootstrap Tests

| Test | Comparison | Metric | Mean Diff | Bonferroni CI | Halves (`2024` / `2025`) | Verdict |
|---|---|---|---:|---|---|---|
| **H4-A (98.75% CI)** | `10:30` vs `09:45` | O1 friction | `-5.51 pp` | `[-9.48, -1.91] pp` | `-4.72` / `-6.30 pp` | **PASS_TIGHTER** |
| **H4-A (98.75% CI)** | `12:00` vs `09:45` | O1 friction | **`-8.03 pp`** | `[-11.67, -4.72] pp` | `-6.98` / `-9.08 pp` | **PASS_TIGHTER** |
| **H4-A (98.75% CI)** | `14:00` vs `09:45` | O1 friction | `-1.48 pp` | `[-8.63, +7.68] pp` | `+3.01` / `-5.96 pp` | NULL |
| **H4-A (98.75% CI)** | `15:45` vs `09:45` | O1 friction | **`-7.74 pp`** | `[-11.42, -4.24] pp` | `-6.67` / `-8.81 pp` | **PASS_TIGHTER** |
| **H4-A (98.75% CI)** | `15:45` vs `09:45` | Near-ATM rel spread | **`-6.09 pp`** | `[-9.85, -2.87] pp` | `-4.90` / `-7.28 pp` | **PASS_TIGHTER** |
| **H4-B (98.33% CI)** | `10:30` vs `15:45` | O1 friction | `+2.23 pp` | `[+0.61, +3.97] pp` | `+1.96` / `+2.51 pp` | **PASS_WIDER** |
| **H4-B (98.33% CI)** | `12:00` vs `15:45` | O1 friction | `-0.29 pp` | `[-1.21, +0.55] pp` | `-0.31` / `-0.27 pp` | NULL (tied best) |
| **H4-B (98.33% CI)** | `14:00` vs `15:45` | O1 friction | `+6.26 pp` | `[+0.43, +15.02] pp` | `+9.68` / `+2.84 pp` | **PASS_WIDER** |
| **H4-B (98.33% CI)** | `10:30` vs `15:45` | Near-ATM rel spread | `+1.65 pp` | `[+0.84, +2.59] pp` | `+1.61` / `+1.69 pp` | **PASS_WIDER** |
| **H4-B (98.33% CI)** | `12:00` vs `15:45` | Near-ATM rel spread | `+0.37 pp` | `[-0.08, +0.93] pp` | `+0.31` / `+0.42 pp` | NULL (tied best) |
| **H4-B (98.33% CI)** | `14:00` vs `15:45` | Near-ATM rel spread | `+7.06 pp` | `[+0.09, +17.99] pp` | `+7.85` / `+6.26 pp` | **PASS_WIDER** |

---

## H4-C: Multi-ETF `Test O1` Institutional OPRA Liquidity Map (`15:45 ET`, `n = 24` sessions)

| ETF | `n` | Median Crossed Friction | p90 Crossed Friction | `<= 25%` Gate Pass Rate | Median Min Depth (Contracts) | OPRA Execution Tier |
|---|---:|---:|---:|---:|---:|---|
| **SPY** | 24 | **2.33%** | 4.55% | **100.0%** | 100.5 | **Tier 1 (Elite)** |
| **QQQ** | 24 | **3.80%** | 6.17% | **100.0%** | 41.5 | **Tier 1 (Elite)** |
| **GLD** | 24 | **4.93%** | 7.70% | **100.0%** | **251.5** | **Tier 1 (Elite)** |
| **IWM** | 24 | **5.01%** | 6.96% | **100.0%** | 67.0 | **Tier 1 (Elite)** |
| **TLT** | 24 | **7.15%** | 9.59% | **100.0%** | 91.5 | **Tier 1 (Strong)** |
| **XLE** | 24 | **12.86%** | **25.97%** | 83.3% | 37.0 | **Tier 2 (Wide / Caution)** |

## Operational Changes Implemented

1. **Live Execution Timing Guard (`src/radar/live_options.py`, `src/tui/options_engine.py`):** Added `options_execution_window_guidance()` flagging `09:30–10:30 ET` (`EXEC_OPEN_WIDE_WAIT_1130`) and `13:55–14:15 ET` (`EXEC_1400_CHECK_FOMC`), while highlighting `11:30–13:30 ET` and `15:15–15:50 ET` as `OPTIMAL_OPRA_WINDOW`, plus attaching OPRA H4-C liquidity tier metadata (`OPRA_O1_LIQUIDITY_TIER`).
2. **Runbook Updated (`docs/GO_LIVE_RUNBOOK.md`):** Codified optimal IBKR options execution windows (`11:30–13:30 ET` / `15:15–15:50 ET`, never `09:30–10:30` or `14:00` FOMC) alongside Study 2E's TFSA shares rebalance guidance (`10:00–15:55 ET`).
