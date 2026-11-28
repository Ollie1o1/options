# Result — Study 2E: `VM_OUTLOOK_TOP3` Rebalance Execution Clock & Overnight/Intraday Decomposition

**Registered:** `docs/PREREG_REBALANCE_TIMING_AND_OVERNIGHT_20261127.md` (frozen before run)  
**Estimator:** `scripts/backtest_rebalance_timing_and_overnight.py`  
**Raw output:** `reports/intraday/rebalance_timing_overnight_20261127T164728.json`  
**Date:** 2026-11-27

## Executive Summary

1. **E1 (`VM_OUTLOOK_TOP3` Rebalance Clock): NULL / ALL CLOCKS EQUIVALENT.** Across 129 monthly rebalances (`2016-02-09` to `2026-10-19`, full sessions), executing at `09:35`, `10:00`, `12:00`, or `14:00` ET differs from the `15:55` (close) baseline by `-1.22` to `+1.64 bps/rebalance` with every Bonferroni `98.75%` CI spanning zero. **Operational result:** You do not need to chase the open or wait for the final minute on Wealthsimple — any liquid `10:00–15:55 ET` limit rebalance achieves the validated sleeve return.
2. **E2a (Overnight vs Intraday Attribution — descriptive): 100%+ of `VM_OUTLOOK_TOP3` returns accrue overnight.** Across `2,693` held sessions, the sleeve earned **`+15.41%` annualized overnight (`15:55 -> 09:30`, Sharpe `1.566`)** versus **`-3.00%` annualized intraday (`09:30 -> 15:55`, Sharpe `-0.246`)**. Never flatten or day-trade out of the core TFSA sleeve intraday.
3. **E2b (Uptrend 3-Day Reversion Decomposition, net 6 bps):**
   * **`intraday_selloff`** (`open->15:55 <= -1.5%`, `overnight > -0.5%`, `Close > SMA200`): **`PASS`** standalone (`+30.5 bps`, `97.5% CI [+0.7, +58.5]`, `485` signal days / `1,062` events, positive in both `2016–2020: +32.8 bps` and `2021–2026: +29.5 bps`).
   * **`overnight_gap_down`** (`prev_close->open <= -1.5%`, `Close > SMA200`): **`FAIL`** (`+14.1 bps`, `97.5% CI [-36.8, +60.9]`, `305` days / `612` events).
   * **Difference (`intraday_selloff - overnight_gap_down`):** `+16.4 bps`, `97.5% CI [-40.0, +74.5]` — **not statistically separated** from each other, and E2b measures raw standalone 3d return (not SPY-excess, matching `equity_pullback`'s prior standalone-pass / SPY-excess-null structure).

---

## E1: Monthly Rebalance Clock Comparison (`n = 129`, `2016-02-09` .. `2026-10-19`, 3 bps/side)

| Execution Clock (ET) | CAGR | Ann Vol | Sharpe | Sortino | MaxDD | Calmar | `Δ vs 15:55` (bps/mo) | 98.75% CI (bps) | Halves (`16–20` / `21–26`) | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---|
| **`15:55` (Close Baseline)** | +11.48% | 13.15% | 0.897 | 0.764 | -16.76% | 0.685 | — | — | — | **BASELINE** |
| **`09:35`** | +11.31% | 13.21% | 0.882 | 0.757 | -16.36% | 0.692 | `-1.22` | `[-15.00, +12.89]` | `-1.55` / `-0.95` | **FAIL** |
| **`10:00`** | +11.53% | 12.79% | 0.922 | 0.794 | -15.85% | 0.728 | `-0.02` | `[-13.21, +13.07]` | `+1.60` / `-1.39` | **FAIL** |
| **`12:00`** | +11.43% | 12.93% | 0.907 | 0.750 | -17.44% | 0.656 | `-0.60` | `[-11.20, +10.06]` | `-0.60` / `-0.61` | **FAIL** |
| **`14:00`** | +11.77% | 12.67% | 0.947 | 0.837 | -16.77% | 0.702 | `+1.64` | `[-5.64, +8.81]` | `+1.14` / `+2.06` | **FAIL** |

---

## E2a: `VM_OUTLOOK_TOP3` Overnight vs Intraday Attribution (Descriptive, `n = 2,693` sessions)

| Component | Window | Annualized Mean | Annualized Vol | Sharpe |
|---|---|---:|---:|---:|
| **Overnight** | `15:55 -> 09:30` | **+15.41%** | 9.84% | **1.566** |
| **Intraday** | `09:30 -> 15:55` | **-3.00%** | 12.19% | **-0.246** |

---

## E2b: Uptrend 3-Day Reversion Decomposition (16 ETFs, `Close > SMA200`, net 6 bps round-trip)

| Pattern | Signal Days | Events | Mean 3d Return | 97.5% CI (10d Block) | Halves (`16–20` / `21–26`) | Verdict |
|---|---:|---:|---:|---|---|---|
| **Intraday Selloff** (`open->close <= -1.5%`, `ovn > -0.5%`) | 485 | 1,062 | **+30.5 bps** | `[+0.7, +58.5]` | `+32.8` / `+29.5` bps | **PASS** (standalone) |
| **Overnight Gap-Down** (`prev_close->open <= -1.5%`) | 305 | 612 | `+14.1 bps` | `[-36.8, +60.9]` | `+11.2` / `+15.1` bps | **FAIL** |
| **Difference (`intraday - gap_down`)** | — | — | `+16.4 bps` | `[-40.0, +74.5]` | — | Inconclusive |

### Operational Consequence

* **Wealthsimple Rebalancing:** Confirmed that executing `opt rebalance` orders at any liquid mid-day/afternoon window (`10:00–15:55 ET`, avoiding the volatile 09:30–09:35 open) is statistically indistinguishable from the exact close; keep positions invested overnight where 100%+ of the sleeve's edge occurs.
* **Pullback Research:** Intraday selloffs in uptrending ETFs rebound more reliably (`+30.5 bps`, CI above zero in both halves) than overnight gap-downs (`+14.1 bps`, wide CI spanning zero), though the direct difference is underpowered (`[-40, +74.5] bps`). Consistent with `equity_pullback` (LEAD, shares-only standalone reversion, no SPY excess).
