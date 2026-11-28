# Result — Study 4B: Walk-Forward Empirical Probability Calibration for `[6] DISCOVER`

**Registered:** `docs/PREREG_PROBABILITY_CALIBRATION_20261127.md` (frozen before execution)  
**Estimator:** `scripts/backtest_probability_calibration.py`  
**Committed UI Artifact:** `configs/calibrated_evidence.json`  
**Raw Audit Report:** `reports/probability_calibration_20261127T222813.json`  
**Date:** 2026-11-27

## Executive Summary — What Actually Calibrates (and What Does Not)

1. **`C1: VM_OUTLOOK_TOP3` Core Sleeve (2006–2026, `n=250` months) — `CALIBRATED` (`ECE <= 2.04%`):**
   * Strictly out-of-sample expanding walk-forward (`min_train=60` months) cleared the `ECE <= 0.05` calibration gate at both `1m` (`ECE = 0.0179`, `n=190`) and `3m` (`ECE = 0.0204`, `n=188`).
   * **Verified win odds:**
     * `1m > 0`: **`63.6% [57.5–69.3%]`** (`n=250`, mean `+0.86%`, `PF 1.83`, AvgLoss `-2.83%`; `CALM` RV20 `<= 14%` `PF 2.51` / AvgLoss `-2.06%` vs `ELEVATED` `PF 1.48` / AvgLoss `-3.52%`).
     * `3m > 0`: **`70.2% [64.2–75.5%]`** (`n=248`, mean `+2.68%`, **`PF 3.49`**, AvgLoss `-3.61%`; `CALM 72.0%` vs `ELEVATED 68.5%`).
     * `12m > 0`: **`82.9% [77.6–87.1%]`** (`n=239`, mean `+11.13%`, **`PF 13.41`**, AvgLoss `-5.23%`).
     * `3m beats SPY`: `43.2% [37.1–49.4%]` (mean `-0.26%`, `PF 0.87`) — confirms the sleeve's verified edge is massive drawdown compression (`-17.3%` vs SPY `-55.0%`), not outperforming SPY every quarter during straight bull surges.
2. **`C2: equity_pullback` Shares Satellite (2018–2026, `n=10,878` trades, 10 bps) — Per-Stock Logistic Model `FAILED` (`ship_check`); Regime Base Rates Used Instead:**
   * Expanding walk-forward logistic model on `[spy_above_200d, rsi2, ret_3d, rv20]` **FAILED** both `BSS` (`-0.052`, Brier `0.2482` vs base `0.2360`) and `pop_calibration.ship_check` (**slope `-0.300`, 95% CI `[-0.463, -0.136]`** — inverted out-of-sample!).
   * **Decision:** Never display fabricated per-stock ML win percentages. Display the honest **regime-stratified empirical base rate**:
     * **`SPY > SMA200` (bull):** **`61.7% [60.7–62.6%]`** (`n=9,473`, mean `+0.43%`, **`PF 1.33`**, AvgLoss `-3.41%`, worst 10% `-4.74%`).
     * **`SPY <= SMA200` (bear — PAUSED):** `59.5% [56.9–62.0%]` (`n=1,405`, mean `+0.07%`, `PF 1.04`, AvgLoss `-4.36%`, worst 10% `-6.53%`).
3. **`C3: Options DTE Curve` (`0DTE–240D`, OPRA `cbbo-1m`, `n=3,617` trades, `k=0.35` limit) — Per-Contract Logistic Model `FAILED` (`ship_check`); Per-DTE Empirical Rates Used Instead:**
   * Expanding walk-forward logistic model on `[abs_delta, dte, credit_to_width, friction_ratio]` (`n=2,817` OOS) **FAILED** `BSS` (`-0.0165`) and `ship_check` (**slope `+0.146`, 95% CI `[-7.168, +7.461]`**). Notably, when the model predicted `90–100%` (`n=700`, mean predicted `93.0%`), realized win rate fell to **`82.6%`**.
   * **Decision:** Never invent per-contract ML win % or display raw Black-Scholes `1 - |delta|`. Display each DTE bucket's exact realized OOS `P(Win) [95% Wilson CI, n]` **always paired with `PF`, `AvgLoss`, `Δ-Adj x_lim`, and `CLOSED` / `PROMISING` verdict** so `0DTE` (`75.7%` win, **`PF 0.59`**, AvgLoss `-34.3%`) and `1–7d` (`80.9%` win, **`PF 0.92`**, AvgLoss `-47.8%`) are unmistakably flagged as negative expectancy.
4. **`C4: Market-Implied Put Delta (`1 - |Δ|`) vs Realized OTM Finish (`144` Wednesdays × 6 ETFs, `n≈775`/bin) — Consistent `+5.5` to `+11.0 pp` VRP Bias:**
   * Realized `P(S_T > K)` exceeded risk-neutral `1 - |Δ|` in **every single bin** at both `30d` and `90d`, with the largest gap at `90d` (`20–30Δ`: `75.0%` implied vs **`85.8% [83.2–88.1%]`** realized = **`+10.81 pp`**; `30–40Δ`: `65.0%` vs `75.9%` = `+10.89 pp`). This empirically explains why `60–120d` short puts carry real variance risk premium while short-DTE gamma losses and crossed bid-ask friction destroy it elsewhere.

---

## Detailed Tables

### Track C1 — `VM_OUTLOOK_TOP3` Empirical & Walk-Forward Calibration (`2006–2026`)

| Horizon / Regime | `n` | `P(Win)` `[95% Wilson CI]` | Mean Ret | Avg Win | Avg Loss | Worst 10% | PF |
|---|---:|---|---:|---:|---:|---:|---:|
| `1m > 0` (All) | 250 | `63.6% [57.5–69.3%]` | +0.86% | +2.96% | -2.83% | -3.18% | 1.83 |
| `1m > 0` (`CALM`, RV20 `<= 14%`) | 118 | `63.6% [54.6–71.7%]` | +1.13% | +2.96% | -2.06% | -2.42% | **2.51** |
| `1m > 0` (`ELEVATED`, RV20 `> 14%`) | 132 | `63.6% [55.2–71.4%]` | +0.61% | +2.97% | -3.52% | -4.39% | 1.48 |
| **`3m > 0` (All)** | 248 | **`70.2% [64.2–75.5%]`** | **+2.68%** | +5.35% | -3.61% | -3.92% | **3.49** |
| `3m > 0` (`CALM`) | 118 | `72.0% [63.3–79.3%]` | +2.63% | +5.12% | -3.79% | -5.10% | 3.48 |
| `3m > 0` (`ELEVATED`) | 130 | `68.5% [60.0–75.8%]` | +2.73% | +5.57% | -3.46% | -3.45% | 3.50 |
| **`12m > 0` (All)** | 239 | **`82.9% [77.6–87.1%]`** | **+11.13%** | +14.52% | -5.23% | -3.28% | **13.41** |
| `3m > SPY` (Relative) | 248 | `43.2% [37.1–49.4%]` | -0.26% | +4.17% | -3.63% | -5.82% | 0.87 |

Walk-forward OOS checks (`min_train=60`m): `1m` (`n=190`) `ECE = 0.0179`, Brier `0.2351` (base `0.2313`, `BSS -0.0166`); `3m` (`n=188`) `ECE = 0.0204`, Brier `0.2108` (base `0.2047`, `BSS -0.0295`).

### Track C2 — `equity_pullback` Regime Base Rates (`2018–2026`, 10 bps Friction)

| Regime | `n` | `P(Win)` `[95% Wilson CI]` | Mean Ret | Avg Win | Avg Loss | Worst 10% | PF | Feature Logistic Gate |
|---|---:|---|---:|---:|---:|---:|---:|---|
| All pullbacks | 10,878 | `61.4% [60.5–62.3%]` | +0.38% | +2.85% | -3.54% | -4.98% | 1.28 | **FAIL** (`BSS -0.052`, slope `-0.300`) |
| **`SPY > SMA200` (Bull)** | 9,473 | **`61.7% [60.7–62.6%]`** | **+0.43%** | +2.82% | -3.41% | -4.74% | **1.33** | Use regime base rate |
| `SPY <= SMA200` (Bear — AVOID) | 1,405 | `59.5% [56.9–62.0%]` | +0.07% | +3.09% | **-4.36%** | **-6.53%** | 1.04 | **Paused in Discover** |

### Track C4 — OPRA Option-Implied Put Delta (`1 - |Δ|`) vs Realized `P(S_T > K)` (`2024–2026`, 6 ETFs)

| Horizon | Put Delta Bin | `n` | Implied OTM (`1 - |Δ|`) | Realized OTM `[95% CI]` | VRP Bias (`Real - Imp`) |
|---|---|---:|---:|---|---:|
| `30d` | `10–20Δ` | 777 | 85.0% | `90.5% [88.2–92.4%]` | `+5.45 pp` |
| `30d` | `20–30Δ` | 778 | 75.1% | `83.2% [80.4–85.6%]` | `+8.09 pp` |
| `30d` | `30–40Δ` | 775 | 65.1% | `74.6% [71.4–77.5%]` | `+9.48 pp` |
| `30d` | `40–50Δ` | 773 | 55.0% | `64.8% [61.4–68.1%]` | `+9.83 pp` |
| `90d` | `10–20Δ` | 778 | 85.1% | `93.2% [91.2–94.8%]` | `+8.11 pp` |
| **`90d`** | **`20–30Δ` (`O1`)** | **775** | **75.0%** | **`85.8% [83.2–88.1%]`** | **`+10.81 pp`** |
| `90d` | `30–40Δ` | 763 | 65.0% | `75.9% [72.7–78.8%]` | `+10.89 pp` |
| `90d` | `40–50Δ` | 772 | 55.1% | `66.1% [62.7–69.3%]` | `+11.02 pp` |

## Operational Contract Wired Into `[6] DISCOVER`

* `src/topdown/best_action.py` reads `configs/calibrated_evidence.json` exclusively — no hardcoded or Black-Scholes win percentages anywhere in `0 · BEST ACTION TODAY` or the `0DTE → 240D` DTE strip.
* Every probability badge prints `P(Win) [95% CI, n] · Avg · PF · AvgLoss` via `format_prob_badge()`.
* Failed ML calibrators (`C2` & `C3`) are explicitly rejected from the UI; Discover uses regime/DTE empirical base rates and flags negative-EV buckets (`0DTE`, `1–7d`, `8–30d`, `31–60d`, `120–240d`) as **`CLOSED`**.
