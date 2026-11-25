# Equity Playbook 2: Volatility-Managed Sector/Asset ETF Rotation — REGISTRATION

**Frozen 2026-11-24, before any portfolio equity curve or vol-scaling backtest has been run on `src/outlook`.**
Part of the stock-first daily profit reset (Phase 1 of the execution plan).
Adheres strictly to `docs/RESEARCH_PROCESS.md` steps 1–9.

---

## Provenance

- **What has already been seen:**
  - `docs/OUTLOOK_FINDINGS.md` (2026-06-08): Over ~8 years aligned history across 16 ETFs (`XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLRE, XLU, XLB, SMH, QQQ, IWM, GLD, TLT`), `src/outlook/engine.py` (default weights `mom_12_1=0.35`, `trend_score=0.30`, `relative_strength=0.20`, `reversal_1m=0.15`) achieved rank IC `+0.053` (42d) and `+0.076` (63d), and 66.5%–71.6% bullish hit rate.
  - Because `src/outlook` weights were previously observed over this ~8-year window, **the ETF ranking model itself is in-sample on pre-2026-06 data**; we state this explicitly rather than pretending it is untouched OOS.
- **What has NOT been seen:**
  - Zero portfolio-level simulations (CAGR, Sharpe, Max Drawdown, turnover, transaction costs) turning `src/outlook` into an investable TFSA-legal ETF allocation.
  - Zero tests of **Moreira & Muir (2017) inverse-variance volatility scaling** or **VIX / VIX3M term-structure de-risking** on this universe or on `SPY`.

---

## Economic Mechanism

Two independent, low-turnover structural effects suitable for a Canadian TFSA (long-only, no leverage, no shorting, ~2–3 bps ETF spread, $0 commission):
1. **Cross-Sectional Leadership Rotation (`src/outlook`):** Holding the top-ranked sector/asset ETFs captures persistent 1–3 month institutional flow and macro-regime leadership (including defensive rotation into `GLD` / `TLT` / `XLP` / `XLU` during equity drawdowns).
2. **Volatility Management (Moreira & Muir, *Journal of Finance* 2017):** Realized volatility clusters far more predictably than returns do; scaling risky weight inversely with trailing realized variance (holding uninvested cash in T-bills / short-term treasury yield) cuts left-tail drawdowns during crises without sacrificing long-run risk-adjusted return.

---

## Population & Data Sources

- **Universe:** `src/outlook/backtest.py::DEFAULT_UNIVERSE` (16 liquid US ETFs: `XLK, XLF, XLE, XLV, XLI, XLC, XLY, XLP, XLRE, XLU, XLB, SMH, QQQ, IWM, GLD, TLT`) + Benchmark `SPY`.
- **Price Data:** `data/equity_ohlcv.db` (adjusted daily closes after Phase 0.3 expansion, covering dividends via `Adj Close`!) from **earliest common aligned date** (2018-06-19 when `XLC` launched; plus an explicit **extended pre-XLC sensitivity** starting 2006 using all available ETFs on each date with `>= 252` trailing bars so 2008 GFC and 2011/2015/2018 are tested!).
- **Cash / Risk-Free Rate:** Daily 3-month Treasury yield (`^IRX` or `SHV`/`BIL` proxy; conservatively `0.0%` cash yield in primary arm if `^IRX` history is absent, with actual T-bill yield reported as a sensitivity so cash return never inflates Sharpe artificially!).
- **VIX Series (for Arm C only):** Local `data/discovery/VIX_History.csv` (`1990-01-02` to `2026-11-23`) + `^VIX3M` (from `equity_ohlcv.db` if added in Phase 0, else VIX vs 60d VIX MA proxy stated explicitly before run).

---

## Strategy Variants (Frozen — Exactly 3 Arms + 2 Benchmarks, No Sweeps)

Rebalance cadence: **Monthly (every 21 trading days)**, weights computed from closes up to `t`, **executed at close of `t + 1`** (1-session execution lag, zero look-ahead).

- **Benchmark B0 (`SPY_BH`):** 100% `SPY` buy-and-hold.
- **Benchmark B1 (`EW_16`):** Equal-weight (`1/N`) across all 16 ETFs, rebalanced every 21 trading days (isolates rotation skill from asset-class diversification into `GLD`/`TLT`).

### Claimants (Bonferroni over 3 arms: 98.33% two-sided CI / `alpha = 0.0167`)
1. **Arm A (`OUTLOOK_TOP3`):** Equal-weight (`1/3` each) the top **3** ETFs ranked by `src/outlook/engine.py::rank_universe` (default frozen config, **no weight changes**). If fewer than 3 ETFs have `score > 0`, hold remaining slots in cash. Max weight `100%` (TFSA long-only).
2. **Arm B (`VM_OUTLOOK_TOP3` — Primary Claimant):** Compute the exact Arm A target basket, then compute the trailing 20-trading-day realized annualized volatility `sigma_20` of that target basket (using daily log returns over `[t-20..t]`). Scale total invested exposure by:
   ```
   scale = clip(TARGET_VOL / sigma_20, 0.25, 1.00)
   ```
   with **`TARGET_VOL = 0.14`** (14% annualized — long-run median US equity vol, fixed a priori, never tuned!). Uninvested `1 - scale` sits in cash.
3. **Arm C (`VM_SPY` — Pure Vol-Managed Benchmark Control):** Same `scale = clip(0.14 / sigma_20_SPY, 0.25, 1.00)` applied directly to `SPY` alone. Separates **"does volatility management work?"** from **"does Outlook rotation add value on top of volatility management?"**.

### Transaction Costs
- **`3 bps` (`0.0003`) per dollar of one-way turnover** (`sum(|w_new - w_old|) * 0.0003`) deducted at every rebalance date (ETF spreads on `SPY/XL*/QQQ/GLD/TLT` are 1–2 bps, $0 commission).

---

## Observation Unit, Clustering & Inference

- **Observation unit:** **Non-overlapping 21-trading-day rebalance periods** (monthly returns).
- **Minimum Sample Floor:** `MIN_PERIODS = 60` monthly periods (~5 years; aligned 2018–2026 gives ~90 periods; extended 2007–2026 gives ~230 periods).
- **Bootstrap:** Stationary/circular block bootstrap (block length = `3` months to preserve volatility clustering) with `10,000` draws, fixed seed `20261124`.
- **Metrics reported for every arm & benchmark:**
  - Net CAGR, Annualized Volatility, **Sharpe Ratio** (excess over risk-free), **Sortino Ratio**, **Max Drawdown**, **Calmar Ratio**, Average Invested Exposure %, Annualized Turnover %, Total Cost Drag (bps/yr).
  - Paired differences vs `SPY_BH` and vs `EW_16`: `ΔSharpe` and `ΔMaxDD` with 98.33% (Bonferroni-3) and 95% CIs.

---

## Binding Pass Rules

### Primary Test (Arm B `VM_OUTLOOK_TOP3` vs `SPY_BH` & `EW_16`)
- **PASS (Risk-Adjusted Upgrade):**
  1. `Sharpe(Arm B) - Sharpe(SPY_BH)` 98.33% CI lower `> 0`, **OR**
  2. `Sharpe(Arm B) >= Sharpe(SPY_BH) - 0.05` **AND** `MaxDD(Arm B)` improves over `MaxDD(SPY_BH)` by at least **25% relative** (e.g., `-34% → <= -25.5%`) with positive net CAGR `> T-bill + 3%`, **AND**
  3. Replicates with the same sign in the **pre-2020** and **2020–2026** sub-periods (using the extended dynamic-universe series starting 2007 so 2008 GFC + 2020 + 2022 are all observed!).
- **Decomposition Check (No Override, Diagnostic Authority):** Compare `Arm B` vs `Arm C (VM_SPY)` and `Arm A` vs `EW_16` to state honestly whether any gain comes from **rotation**, **volatility scaling**, or **both**.

### Sensitivities (No Authority)
1. Cash earns actual 3-month T-bill yield (`^IRX`) vs `0%`.
2. Adding a `VIX > VIX3M` backwardation cap (`max scale = 0.50` when VIX term structure is inverted).
3. Top-4 and Top-5 ETFs vs Top-3.
4. `10 bps` turnover cost stress test.
5. Sub-period table: `2007–2011` (GFC/Euro), `2012–2019`, `2020–2021` (COVID), `2022` (rate shock), `2023–2026`.

---

## Authority & Prior

- **Authority:** If Arm B (or Arm C) passes, it becomes the **Core TFSA Allocation Sleeve** surfaced on `opt today` / `TODAY` tab (showing current target ETF weights, current `sigma_20` scale %, and next rebalance date). No auto-execution; human places monthly orders.
- **Prior (stated before running):** High prior that volatility scaling (Arm C / Arm B) meaningfully cuts Max Drawdown (especially in 2008, 2020, 2022) and lifts Sharpe by `+0.15` to `+0.30`; uncertain whether `OUTLOOK_TOP3` clears `98.33%` Sharpe significance over `EW_16` on only ~90 months, which is why the extended 2007–2026 dynamic-universe test is essential for statistical power.

**Estimator:** `scripts/backtest_vol_managed_outlook.py` (backed by `src/equity_playbooks/backtest.py`).
**Result file:** `docs/RESULT_VOL_MANAGED_OUTLOOK_20261124.md`.
