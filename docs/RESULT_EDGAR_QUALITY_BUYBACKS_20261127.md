# Result — Study 2D: Point-in-Time SEC EDGAR Quality + Net Buybacks TFSA Sleeve

**Registered:** `docs/PREREG_EDGAR_QUALITY_BUYBACKS_20261127.md` (frozen before XBRL pull or run)  
**Estimator:** `scripts/backtest_edgar_quality_buybacks.py`  
**Raw output:** `reports/backtest_edgar_quality_buybacks_20261127T165927.json`  
**Date:** 2026-11-27

## Verdict: ALL THREE ARMS **FAIL** — Keep `edgar_quality` as Descriptive `WATCHLIST` Only

* **Data coverage:** `286 / 286` non-ETF US equities (`2016-06-30` .. `2026-06-30`, `n = 41` quarters, median `268` eligible stocks/quarter, zero SEC XBRL download errors), strictly point-in-time (`filed_date <= signal_date`) using split-safe same-filing (`accn`) comparative diluted share counts.
* **Survivorship control proved essential:** Over `2016–2026`, `SPY` compounded at `+15.25%` (`Sharpe 0.918`), whereas the equal-weight 286-stock pool (`EW_UNIV`) compounded at **`+17.47%` (`Sharpe 0.974`)** — `+2.22%/yr` of pure survivorship inflation. Benchmarking against `EW_UNIV` of the exact same eligible stocks prevents mistaking survivorship bias for factor alpha.
* Under the frozen adoption rule (circular 2-quarter block bootstrap, 10,000 draws, seed `20261127`, Bonferroni `98.33%` CI across 3 arms + both-halves sign check + `20 bps` stress), **none of the three concentrated top-20 sleeves beat `EW_UNIV` reliably**:
  * **Arm Q1 (`TOP20_NBY`):** Excess vs `EW_UNIV` `+6.67%/yr`, but wide `98.33% CI [-4.43%, +25.58%]`, high volatility (`29.88%`), deeper MaxDD (`-32.54%` vs `-27.04%`), and *lower* Sharpe than `EW_UNIV` (`0.831` vs `0.974`) → **`FAIL`**.
  * **Arm Q2 (`TOP20_GPA`):** Excess vs `EW_GPA_UNIV` `+0.64%/yr`, `98.33% CI [-8.75%, +13.91%]`, flipped sharply negative in `2021H2–2026` (`-7.79%/yr`), lower Sharpe (`0.795` vs `1.075`) → **`FAIL`**.
  * **Arm Q3 (`TOP20_COMPOSITE`):** Excess vs `EW_UNIV` `+1.01%/yr`, `98.33% CI [-3.35%, +5.13%]`, negative in `2021H2–2026` (`-0.81%/yr`), Sharpe `0.970` vs `EW_UNIV` `0.974` → **`FAIL`**.

---

## Primary Summary Table (`2016-06-30` .. `2026-06-30`, `n = 41` quarters, 10 bps one-way friction)

| Portfolio | CAGR | Ann Vol | Sharpe | MaxDD | Calmar | Ann Turnover | Excess vs EW (`10 bps`) | 98.33% CI | Halves (`16–21H1` / `21H2–26`) | 20 bps Excess | Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| **`EW_UNIV` (Benchmark)** | +17.47% | 18.63% | 0.974 | -27.04% | 0.646 | 0.54x | — | — | — | — | **BENCHMARK** |
| **`EW_GPA_UNIV` (GP/A Pool)** | +20.59% | 19.54% | 1.075 | -24.19% | 0.851 | — | — | — | — | — | **BENCHMARK** |
| **`SPY` (Reference Only)** | +15.25% | 17.39% | 0.918 | -23.79% | 0.641 | — | — | — | — | — | Reference |
| **Arm Q1 (`TOP20_NBY`)** | +22.84% | 29.88% | 0.831 | -32.54% | 0.702 | 2.71x | `+6.67%` | `[-4.43%, +25.58%]` | `+12.47%` / `+0.59%` | `+6.44%` | **FAIL** |
| **Arm Q2 (`TOP20_GPA`)** | +19.80% | 27.23% | 0.795 | -26.48% | 0.748 | 1.10x | `+0.64%` | `[-8.75%, +13.91%]` | `+8.66%` / `-7.79%` | `+0.58%` | **FAIL** |
| **Arm Q3 (`TOP20_COMPOSITE`)** | +18.36% | 19.74% | 0.970 | -27.61% | 0.665 | 2.10x | `+1.01%` | `[-3.35%, +5.13%]` | `+2.74%` / `-0.81%` | `+0.85%` | **FAIL** |

---

## Diagnostics (No Adoption Authority)

* **Broad Quintile Spread (`Q5` top 20% vs `Q1` bottom 20% on Composite `Q_NBY`):** **`+3.13%/yr`** (`Q5` CAGR `+19.20%`, Sharpe `1.022`, MaxDD `-26.26%` vs `Q1` CAGR `+15.44%`, Sharpe `0.821`, MaxDD `-28.07%`). Most of the separation comes from **avoiding heavy diluters / low-profitability bottom-quintile names** (`Q1` trailed `EW_UNIV` by `-2.02%/yr`), rather than concentrating into the extreme top 20.
* **Momentum-Neutral Double Sort (Top 20% vs Bottom 20% `Q_NBY` within each 12-1m momentum quintile):** **`+3.79%/yr`** (`95% CI [-1.33%, +8.96%]` — directional, spans zero).
* **`Q3_TOP20_COMPOSITE` + `Close > SMA200` Trend Filter:** CAGR `+10.45%`, Ann Vol `13.00%`, Sharpe `0.839`, MaxDD `-20.45%`, turnover `2.16x` — degraded Sharpe vs unfiltered `0.970`.
* **Diversification Audit:** `Q3_TOP20_COMPOSITE` has **`0.929` quarterly correlation with `SPY`** — it behaves primarily as equity beta, not an uncorrelated sleeve alongside `VM_OUTLOOK_TOP3`.

## Operational Action

1. **Do NOT promote `edgar_quality` to `LEAD` or `VALIDATED` in `configs/playbooks.json`.** Keep it cleanly labeled **`WATCHLIST`** in `opt discover --lens edgar_quality` and `opt alpha fundamentals`.
2. **Keep the 286-issuer live SEC EDGAR XBRL cache (`data/sec_fundamentals.db`)** populated with split-safe diluted Net Buyback Yield, GP/A, and FCF/A — it is genuinely useful as a **fundamental anti-dilution / quality filter** when manually vetting single stocks, without claiming unproven excess alpha over a diversified index.
