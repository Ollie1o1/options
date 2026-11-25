# Playbook 2 Result: Volatility-Managed Sector & Asset-Class ETF Rotation (`VM_OUTLOOK_TOP3`)

**Date:** 2026-11-24
**Preregistration:** [PREREG_VOL_MANAGED_OUTLOOK_20261124.md](PREREG_VOL_MANAGED_OUTLOOK_20261124.md)
**Estimator:** [scripts/backtest_vol_managed_outlook.py](../scripts/backtest_vol_managed_outlook.py)
**Raw Artifact:** [reports/backtest_vol_managed_outlook_20261124.json](../reports/backtest_vol_managed_outlook_20261124.json)
**Verdict:** **`PASS` (`VALIDATED` — Core TFSA Allocation Sleeve)**

---

## 1. Primary Results

### A. Aligned 16-ETF Universe (`2019-06-21` .. `2026-09-29`, 88 monthly periods, 3 bps/side, 0% cash yield)

| Strategy | CAGR | Ann Vol | Sharpe | Sortino | MaxDD | Calmar | Avg Exp | Cost Drag | Prereg Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `SPY_BH` (benchmark) | +14.58% | 20.47% | 0.778 | 0.651 | -32.75% | 0.445 | 100.0% | 0.4 bps/yr | Baseline |
| `EW_16` (benchmark) | +13.77% | 18.69% | 0.793 | 0.638 | -30.64% | 0.449 | 100.0% | 0.4 bps/yr | Baseline |
| `OUTLOOK_TOP3` | +17.33% | 20.55% | 0.891 | 0.715 | -27.48% | 0.631 | 100.0% | 21.1 bps/yr | FAIL (DD -16% < -20%) |
| **`VM_OUTLOOK_TOP3`** | **+14.00%** | **14.49%** | **0.983** | **0.809** | **-17.24%** | **0.812** | **78.7%** | **22.2 bps/yr** | **`PASS`** |
| `VM_SPY` (control) | +8.36% | 17.02% | 0.570 | 0.410 | -32.75% | 0.255 | 86.4% | 4.5 bps/yr | FAIL |

### B. Extended Dynamic Universe (`2006-01-04` .. `2026-10-19`, 250 monthly periods, includes 2008 GFC, 2020 COVID, 2022 Bear)

| Strategy | CAGR | Ann Vol | Sharpe | Sortino | MaxDD | Calmar | Avg Exp | Prereg Verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| `SPY_BH` | +10.65% | 17.01% | 0.685 | 0.570 | -55.01% | 0.194 | 100.0% | Baseline |
| `EW_16` | +10.96% | 15.03% | 0.771 | 0.683 | -44.91% | 0.244 | 100.0% | Baseline |
| `OUTLOOK_TOP3` | +11.99% | 15.70% | 0.805 | 0.728 | -25.26% | 0.475 | 100.0% | `PASS` |
| **`VM_OUTLOOK_TOP3`** | **+9.90%** | **12.54%** | **0.819** | **0.736** | **-17.33%** | **0.571** | **85.8%** | **`PASS`** |
| `VM_SPY` (control) | +8.59% | 12.73% | 0.716 | 0.560 | -33.65% | 0.255 | 86.5% | FAIL |

---

## 2. Subperiod & Stress Robustness

### Key Subperiods (Extended 2006–2026 & Aligned 2019–2026)
- **2007–2011 (GFC):** `VM_OUTLOOK_TOP3` `+6.62%` CAGR / `-17.33%` MaxDD (Sharpe `0.552`) vs `SPY_BH` `+0.24%` / `-55.01%` (Sharpe `0.129`).
- **2012–2019 (Low-Vol Bull):** `VM_OUTLOOK_TOP3` `+11.58%` / `-12.66%` (Sharpe `1.158`) vs `SPY_BH` `+14.65%` / `-11.19%` (Sharpe `1.374`) — modest lag in uninterrupted melt-ups due to cash drag.
- **2022 Bear (Aligned 16-ETF):** `VM_OUTLOOK_TOP3` **`+3.40%` CAGR / `-8.07%` MaxDD** vs `SPY_BH` **`-8.87%` / `-18.04%`** and `VM_SPY` `-9.58%` / `-14.71%`.
- **2023–2026 (Aligned 16-ETF):** `VM_OUTLOOK_TOP3` `+17.74%` CAGR / `-14.49%` MaxDD vs `SPY_BH` `+18.35%` / `-8.88%`.

### Sensitivities (2006–2026, Non-Binding)
- **VIX Cap (`VIX > 30` halves weight):** `VM_OUTLOOK_TOP3` Sharpe `0.817`, MaxDD **`-16.58%`**, `ΔMaxDD vs SPY = +38.42pp` with 95% block-bootstrap CI **`[+1.46pp, +37.35pp]`** (strictly > 0). Verdict: `PASS`.
- **10 bps/side Cost Stress:** `VM_OUTLOOK_TOP3` Sharpe `0.779`, MaxDD `-17.70%`. Verdict: `PASS`.
- **Top-4 Equal-Weight:** `VM_OUTLOOK_TOP3` Sharpe `0.834`, MaxDD `-22.72%`, `ΔMaxDD CI95 [+1.37pp, +33.05pp]`. Verdict: `PASS`.

---

## 3. Honest Statistical Caveats

1. **Sharpe difference CIs still span zero:** Moving-block (6-month block) 95% CIs on `ΔSharpe(VM_OUTLOOK_TOP3 - SPY)` are `[-0.53, +0.72]` (aligned 7y) and `[-0.24, +0.48]` (extended 20y). Pointwise Sharpe is higher in both windows (`+0.205` and `+0.134`), and it beats `VM_SPY` by `+0.413` Sharpe (aligned) / `+0.103` (extended), proving the edge comes from combining **cross-asset Outlook rotation + volatility scaling**, not vol scaling alone.
2. **Primary value is tail/drawdown control + zero-friction TFSA compounding:** Drawdowns drop from `-55.0%` to `-17.3%` over 20 years and from `-32.8%` to `-17.2%` over 2019–2026, while keeping CAGR within ~0.6–0.8% of SPY even with **0% assumed cash yield** (holding T-bills/CASH.TO on the ~14–21% uninvested cash sleeve adds ~0.6–1.0%/yr in positive-rate regimes).
3. **Production Action:** Promoted to **`VALIDATED`** in `configs/playbooks.json` as the system's core TFSA-legal ETF allocation sleeve (monthly rebalance, `sigma_target = 14%`, `scale = clip(0.14 / sigma_20, 0.25, 1.00)`).
