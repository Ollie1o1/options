# Result — Study 2B: Upgrading the VALIDATED `VM_OUTLOOK_TOP3` Sleeve

**Registered:** `docs/PREREG_VM_OUTLOOK_V2_20261127.md` (frozen before run)  
**Estimator:** `scripts/backtest_vm_outlook_v2.py`  
**Raw output:** `reports/backtest_vm_outlook_v2_20261127T164237.json`  
**Date:** 2026-11-27

## Verdict: ALL THREE ARMS **FAIL** — Live `src/outlook/vm_rotation.py` Unchanged

* **Parity check vs validated `VM_OUTLOOK_TOP3` (`scripts/backtest_vol_managed_outlook.py::_simulate_arms`):** `max |diff| = 0.00e+00` (`< 1e-12` PASS).
* Under the frozen adoption rule (circular 3-month block bootstrap, 10,000 draws, seed `20261127`, Bonferroni 98.33% CI across 3 arms + both-halves sign check + 10 bps stress), **no arm clears the bar for promotion**.
* **Arm A (`EXPANDED_23`)** actively degraded Sharpe (`-0.166`) and deepened MaxDD by `-9.35 pp` (`-26.68%` vs `-17.33%`).
* **Arm B (`INV_VOL_WEIGHTS`)** slightly degraded Sharpe (`-0.049`) while raising annual turnover (`8.21x` vs `7.23x`).
* **Arm C (`INTRA_MONTH_TRIGGER`)** showed the right economic direction across both halves and 10 bps stress (Sharpe `0.989` vs `0.903`, MaxDD `-12.22%` vs `-16.58%`, Calmar `0.902` vs `0.696`, 45 triggers in 129 months), **but both 98.33% CIs comfortably span zero** (`ΔSharpe [-0.176, +0.315]`, `ΔMaxDD [-3.12pp, +16.26pp]`). Per the frozen rule, Arm C is **`FAIL`** and is **not** promoted.

---

## Primary Summary Table (3 bps one-way turnover, 0% cash yield)

| Arm | Window (`n` mo) | CAGR | Ann Vol | Sharpe | Sortino | MaxDD | Calmar | Turnover | `ΔSharpe` [98.33% CI] | Halves `ΔSharpe` | `ΔMaxDD` [98.33% CI] | 10 bps Stress | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---|
| **Baseline `B0`** (`DEFAULT_UNIVERSE` 16) | 2006–2026 (`250`) | +9.90% | 12.54% | **0.819** | 0.736 | **-17.33%** | 0.571 | 7.23x | — | — | — | Sharpe `0.779`, DD `-17.70%` | **BASELINE** |
| **Arm A (`EXPANDED_23`)** | 2006–2026 (`250`) | +8.34% | 13.78% | 0.653 | 0.627 | -26.68% | 0.313 | 7.61x | `-0.166` `[-0.460, +0.097]` | `-0.122` / `-0.203` | `-9.35pp` `[-22.90, +6.89]` | FAIL (`0.614`, `-27.13%`) | **FAIL** |
| **Arm B (`INV_VOL_WEIGHTS`)** | 2006–2026 (`250`) | +9.39% | 12.78% | 0.770 | 0.694 | -17.31% | 0.543 | 8.21x | `-0.049` `[-0.149, +0.051]` | `+0.028` / `-0.114` | `+0.02pp` `[-5.96, +4.85]` | FAIL (`0.725`, `-17.33%`) | **FAIL** |
| **Baseline `B0` (Intraday Window)** | 2016–2026 (`129`) | +11.55% | 13.13% | 0.903 | 0.764 | -16.58% | 0.696 | — | — | — | — | Sharpe `0.866`, DD `-16.60%` | **BASELINE** |
| **Arm C (`INTRA_MONTH_TRIGGER`)** | 2016–2026 (`129`) | +11.03% | 11.26% | 0.989 | 0.909 | -12.22% | 0.902 | 7.54x | `+0.086` `[-0.176, +0.315]` | `+0.127` / `+0.054` | `+4.36pp` `[-3.12, +16.26]` | PASS (`0.941`, `-12.24%`) | **FAIL** |

### Diagnostics (No Adoption Authority)

* **Arm A+B (`EXPANDED_23` + `INV_VOL_WEIGHTS`, 2006–2026):** CAGR `+9.04%`, Ann Vol `13.51%`, Sharpe `0.711`, Sortino `0.697`, MaxDD `-24.36%`, Calmar `0.371`. Confirms expanding the universe degrades the sleeve even when inverse-vol weighted.
* **Key Crisis & Dispersion Years for Arm A (`EXPANDED_23`) vs `B0`:**
  * `2008`: `-7.62%` vs `-6.21%`
  * `2018`: `-8.93%` vs `-5.34%`
  * `2020`: `+15.08%` vs `+16.73%`
  * `2022`: `-1.54%` vs `+2.33%` (contrary to hypothesis, adding `DBC`/`UUP`/`IEF`/`EFA`/`EEM`/`HYG`/`SLV` hurt in 2022 by rotating into commodities/EM whipsaws instead of staying tightly concentrated in `XLE`/`GLD`/cash)
  * `2025`: `+43.53%` vs `+19.32%` (driven by the 2025 silver `SLV` spike, insufficient to offset 20 years of higher whipsaw and drawdown).

## Interpretation & Operational Action

1. **Keep `DEFAULT_UNIVERSE` (16 ETFs) and equal-weight top-3 unchanged in `src/outlook/vm_rotation.py`.** The validated 16-ETF universe is already well-balanced (`XLE` + `GLD` + `TLT` + defensive sectors `XLP`/`XLV`/`XLU` + cash threshold); adding higher-noise cross-asset ETFs (`SLV`, `DBC`, `EEM`, `HYG`, `UUP`, `EFA`, `IEF`) dilutes selection quality and deepens drawdowns.
2. **Do not promote Arm C (`INTRA_MONTH_TRIGGER`) live.** While cutting exposure next-day when 5-day intraday basket RV reaches `1.50x` entry RV20 trimmed `4.36 pp` off MaxDD (`-16.58% -> -12.22%`) and lifted Calmar (`0.696 -> 0.902`) at negligible extra turnover (`7.54x`), 10 years (129 months, 45 triggers) does not give statistical power to separate `+0.086` Sharpe / `+4.36 pp` DD from sampling noise at 98.33% confidence. Record cleanly as **CLOSED (not promoted)** unless a longer intraday history or independent forward test later justifies reopening it.
