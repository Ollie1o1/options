# Preregistration — Study 2B: Upgrading the VALIDATED `VM_OUTLOOK_TOP3` Sleeve

**Registered:** 2026-11-27 (frozen before `scripts/backtest_vm_outlook_v2.py` is executed).
**Baseline (`B0`):** Exact `VM_OUTLOOK_TOP3` rule validated in `docs/RESULT_VOL_MANAGED_OUTLOOK_20261124.md` (`src/outlook/vm_rotation.py`). Parity against `scripts/backtest_vol_managed_outlook.py::_simulate_arms` must be `< 1e-12` before reading any arm.

## Why

`docs/RESULT_SLEEVE_INTRADAY_VOL_20261127.md` proved that swapping the monthly volatility estimator (`RV20` → 5m `E1` / HAR `E2`) does not improve the sleeve because **volatility shifts within the month after weights are locked**, not because trailing `RV20` is noisy. Meanwhile, `B0` only selects from 16 mostly-equity US ETFs (`GLD` and `TLT` are the only non-equities) and weights the top 3 equally regardless of their individual volatilities.

## Frozen Claimants (3 arms; zero parameter tuning)

| Arm | Rule | Window | Hypothesis |
|---|---|---|---|
| **Arm A (`EXPANDED_23`)** | Identical `VM_OUTLOOK_TOP3` rule over `DEFAULT_UNIVERSE` (16) + 7 crisis/cross-asset diversifiers already in `data/equity_ohlcv.db` from 2006–2007: `DBC`, `UUP`, `IEF`, `EFA`, `EEM`, `HYG`, `SLV`. | 2006–2026 (`n ≈ 250` mo) | Adding commodities, USD, intermediate bonds, credit, and international ETFs improves risk-adjusted return and bear-market crisis alpha (e.g. 2022 when stocks & `TLT` fell together). |
| **Arm B (`INV_VOL_WEIGHTS`)** | On `DEFAULT_UNIVERSE` (16), weight the positive top-3 ETFs inversely to each ETF's own trailing 20d volatility (`w_i ∝ 1 / RV20_i`, clipped per-slot to `[0.15, 0.55]` of the active equity share `len(pos)/3`), then apply the exact same `clip(0.14 / RV20_basket, 0.25, 1.00)` scaling. | 2006–2026 (`n ≈ 250` mo) | Prevents high-vol ETFs (e.g. `SMH`) from dominating risk inside the 3-ETF basket. |
| **Arm C (`INTRA_MONTH_TRIGGER`)** | On `DEFAULT_UNIVERSE` (16) with Alpaca 5-minute bars (`data/intraday_5m/`). Enter monthly identically to `B0`. On any session `d` strictly inside `[exec_in + 1, exec_out - 2]`, compute the held basket's trailing 5-session intraday annualized RV (`rv5_intra`). If `rv5_intra >= 1.50 * rv20_entry` (once per month max), scale exposure down at the **next session's close (`d + 1`)** to `clip(0.14 / rv5_intra, 0.25, scale_entry)`, paying 3 bps on the turnover. | 2016–2026 (`n ≈ 129` mo) | Cuts intra-month volatility spikes (2018 Q1/Q4, Mar 2020, Apr 2025) without adding monthly churn in calm regimes. |

## Statistics & Friction (frozen)

* **Primary friction:** `3 bps` one-way turnover cost, `0%` cash yield (exact match to `B0`).
* **Stress check:** `10 bps` one-way turnover cost.
* **Diagnostic (no authority):** `Arm A+B` combination and `SGOV`/T-bill cash yield diagnostic (`DFF` / 3-month rate from FRED on uninvested cash) reported purely for operational realism.
* **Inference:** Circular moving-block bootstrap over monthly returns (`block = 3` months, `10,000` draws, seed `20261127`), **Bonferroni across 3 arms (`alpha = 0.05 / 3`, 98.33% two-sided CI)**.
* **Sub-period halves:** `2006–2015` vs `2016–2026` (Arms A/B); `2016–2020` vs `2021–2026` (Arm C).

## Adoption Rule (frozen)

An arm earns **`PASS`** (eligible for live promotion to `src/outlook/vm_rotation.py`) **only if**:
1. Either **Sharpe lift** (`ΔSharpe = Sharpe_arm - Sharpe_B0`) has `98.33% CI lower bound > 0` AND is positive in both halves, **OR** **Drawdown reduction** (`ΔMaxDD = MaxDD_arm - MaxDD_B0`, positive = shallower DD) has `98.33% CI lower bound > 0` AND pointwise `Sharpe_arm >= Sharpe_B0`; **AND**
2. It still beats `B0` pointwise on Sharpe and MaxDD under the `10 bps` friction stress.

Otherwise verdict = **`FAIL`** and `src/outlook/vm_rotation.py` remains unchanged.
