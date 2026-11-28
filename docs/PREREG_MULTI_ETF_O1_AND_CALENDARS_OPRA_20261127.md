# Preregistration — Study 3A: 2024–2026 OOS Multi-ETF `Test O1` & ATM `1M/3M` Calendar Spreads (Databento OPRA `cbbo-1m`)

**Registered:** 2026-11-27 (frozen BEFORE seeing Study H4 results and BEFORE downloading Study 3A weekly OPRA snapshots).  
**Target Account:** Taxable Margin / IBKR only (`[TAXABLE_ONLY]`; CRA prohibits multi-leg spreads in Canadian TFSAs).

## Why

1. **`Test O1` (`60–120 DTE`, `5%`-wide Bull Put, TP50/no-stop/21-DTE)** is currently the repo's only `LEAD` defined-risk credit spread rule, but was tested strictly on **SPY 2010–2023** (`optionsdx.db`). It has never been tested **out-of-sample on 2024–2026** nor on **non-SPY ETFs (`QQQ, IWM, XLE, GLD, TLT`)** using institutional OPRA NBBO quotes.
2. **ATM `1M/3M` Call Calendar Spreads** are now scanned live by `src/tui/options_engine.py`, but have never been backtested on institutional OPRA NBBO bid/ask quotes (`k=0.35` limit vs `k=1.0` crossed).

## Data Design (Zero Peeking, Budget-Capped)

* **Source:** Databento `OPRA.PILLAR`, schema `cbbo-1m`, `15:45 ET` 1-minute consolidated NBBO snapshots.
* **Universe (6 liquid ETFs, frozen before H4 read):** `SPY`, `QQQ`, `IWM`, `XLE`, `GLD`, `TLT`.
* **Dates:** Every full-session (`16:00 ET` close via exchange calendar `calendar.parquet`) Wednesday from `2024-01-03` through `2026-09-30` (~140 Wednesdays; 24 first-Wednesdays already cached by Study H4 -> ~116 incremental pulls).
* **Entry Window:** `2024-01-03` .. `2026-06-24` (strictly out-of-sample vs `2010–2023` `optionsdx.db`); exit tracking through `2026-09-30` so every `60–120 DTE` trade settles cleanly with zero right-censoring.
* **Hard Incremental Cost Ceiling:** `max_usd = $6.50` (expected `~$5.15`, keeping total cumulative Databento spend `<= $11.00` out of the `$125.00` credit).
* **Spot Prices:** Split-unadjusted official daily closes (`src/raw_closes.py::official_raw_closes`) matched to OPRA strikes.

## Frozen Claimants (Zero Parameter Tuning)

| Arm | Structure & Entry Rule (`15:45 ET` Wednesday OPRA NBBO) | Exit Rule (Checked Weekly on Wednesdays) | Fill Policies & Fees |
|---|---|---|---|
| **Arm 3A-1 (`O1_MULTI_ETF_OOS`)** | Exact shared `src/radar/wide_spread_rule.py::select_legs(..., side="bull_put", max_friction_to_credit=0.25)` (`60–120 DTE` nearest `90`, short nearest `-0.25Δ` ±0.08, long `~5%` OTM). | `TP50` (`close_debit <= 0.50 * entry_credit`), OR `DTE <= 21`, OR intrinsic at expiration (no stop loss; exact `O1-M`). | `limit (k=0.35)` primary, plus `crossed (k=1.0)` and `mid (k=0.0)`. Net of IBKR tiered fees (`$0.026/share` round-trip = `$0.65/leg × 4`). |
| **Arm 3A-2 (`ATM_CALENDAR_1M3M`)** | Long ATM Call Calendar: short call in `[25, 45]` DTE nearest `35` vs long call same strike in `[75, 125]` DTE nearest `90`, closest strike to spot (`|K/spot - 1| <= 0.03`), `debit_mid > 0`, `debit_cross > 0`, round-trip spread `<= 25%` of `debit_mid`. | `+25%` profit target, `-25%` stop loss, OR front-leg `DTE <= 7`. | `limit (k=0.35)` primary, plus `crossed (k=1.0)` and `mid (k=0.0)`. Net of `$0.026/share` IBKR fees. |

## Metrics, Inference & Adoption Rule (Frozen)

* **Metrics:**
  * For **3A-1 (`O1_MULTI_ETF_OOS`)**: Return on Capital at Risk (`RoC = pnl / (width - entry_credit)`), Profit Factor (`PF`), and **Delta-Adjusted Excess (`x_entry = (pnl - net_delta_entry * (spot_exit - spot_entry)) / car`)** — exact parity with `scripts/backtest_odx_wide_60_120.py`. Reported for `SPY` alone, each non-SPY ETF (`QQQ, IWM, XLE, GLD, TLT`), and pooled 6-ETF equal-weight.
  * For **3A-2 (`ATM_CALENDAR_1M3M`)**: Return on Debit (`RoC = pnl / entry_debit`), Profit Factor (`PF`), and win rate across all 6 ETFs (and per ETF).
* **Bootstrap:** Month-clustered moving-block bootstrap (`block = 3` months, `10,000` draws, seed `20261127`), **Bonferroni across 2 families (`alpha = 0.05 / 2`, `97.5%` two-sided CI)**.
* **Halves:** `2024` vs `2025–2026`.
* **Verdicts:**
  * **3A-1 (`O1_MULTI_ETF_OOS`, `k=0.35`):** `PASS_OOS` if pooled `x_entry` `97.5% CI lo > 0` AND `PF > 1.0` in both halves (allowing `TESTED_ON_SPY_ONLY` flag removal for ETFs that individually post positive `x_entry` and `<= 15%` median friction); `UNDERPOWERED_CONSISTENT` if `x_entry > 0` pointwise on SPY/pooled but 30-month CI spans zero (retains `LEAD` pending live fill experiment); `FAIL_OOS` if pooled `x_entry <= 0` (triggers warning/review of `etf_wide_spread`).
  * **3A-2 (`ATM_CALENDAR_1M3M`, `k=0.35`):** `PASS` (earns `LEAD [TAXABLE_ONLY]`) ONLY if pooled `RoC` `97.5% CI lo > 0`, `PF CI lo > 1.0`, and positive in both `2024` and `2025–2026` halves; otherwise **`CLOSED`**.
