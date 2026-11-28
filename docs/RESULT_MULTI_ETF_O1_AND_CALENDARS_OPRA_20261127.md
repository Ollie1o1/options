# Result — Study 3A: 2024–2026 OOS Multi-ETF `Test O1` & ATM `1M/3M` Calendar Spreads (Databento OPRA `cbbo-1m`)

**Registered:** `docs/PREREG_MULTI_ETF_O1_AND_CALENDARS_OPRA_20261127.md` (frozen before Study H4 read and before Study 3A pull)  
**Estimator:** `scripts/backtest_opra_o1_and_calendars.py`  
**Raw output:** `reports/backtest_opra_o1_calendars_20261127T190735.json`  
**Data:** `144` full-session Wednesdays (`2024-01-03 .. 2026-09-30`, entries `2024-01-03 .. 2026-06-24`, `15:45 ET` `OPRA.PILLAR` `cbbo-1m`, `4,024,417` new rows downloaded for **`$6.0061`**; cumulative Databento spend across Study H4 + Study 3A = **`$11.5510` of `$125.00` credit**, leaving `$113.45` unused).  
**Date:** 2026-11-27

## Executive Summary

1. **Arm 3A-1 (`O1_MULTI_ETF_OOS`, `60–120 DTE` `5%`-wide Bull Put, `TP50 / 21-DTE`, `n = 740` trades, `effN = 41.0`): `UNDERPOWERED_CONSISTENT` (Retains `LEAD [TAXABLE_ONLY]`, limit-only `k <= 0.35`).**
   * Pooled at `k=0.35` limit fills (net `$2.60`/spread IBKR fees): **Raw RoC `+5.78%`**, Win Rate **`88.0%`**, **Profit Factor `2.306 [1.131, 6.995]`**, median friction **`4.87%`**, and **Delta-Adjusted Excess `x_entry = +0.18%` (`97.5% CI [-2.36%, +1.98%]`, halves `+0.99%` / `-0.41%`)**.
   * Pooled at `k=1.0` full crossed touch: Raw RoC `+5.07%` (`PF 2.056`), **Delta-Adjusted `x_entry = -0.50%` (`[-3.16%, +1.40%]`)** — reconfirming out-of-sample that crossing the bid-ask eliminates delta-adjusted edge (`O1` strictly requires measured limit fills `k <= 0.35`).
   * **Striking cross-ETF dispersion (descriptive; not Bonferroni-adjusted for 6-way selection):**
     * **`TLT`** (`n=130`, med friction `6.04%`): posted **significant positive delta-adjusted excess even at full crossed touch** — `k=0.35` limit `x_entry = +4.11% [+1.95%, +6.93%]` (`PF 1.885`, positive both halves `+2.35% / +5.27%`); `k=1.0` crossed `x_entry = +3.57% [+1.31%, +6.47%]` (`PF 1.775`). Sideways/choppy high-IV bond regimes rewarded short-put VRP handsomely.
     * **`QQQ`** (`+0.35%`, positive both halves, `PF 3.495 [1.07, 98.3]`, `3.39%` med friction) and **`XLE`** (`+2.40%`, positive both halves, `11.1%` med friction) stayed positive; **`SPY`** (`-0.24% [-3.78%, +2.00%]`, `PF 3.132`, `1.93%` med friction) and **`IWM`** (`-0.68%`) were flat vs delta across the 2024–26 bull market + April 2025 shock.
     * **`GLD` WARNING (`n=127`, `5.17%` med friction):** Despite having the deepest OPRA book (`252` lots in Study H4) and `+4.17%` raw RoC, **`GLD` Bull Puts significantly lagged linear gold delta (`x_entry = -4.33% [-10.93%, -0.01%]`, negative in both `2024: -1.14%` and `2025–2026: -6.54%`)**. Gold's parabolic uptrend + call-skew structure makes **owning `GLD` shares in TFSA strictly superior to selling `GLD` Bull Puts**.
2. **Arm 3A-2 (`ATM_CALENDAR_1M3M`, Long ATM Call Calendar `35d/90d`, `n = 721` trades, `effN = 34.0`): `CLOSED` (`FAIL`).**
   * At **`mid (k=0.0)`**: Raw RoC `+12.29% [+0.47%, +33.94%]`, `PF 1.976 [1.032, 3.711]` — appears attractive on paper!
   * At **`limit (k=0.35)`**: Raw RoC `+8.87%`, **`97.5% CI [-2.76%, +29.36%]`**, `PF 1.613 [0.834, 2.985]`, **negative in `2024` (`-0.92%`)** (`2025–2026` was carried by a single noisy `TLT` outlier `+82.2%`) → **`CLOSED`**.
   * At **`crossed (k=1.0)`**: **Loses `-5.21%` per trade (`[-11.48%, +0.71%]`, `PF 0.712`, negative both halves `-5.28% / -5.16%`)** — repeating the exact friction wall seen in earnings condors.

---

## Arm 3A-1: 2024–2026 OOS Multi-ETF `Test O1-M` Bull Put Results (Net `$2.60`/spread IBKR fees)

### Pooled 6-ETF Summary by Fill Policy (`n = 740`, `30` months)

| Fill Policy | `effN` | Raw RoC | Win Rate | Profit Factor `[97.5% CI]` | Delta-Adj `x_entry` | `97.5% CI` | Halves (`2024` / `25–26`) | Med Friction |
|---|---:|---:|---:|---|---:|---|---|---:|
| **`mid (k=0.0)`** | 43.0 | +6.01% | 88.1% | `2.382 [1.168, 7.214]` | `+0.54%` | `[-1.85%, +2.20%]` | `+1.25%` / `+0.02%` | 4.87% |
| **`limit (k=0.35)` (Primary)** | 41.0 | **+5.78%** | **88.0%** | **`2.306 [1.131, 6.995]`** | **`+0.18%`** | `[-2.36%, +1.98%]` | `+0.99%` / `-0.41%` | 4.87% |
| **`crossed (k=1.0)`** | 39.0 | +5.07% | 87.4% | `2.056 [1.042, 5.511]` | `-0.50%` | `[-3.16%, +1.40%]` | `+0.41%` / `-1.15%` | 4.87% |

### Per-ETF Breakdown (`limit k=0.35`, Primary Policy)

| ETF | `n` | `effN` | Med Friction | Raw RoC | Win Rate | PF `[97.5% CI]` | Delta-Adj `x_entry` | `97.5% CI` | Halves (`2024` / `25–26`) | Crossed `x_entry` |
|---|---:|---:|---:|---:|---:|---|---:|---|---|---:|
| **SPY** | 130 | 22.0 | 1.93% | +5.66% | 91.5% | `3.132 [0.88, 578]` | `-0.24%` | `[-3.78%, +2.00%]` | `+0.43%` / `-0.68%` | `-0.47%` |
| **QQQ** | 130 | 23.0 | 3.39% | +7.40% | 91.5% | `3.495 [1.07, 98.3]` | `+0.35%` | `[-3.71%, +3.02%]` | `+0.84%` / `+0.03%` | `-0.12%` |
| **IWM** | 130 | 23.0 | 4.81% | +6.47% | 90.0% | `2.341 [0.62, inf]` | `-0.68%` | `[-6.25%, +2.58%]` | `+0.24%` / `-1.30%` | `-1.32%` |
| **XLE** | 93 | 18.0 | 11.11% | +6.96% | 89.2% | `2.547 [1.06, 15.6]` | `+2.40%` | `[-1.50%, +5.25%]` | `+3.34%` / `+1.30%` | `+0.75%` |
| **GLD** | 127 | 29.0 | 5.17% | +4.17% | 85.8% | `1.613 [0.41, inf]` | **`-4.33%`** | **`[-10.93%, -0.01%]`** | `-1.14%` / `-6.54%` | **`-5.13%`** |
| **TLT** | 130 | 21.0 | 6.04% | +4.34% | 80.0% | `1.885 [1.07, 5.96]` | **`+4.11%`** | **`[+1.95%, +6.93%]`** | `+2.35%` / `+5.27%` | **`+3.57%`** |

---

## Arm 3A-2: ATM `1M/3M` Call Calendar Spreads (`35d/90d`, `+25% TP / -25% SL / 7 DTE`)

### Pooled 6-ETF Summary by Fill Policy (`n = 721–723`, `30` months)

| Fill Policy | `effN` | Mean RoC | `97.5% CI` | Win Rate | Profit Factor `[97.5% CI]` | Halves (`2024` / `25–26`) | Verdict |
|---|---:|---:|---|---:|---|---|---|
| **`mid (k=0.0)`** | 32.0 | +12.29% | `[+0.47%, +33.94%]` | 56.6% | `1.976 [1.032, 3.711]` | `+1.31%` / `+20.21%` | Paper-only |
| **`limit (k=0.35)` (Primary)** | 34.0 | +8.87% | `[-2.76%, +29.36%]` | 52.1% | `1.613 [0.834, 2.985]` | **`-0.92%`** / `+15.92%` | **CLOSED** |
| **`crossed (k=1.0)`** | 36.0 | **-5.21%** | `[-11.48%, +0.71%]` | 46.1% | `0.712 [0.483, 1.050]` | `-5.28%` / `-5.16%` | **CLOSED** |

### Per-ETF Breakdown (`limit k=0.35` vs `crossed k=1.0`)

| ETF | `n` | Med Friction | Limit `k=0.35` RoC `[97.5% CI]` | Limit PF | Crossed `k=1.0` RoC `[97.5% CI]` | Crossed PF |
|---|---:|---:|---|---:|---|---:|
| **SPY** | 128 | 1.40% | `+1.19% [-10.3%, +12.5%]` | 1.10 | `-1.21% [-12.5%, +10.2%]` | 0.91 |
| **QQQ** | 127 | 2.91% | `+5.31% [-4.2%, +12.7%]` | 1.51 | `+1.41% [-9.4%, +9.9%]` | 1.11 |
| **IWM** | 130 | 3.53% | `+8.42% [-3.6%, +16.1%]` | 1.78 | `+3.46% [-8.2%, +11.5%]` | 1.26 |
| **XLE** | 86 | 12.50% | `-5.00% [-14.8%, +9.3%]` | 0.73 | `-16.43% [-25.2%, -3.7%]` | 0.35 |
| **GLD** | 126 | 6.78% | `-3.07% [-14.5%, +9.7%]` | 0.84 | `-11.31% [-26.5%, +3.9%]` | 0.55 |
| **TLT** | 124 | 5.78% | `+42.65% [-11.0%, +177%]` | 3.49 | `-11.23% [-18.5%, -2.1%]` | 0.47 |

## Operational Consequence

1. **Keep `etf_wide_spread` (`Test O1`) at `LEAD [TAXABLE_ONLY]` (`k <= 0.35` limit-only, pending `20`-fill `fill_experiment`):** Do NOT promote to `VALIDATED` and keep `TESTED_ON_SPY_ONLY` intact on non-SPY ETFs.
2. **Flag `GLD` Bull Puts (`GLD_OOS_LAGGED_DELTA`) in `src/radar/live_options.py`:** `GLD` lagged linear delta by `-4.33%` per spread (`97.5% CI [-10.9%, -0.01%]`); hold `GLD` exposure via shares (or `VM_OUTLOOK_TOP3`), not short puts.
3. **Close naive ATM `1M/3M` Call Calendars:** Do NOT promote calendars to `LEAD`/`VALIDATED` in `configs/playbooks.json`; keep live calendar scans strictly descriptive (`CONTEXT`).
