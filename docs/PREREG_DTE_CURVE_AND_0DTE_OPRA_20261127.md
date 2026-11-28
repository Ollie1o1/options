# Preregistration — Study 4A: Complete Options DTE Curve (`0DTE` through `240 DTE`) on Institutional OPRA NBBO (`cbbo-1m`)

**Registered:** 2026-11-27 (frozen before running `scripts/backtest_dte_curve_opra.py` or pulling incremental `10:30 ET` `0DTE` quotes).  
**Target Account:** Taxable Margin / IBKR only (`[TAXABLE_ONLY]`; CRA prohibits multi-leg spreads in Canadian TFSAs).

## Why

To let the `[6] DISCOVER` desk sort live options setups from **`0DTE` out to `3M–8M`**, every DTE bucket needs an honest, institutional-quote backtest under identical friction assumptions (`limit k=0.35`, `crossed k=1.0`, net `$2.60`/spread IBKR fees) so the UI shows empirically backed verdicts (`PROMISING` vs `CLOSED / NEGATIVE EV`) instead of guessing.

## Data Design (Zero Peeking, Budget-Capped)

1. **Multi-DTE Term Structure (`1–7d`, `8–30d`, `31–60d`, `60–120d`, `120–240d`):**
   * **Cost: `$0`** — uses the existing `144` full-session Wednesdays (`2024-01-03 .. 2026-09-30`, entries through `2026-06-24` so every bucket except `120–240d` settles cleanly; `120–240d` entries capped at `2026-03-31`) across `SPY, QQQ, IWM, XLE, GLD, TLT` at `15:45 ET` in `data/databento_opra.db`.
2. **True Same-Day `0DTE` (`SPY`, `QQQ` — `10:30 ET` entry → `15:45 ET` exit / settlement):**
   * `24` first-Wednesdays (`2024-01..2025-12`) ALREADY have both `10:30 ET` and `15:45 ET` OPRA quotes cached locally (`$0`).
   * Pull incremental `10:30 ET` `cbbo-1m` quotes for **`SPY, QQQ` only** on `56` additional Wednesdays evenly spaced across `2024-01 .. 2026-06` (giving **`80` full-session `0DTE` dates × 2 ETFs = up to `160` institutional `0DTE` trades**; `15:45 ET` exit quotes are already cached for all `144` Wednesdays!).
   * **Hard Incremental Cost Ceiling:** `max_usd = $2.50` (expected `~$1.30–$1.80` against `$113.45` remaining Databento credit).

## Frozen DTE Buckets & Rules (Zero Parameter Tuning)

| Bucket | Expirations (`DTE`) | Entry Rule (Short `~0.25Δ` ±0.08, `max_fric = 0.25`) | Exit Rule |
|---|---|---|---|
| **`0DTE_BP`** (`SPY, QQQ`) | `0 DTE` (`exp == date`) | Enter `10:30 ET` (post-open H4 window): Bull Put short `~-0.20Δ` (±0.08), `$5`-wide wing (`~1%` of spot). | Close at `15:45 ET` OPRA quote (never carry pin risk into 16:00). |
| **`0DTE_IC`** (`SPY, QQQ`) | `0 DTE` (`exp == date`) | Enter `10:30 ET`: Iron Condor (`~-0.20Δ` put + `~+0.20Δ` call, `$5`-wide both wings), total RT spread `<= 25%` of credit. | Close all 4 legs at `15:45 ET` OPRA quote (`$5.20`/IC IBKR fee). |
| **`1_7D`** (6 ETFs) | `1–7 DTE` (nearest `2–5d`) | Enter `15:45 ET` Wed: Bull Put short `~-0.25Δ` (±0.10), `~2%`-wide wing. | Hold to expiration intrinsic settlement (Friday close; 1 crossing only, `$1.30`/spread open fee). |
| **`8_30D`** (6 ETFs) | `8–30 DTE` (nearest `21d`) | Enter `15:45 ET` Wed: Bull Put short `~-0.25Δ` (±0.08), `~3%`-wide wing. | Weekly check: `TP50`, OR `DTE <= 5`, OR expiry intrinsic. |
| **`31_60D`** (6 ETFs) | `31–60 DTE` (nearest `45d`) | Enter `15:45 ET` Wed: Bull Put short `~-0.25Δ` (±0.08), `~5%`-wide wing. | Weekly check: `TP50`, OR `DTE <= 14`, OR expiry intrinsic. |
| **`60_120D`** (6 ETFs — `O1`) | `60–120 DTE` (nearest `90d`) | Exact `Test O1-M`: short `~-0.25Δ` (±0.08), `~5%`-wide wing. | Weekly check: `TP50`, OR `DTE <= 21`, OR expiry intrinsic. |
| **`120_240D`** (6 ETFs) | `120–240 DTE` (nearest `180d`) | Enter `15:45 ET` Wed (`<= 2026-03-31`): Bull Put short `~-0.25Δ` (±0.08), `~5%`-wide wing. | Weekly check: `TP50`, OR `DTE <= 45`, OR expiry intrinsic. |

## Metrics, Inference & Adoption Rule (Frozen)

* **Reported per bucket (`limit k=0.35` primary & `crossed k=1.0`):** `n`, `effN`, **Win Rate + 95% Wilson CI**, **Mean RoC**, **Avg Win**, **Avg Loss (Tail Risk)**, **Worst 5% Trade**, **Profit Factor (`PF`)**, **Median Friction %**, and **Delta-Adjusted Excess (`x_entry`)** with month-clustered moving-block bootstrap (`block = 3` months, `10,000` draws, seed `20261127`, Bonferroni across 6 new buckets → `99.17%` two-sided CI; `60–120D` is reference).
* **PASS Bar for any new DTE bucket (to earn `LEAD [TAXABLE_ONLY]`):** At `k=0.35` limit fills, `99.17% CI` lower bound of `x_entry > 0`, `PF CI lo > 1.0`, positive in both `2024` and `2025–2026` halves, and `med_friction <= 15%`. Otherwise the bucket is stamped **`CLOSED / UNPROVEN`** in the Discover DTE strip (showing its exact backtested win rate, PF, tail loss, and honest verdict).
