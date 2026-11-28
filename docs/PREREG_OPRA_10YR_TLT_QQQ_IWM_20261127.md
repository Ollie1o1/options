# Preregistration — Study 5A: 11-Year Institutional OPRA Test (`2015–2026`) of `TLT`, `QQQ`, and `IWM` `60–120 DTE` Spreads (Bull Put & Bear Call)

**Registered:** 2026-11-27 (frozen before any incremental `2015–2023` Databento OPRA download or run).

## Motivation

In Study 3A (`2024–2026` OOS OPRA `cbbo-1m`), `TLT` `60–120 DTE` Bull Puts outperformed delta even at full crossed bid-ask touch (`+4.11% [+1.95%, +6.93%]` at `k=0.35` limit; `+3.57%` crossed), while `SPY` was previously validated on 2010–2023 OptionsDX (`+1.59%` at `k=0.35` limit). Because `OPRA.PILLAR cbbo-1m` extends back to `2013-04-01`, testing `TLT`, `QQQ`, and `IWM` across a full **11.7-year multi-regime cycle (`2015-01 .. 2026-09`)** — including the 2018 vol shock, 2020 COVID crash, and the historic 2022–2023 `-45%` bond bear market — definitively determines whether `TLT`, `QQQ`, or `IWM` earn `VALIDATED` status alongside `SPY` (or whether 2024–2026 was regime luck).

## Frozen Data & Cost Guard

* **Dataset:** Databento `OPRA.PILLAR`, schema `cbbo-1m`, snapshot clock `15:45 ET` (1-minute window; validated peak liquidity clock from Study H4).
* **Symbols:** `TLT`, `QQQ`, `IWM`.
* **Calendar:** Every full-session (`16:00 ET` close) Wednesday from `2015-01-07` through `2026-09-30` (rolling Thu/Fri/Tue on holidays, matching `scripts/backtest_opra_o1_and_calendars.py`). `2024-01 .. 2026-09` (`144` dates) are already cached in `data/databento_opra.db`; `~468` incremental `2015–2023` Wednesdays will be cost-estimated via `/v0/metadata.get_cost` first and strictly capped at **`$9.00`** (`--max-usd 9.0`).
* **Underlying Spots:** Official split-unadjusted raw closes via `src/raw_closes.official_raw_closes()` (`2015-01 .. 2026-10`), zero forward-parity drift.

## Frozen Strategy Specification (Exact `O1-M` Rule, Zero Parameter Tuning)

* **Expiration:** `src/radar/wide_spread_rule.choose_expiration` (`60–120 DTE`, nearest `90d`, preferring monthly third-Friday).
* **Legs (`select_legs`):**
  * **Bull Put (`bull_put`):** Short put nearest `-0.25Δ`, long put `5%` of spot lower, `max_friction_to_credit = 0.25`.
  * **Bear Call (`bear_call`):** Short call nearest `+0.25Δ`, long call `5%` of spot higher, `max_friction_to_credit = 0.25`.
* **Exit Rules:** Checked weekly on every subsequent Wednesday `15:45 ET` OPRA quote:
  1. Take profit at `50%` of entry credit (`debit_close <= 0.50 * credit_in`).
  2. Hard exit at `<= 21 DTE` (`21dte`).
  3. Cash intrinsic settlement if `<= 0 DTE`.
* **Execution Policies & Fees:**
  * IBKR fees: `$0.65/contract × 4 legs = $2.60/spread` (`$0.026/share`) deducted from every trade.
  * **Primary Policy:** Realistic limit fill `k = 0.35` (conceding 35% of each leg's half-spread on both entry and exit).
  * **Stress Policy:** Full crossed NBBO touch (`k = 1.0`) + mid diagnostic (`k = 0.0`).
* **Entries:** `2015-01-07` through `2026-06-24` (leaving ≥ 3 months for every open trade to settle by `2026-09-30`).

## Binding Hypotheses & Multiple-Testing Gate (6 Arms)

Six arms are tested simultaneously: `{TLT, QQQ, IWM} × {Bull Put, Bear Call}` → **Bonferroni `α = 0.05 / 6 = 0.00833` (`99.17%` CI)** via moving-block monthly cluster bootstrap (`block = 3` months, `10,000` draws, seed `20261127`).

An arm **PASSES** (earning `VALIDATED (limit-only)` promotion) **only if ALL four criteria hold simultaneously**:
1. Delta-adjusted excess `x_entry = (pnl - net_delta * (S_exit - S_entry)) / CAR` has **`99.17% CI lo > 0`** at `k = 0.35` limit fills (net IBKR fees).
2. Raw Profit Factor `PF > 1.0` (`99.17% CI lo > 1.0`).
3. Effective independent sample size `effN >= 30`.
4. **Sub-period stability:** Pointwise `x_entry > 0` in **BOTH** halves: `H1 (2015–2020)` AND `H2 (2021–2026)`.

Any arm failing any criterion is **`CLOSED`** (or `UNDERPOWERED` if `effN < 30`). Non-binding diagnostics: `k=1.0` crossed survival, 2022–2023 bond-bear regime breakdown for `TLT`, and monthly return correlation between `TLT`, `QQQ`, and `IWM` O1 sleeves.
