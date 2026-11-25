# Equity Playbook 3: Path-Managed Short-Squeeze Momentum (Shares) — REGISTRATION

**Frozen 2026-11-24, before any equity trading P&L simulation has been run on `src/squeeze/backtest/`.**
Part of the stock-first daily profit reset (Phase 1 of the execution plan).
Adheres strictly to `docs/RESEARCH_PROCESS.md` steps 1–9.

---

## Provenance

- **What has already been seen (`docs/SQUEEZE_BACKTEST.md`, 2026-07-28..08-07):**
  - Across 205 FINRA settlement dates (`2018-01` to `2026-08`, 480,744 graded observations, survivorship-free `data/squeeze_prices.db`, exact +8 trading-day FINRA dissemination lag), the rule **Top-5% Short Interest (`>= ~19%` of assumed float) AND `5d return >= +10%`** (`~16` names/settlement date) produced:
    - `P(+20% path max within 42d) = 50.5%` (vs `22.5%` base rate)
    - `P(+30% within 42d) = 29.9%–37.6%`
    - `P(+50% within 42d) = 21.8%` (vs `4.8%` base rate)
    - **Median path max: `+20.5%`**, while **median 42d endpoint return: `-1.9%`**!
    - Right-tail asymmetry `+4.43pp` OOS (`2023–2026` holdout), surviving exclusion of the Jan 2021 meme window.
- **Why this has NEVER been monetized yet:**
  - The prior study measured *path maximums* only, never an executable **trading rule** with stops, trailing exits, and realistic small/mid-cap execution friction. Because high-SI names carry expensive IV (~100%+), buying blind calls is untestable and likely negative-EV — whereas **trading shares with path-managed exits** directly targets the `+20.5%` median peak while cutting the `-1.9%` give-back!
- **What has NOT been seen:**
  - Zero realized trade returns, win rates, drawdowns, or stop-out rates for any equity exit rule on this panel.

---

## Population & Data Sources

- **Data:** Exact existing survivorship-free stores — **reused directly, never rebuilt**:
  - `data/squeeze_backtest.db` (`si`, `si_dates`, `shares_out`) — FINRA consolidated short interest, `2018-01` to `2026-08`.
  - `data/squeeze_prices.db` (`px`) — DoltHub daily closes & volumes including delisted tickers (`BBBY`, `MULN`, `NKLA`, etc.).
- **Point-in-Time Entry Lag (Non-Negotiable):** Entry occurs **strictly `>= 8` trading days after each FINRA settlement date** (the exact public dissemination lag proven in `docs/SQUEEZE_BACKTEST.md`), plus **1 additional session** (`entry_bar = dissemination_bar + 1`) so the `5d return >= +10%` filter measured at `dissemination_bar` close never enters at the same close that triggered it!
- **Tradability Filters (Computed strictly `<= dissemination_bar`, zero look-ahead):**
  - `close >= $5.00` (eliminates sub-$5 illiquid penny stocks)
  - `median_20d(close * volume) >= $5,000,000` (tradeable liquidity floor for retail equity fills)
  - `SI_pct_float` in **top 5% cross-sectionally** on that settlement date (using the same `si_scale = 1.25` = 80% float assumption as `SQUEEZE_BACKTEST.md`)
  - `5d_return` up to `dissemination_bar` `>= +10.0%`
  - **Max 1 active position per ticker** (no re-entry while already held).
- **Delisting / Missing-Price Conservative Treatment:** If a held stock ceases trading (`ticker_ended`) before hitting an exit rule, book exit at its **last available close minus an extra 20% liquidation penalty** (or track `ticker_ended` count explicitly; never drop delisted losers!).

### Locked Split
- **Period 1:** `2018-01-01` to `2022-12-31` (includes 2020 crash, Jan 2021 meme mania, 2022 bear market).
- **Period 2 (Holdout):** `2023-01-01` to `2026-08-07` (post-meme regime).
- **Robustness Cut (Mandatory):** Every headline re-reported **excluding `2021-01-01` .. `2021-03-31`** (meme window) so GME/AMC cannot carry the result!

---

## Frozen Exit Arms (Exactly 3 Pre-Declared Rules — No Grid Sweep!)

All stops/targets are evaluated on daily closes `k > entry_idx` and **executed at `k + 1` close** (conservative next-day fill; catches overnight gap-through-stop honestly!).

| Arm | Initial Hard Stop | Trailing Stop (from Peak Close Since Entry) | Profit-Taking Rule | Max Hold |
|---|---|---|---|---|
| **S1 (Tight Trail)** | `-12%` from entry | `-12%` from running peak close | None (let winners run until trail fires) | 42 trading days |
| **S2 (Scaled + Trail — Primary)** | `-12%` from entry | `-15%` from running peak close | Sell **50% of shares** on first close `>= +20%`; remaining 50% rides `-15%` trail | 42 trading days |
| **S3 (Target / Stop Bracket)** | `-12%` from entry | None | Take profit 100% on first close `>= +25%` | 42 trading days |

### Transaction Costs (Conservative Small/Mid-Cap Slippage)
- **`20 bps` (`0.0020`) per side = `40 bps` (`0.0040`) round-trip** deducted on every trade (8× higher than large-cap Assumption in Playbook 1, reflecting wider spreads on volatile high-SI equities).
- *(Sensitivity, no authority: `100 bps` = `1.0%` round-trip stress test.)*

---

## Observation Unit, Clustering & Inference

- **Cluster unit:** **FINRA settlement date** (~11 trading days apart). Because holds run up to 42 trading days (overlapping ~4 consecutive settlement cycles), use a **moving-block bootstrap over settlement dates with block length = 4** (`10,000` draws, seed `20261124`), matching `src/squeeze/backtest/study.py`!
- **Multiplicity:** Bonferroni across the 3 arms (`98.33%` two-sided CI, `alpha = 0.0167`).
- **Minimum Sample Floor:** `MIN_SETTLEMENT_DATES = 50` in the `2023–2026` holdout (and `>= 100` full sample).
- **Benchmark:** Matched-window `SPY` return `b_i` over each trade's exact `[entry_date, exit_date]`, plus **Matched Random-Entry High-SI Control** (same SI>=top 5% without `+10%` momentum) and **Matched Hold-to-42d-Endpoint Control** (verifying whether path management actually adds value over holding to endpoint!).
- **Promotion Gate:** `src/alloc/report.py::summarise(n_trials=3)` + `promotion_verdict`.

---

## Binding Pass Rules (Per Arm)

An arm **PASSES** iff **all four** hold:
1. Full-sample **excluding Q1 2021 meme window**: moving-block 98.33% CI on mean net return `> 0` **AND** PF CI lower `> 1.0`.
2. **2023–2026 holdout**: point estimate positive (`mean(r) > 0`, `PF > 1.10`) and 95% CI lower `> 0`.
3. **Beats matched SPY exposure:** `mean(r - b)` 95% CI lower `> 0` (full ex-meme sample).
4. **Survives single-ticker dominance check:** Still positive mean return and `PF > 1.0` after dropping the top 3 highest-contributing tickers.

### Sensitivities (No Authority)
1. `100 bps` (1.0%) round-trip slippage stress test.
2. Liquidity floor raised to `$20M/day` average dollar volume (large/liquid mid-caps only).
3. Capacity-constrained book: max `4` concurrent positions (`10%` equity each, max 40% total sleeve exposure — appropriate for a high-vol satellite sleeve).
4. Path-managed return vs raw 42d hold-to-endpoint return (directly quantifying the path-management lift).

---

## Authority & Prior

- **Authority:** Passing promotes `squeeze_equity` to `[VALIDATED]` as a **small satellite equity sleeve** (capped at small position sizes) surfaced on the daily Radar (`src/radar/`) with live trailing-stop alerts — replacing the unvalidated high-IV call-buying assumption. Authorizes **no real money** until 8+ forward paper/`own_picks` weeks confirm fill slippage.
- **Prior (stated before running):** Genuine uncertainty. The `+20.5%` median peak vs `-1.9%` median endpoint is real, *but* high-SI stocks gap violently downward overnight (overshooting `-12%` stops on daily closes), which can turn an asymmetric right tail into a break-even or noisy trade once stop-overshoot and 40 bps friction are charged honestly. That gap-through-stop effect is exactly what this test is designed to measure!

**Estimator:** `scripts/backtest_squeeze_equity.py` (backed by `src/equity_playbooks/backtest.py`).
**Result file:** `docs/RESULT_SQUEEZE_PATH_EQUITY_20261124.md`.
