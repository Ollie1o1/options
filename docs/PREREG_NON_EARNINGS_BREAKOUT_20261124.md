# Equity Playbook: Non-Earnings Volume Breakout (Shares) — PREREGISTRATION

**Frozen 2026-11-24, before any outcome has been computed on this strategy.**
Part of Phase 4 of the profit-engine expansion, immediately following the 6× expansion of `data/equity_ohlcv.db` (`62 → 371` liquid US stocks/ETFs).
Adheres strictly to `docs/RESEARCH_PROCESS.md` steps 1–9.

---

## Provenance

- **What has already been seen:**
  - `docs/RESULT_PEAD_EQUITY_20261124.md`: Post-earnings breakouts on the large-cap Dolt universe failed out-of-sample (PF 1.296, 95% CI `[0.860, 2.011]`, excess `+0.04%` vs SPY) → **CLOSED**.
  - `docs/RESULT_EQUITY_PULLBACKS_20261124.md`: Short-horizon uptrend pullbacks in shares passed H-A on both broad (`PF 1.232 [1.019, 1.496]`) and liquid (`PF 1.431`) holdouts at 10 bps friction, while failing H-B vs SPY → **LEAD**.
  - `src/breakout/backtest.py`: Measured raw 20-day breakout probabilities vs realized volatility and found no edge vs vol in options.
- **What has NOT been seen:**
  - Zero returns, profit factors, or benchmark excess numbers have been computed for **non-earnings high-volume breakouts (`+5% day on >=2× 20d volume` to a new 20d high in a confirmed uptrend)** held in **shares** at `10 bps` friction across either `data/squeeze_prices.db` or the newly expanded 371-ticker `data/equity_ohlcv.db`.

---

## Economic Mechanism

Institutional accumulation triggered by non-earnings catalysts (product launches, contract awards, sector rotation, or analyst re-ratings) often takes multi-day execution slices to complete, creating measurable short-to-medium horizon continuation (5–20 trading days). Conversely, retail breakout chasing often mean-reverts sharply after a single `+5%` spike. Testing **shares-only** at `10 bps` round-trip friction with explicit earnings exclusion cleanly separates genuine institutional drift from either PEAD or options-spread drag.

---

## Population & Data Sources

Two strata tested under the **exact same frozen rule**, reported separately:

1. **Primary (Survivorship-Free Broad Universe):** `data/squeeze_prices.db` (`px` table: 21,244 symbols, daily closes & volumes, `2017-10-02` to `2026-08-07`).
   - **Liquidity & Price Gate (computed strictly from data `<= t`):**
     - `close[t] >= $10.00`
     - `median_20d(close * volume) >= $20,000,000`
     - At least `252` trailing trading sessions prior to `t`.
   - **Data Integrity Guard:** Exclude any bar where `abs(close[t] / close[t-1] - 1) > 0.45` (split/bad-tick guard).
2. **Secondary (Auditable Adjusted Liquid Universe):** `data/equity_ohlcv.db` (`ohlcv` table: 371 liquid US stocks & ETFs, `2017-10-02` to `2026-11-24`), excluding pure sector/index ETFs (tested on individual equities, plus full-universe diagnostic).

### Locked Chronological Split (Purged)
- **In-Sample / Development Check:** `2018-10-01` to `2022-12-31`
- **Purge Embargo:** `25` trading days (`> max_hold = 20` trading days, guaranteeing zero label overlap)
- **Locked Out-of-Sample Holdout:** `2023-02-07` to `2026-08-07` (`squeeze_prices.db`) / `2026-11-24` (`equity_ohlcv.db`).
- Zero parameter changes allowed between train and holdout.

---

## Frozen Strategy Specification (1 Rule, 3 Pre-Registered Horizons)

All indicators are computed strictly from data up to and including signal day `t`.

### Entry Rule (Close of Day `t`)
A ticker triggers an entry signal at close of day `t` iff **all** hold:
1. **Structural Uptrend:** `close[t] > SMA200[t]`
2. **Price Surge:** `close[t] / close[t - 1] - 1.0 >= 0.05` (`+5.0%` daily gain)
3. **Volume Confirmation:** `volume[t] >= 2.0 * mean(volume[t - 20 : t])` (`>= 2.0×` trailing 20d avg volume)
4. **20-Day Closing High:** `close[t] >= max(close[t - 20 : t])`
5. **Non-Earnings Filter:** Exclude any signal where `data/dolt_options.db::earnings_cal` has an earnings date within `[t - 2 calendar days, t + 2 calendar days]`. (Diagnostics also report the subset of tickers covered by `earnings_cal` separately.)
6. **No Overlapping Position in Same Ticker:** A ticker already open in horizon `H` cannot re-enter until after exit day.

### Execution Timing & Friction (No Look-Ahead)
- **Fill Model:** Signal detected after close of day `t` → **enter at close of day `t + 1`** → **exit at close of day `t + 1 + H`** for `H ∈ {5, 10, 20}` trading days. (Eliminates same-day close-bounce bias.)
- **Friction:** `5 bps` per side = `10 bps` (`0.0010`) round-trip deducted from every trade return.
- **Matched Benchmark (`SPY`):** For every trade `[entry_date, exit_date]`, `b_i = SPY[exit_date] / SPY[entry_date] - 1.0`, `excess_i = r_i - b_i`.

---

## Inference & Multiple-Testing Correction

- **Cluster Unit:** ISO entry week (`YYYY-Www`).
- **Bootstrap:** Circular moving-block bootstrap (`block_len = 4` weeks to absorb up to 20d hold overlap), `10,000` resamples, fixed seed `20261124`.
- **Bonferroni Family Correction (3 horizons):** `alpha = 0.05 / 3 = 0.01667` (`98.33%` two-sided CI).
- **Minimum Cluster Floor:** `MIN_CLUSTERS = 40` distinct entry weeks in holdout.

---

## Binding Promotion Verdict

Evaluated strictly on the **2023–2026 Locked Holdout**:
- **H-A (Absolute Net Edge):** `PF CI_lo > 1.00` AND `mean_ret CI_lo > 0.0` at Bonferroni `98.33%` CI.
- **H-B (Beat SPY):** `mean_excess vs SPY CI_lo > 0.0` at Bonferroni `98.33%` CI.

| Outcome | Action |
|---|---|
| Passes **both H-A and H-B** on Primary Broad Holdout AND replicates directionally (`mean_ret > 0`, `mean_excess > 0`) on Secondary 371-Ticker Liquid Holdout | Promote to **`VALIDATED`** in `configs/playbooks.json` and wire live scanner into Unified Radar |
| Passes **H-A only** on BOTH Primary and Secondary Holdouts (but fails H-B vs SPY) | Register as **`LEAD`** (shares-only, with regime caveats) in `configs/playbooks.json` |
| Fails H-A on Primary Holdout OR fails to replicate (`mean_ret <= 0` or `PF <= 1.0`) on Secondary Liquid Holdout | Register as **`CLOSED`** in `configs/playbooks.json` (do NOT add to Radar; protect user from breakout chasing) |
