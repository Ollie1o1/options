# Equity Playbook 1: Uptrend Pullback Mean Reversion (Shares) — REGISTRATION

**Frozen 2026-11-24, before any outcome has been computed on this strategy.**
Part of the stock-first daily profit reset (Phase 1 of the execution plan).
Adheres strictly to `docs/RESEARCH_PROCESS.md` steps 1–9.

---

## Provenance

- **What has already been seen:**
  - `docs/OUTLOOK_FINDINGS.md` (2026-06-08): cross-sectional sector/asset ETF ranking (12-1 momentum, 200d trend, 1m reversal, relative strength vs SPY) has measured out-of-sample IC `+0.053` (42d) and `+0.076` (63d), with 66.5%–71.6% bullish hit rate.
  - `docs/RESEARCH_FINDINGS.md` (2026-10-10..14): short-horizon directional options lost ~8% of premium primarily due to the options bid-ask spread (~0.70% underlying cost hurdle vs 0.14% drift).
  - `src/longterm/discover.py`: computes 200d MA distance, 12-1 momentum, RSI(14), and support levels descriptively; never backtested as a short-horizon equity swing rule.
- **What has NOT been seen:**
  - Zero returns, win rates, or parameter sweeps for short-term equity pullback entries in `data/squeeze_prices.db` or `data/equity_ohlcv.db`. Nothing below was tuned on outcomes.

---

## Economic Mechanism

Liquid US equities in confirmed structural uptrends (`Close > SMA200`, positive `12-1m` momentum) experience transient 2–5 session selloffs driven by liquidity/rebalancing flows and retail stop-runs rather than fundamental regime breaks. In options, a 1–3% 5-day rebound is destroyed by bid-ask spread, theta, and IV crush. In **shares** on a $0-commission account (Wealthsimple USD / IBKR), round-trip spread friction on liquid mid/large caps is ~2–5 bps — roughly 50× lower — making short-horizon mean reversion inside uptrends structurally testable net of costs.

---

## Population & Data Sources

Two strata tested under the **same frozen rule**, kept strictly separate (never pooled without stratum breakdown):

1. **Primary (Survivorship-Free Broad Universe):** `data/squeeze_prices.db` (`px` table: 21,244 symbols, daily closes & volumes, `2017-10-02` to `2026-08-07`).
   - **Liquidity & Price Gate (computed strictly from data `<= t`, never future data):**
     - `close[t] >= $10.00`
     - `median_20d(close * volume) >= $20,000,000` (institutional-grade liquid mid/large caps where bid-ask spreads are ≤ 3–5 bps)
     - At least `252` trailing trading sessions present prior to `t` (so `SMA200` and `12-1m` momentum are well-defined without NaN substitution).
   - **Data Integrity Guard:** Exclude any bar where `abs(close[t] / close[t-1] - 1) > 0.45` (unadjusted split or bad tick guard matching `DATA_COVERAGE_AUDIT_20261124.md`, flagged & counted, never guessed).
2. **Secondary (Auditable Adjusted Liquid Universe):** `data/equity_ohlcv.db` (`ohlcv` table, adjusted closes, expanded in Phase 0 to ~75 liquid names + Outlook ETFs, `2017-10-02` to `2026-11-23`).

### Locked Chronological Split (Purged)
- **In-Sample / Development Check:** `2018-10-01` to `2022-12-31` (spans 2018 Q4, 2020 COVID crash, 2021 rally, and 2022 bear market).
- **Purge Embargo:** `15` trading days between train end and holdout start (`> max_hold = 10` trading days, guaranteeing zero label overlap).
- **Locked Out-of-Sample Holdout:** `2023-01-24` to `2026-08-07` (`squeeze_prices.db`) / `2026-11-23` (`equity_ohlcv.db`).
- No parameter may be altered between train and holdout.

---

## Strategy Specification (Frozen — 1 Primary Rule, 0 Sweeps)

All indicators are computed strictly from closes up to and including signal day `t`.

### Entry Rule (Evaluated at Close of Day `t`)
A ticker `i` triggers an entry signal at close of trading day `t` iff **all** hold:
1. **Structural Uptrend:** `close[t] > SMA200[t]`
2. **Intermediate Momentum:** `ret_12_1[t] = close[t - 21] / close[t - 252] - 1.0 > 0.0`
3. **Short-Term Oversold Pullback:** `RSI_2[t] < 10.0` (2-period Wilder RSI computed strictly on `close[..t]`) **AND** `close[t] / close[t - 3] - 1.0 <= -0.03` (at least a 3% 3-day pullback so tiny low-vol wiggles don't trigger).
4. **Earnings Exclusion (where covered):** Exclude entries where `data/dolt_options.db::earnings_cal` has an earnings date in `[t, t + 10 calendar days]`. Symbols not in `earnings_cal` are flagged `earnings_coverage = unknown` and reported separately in diagnostics (never silently assumed clear).
5. **No Duplicate Overlapping Position in Same Ticker:** A ticker already in an open trade cannot open a second position until after exit day.

### Execution Timing & Friction (No Look-Ahead)
- **Primary fill model:** Signal detected after close of day `t` → **enter at close of day `t + 1`** (conservative 1-session execution delay; eliminates close-price microstructure bounce / bid-ask bounce bias!).
- **Exit fill model:** Exit condition checked at close of day `k` (`k >= t + 2`) → **exit at close of day `k + 1`**? No — exit rule evaluates intraday/close at day `k`: to remain strictly free of close-to-close bounce inflation, primary spec executes **signal on `t` → enter `t + 1` close; exit condition on `k` → exit `k + 1` close** (next-session close on BOTH ends!).
- **Friction:** **`5 bps` (`0.0005`) per side = `10 bps` (`0.0010`) round-trip** deducted from every trade return (conservative vs 1–3 bps on `$20M+/day` US equities at Wealthsimple USD $0 commission).
- *(Sensitivity, no authority: enter at `t` close / exit at `k` close — standard MOC execution — reported alongside to quantify the 1-day delay penalty.)*

### Exit Rule (Checked Daily Starting `t + 2`, In Order)
1. **Mean-Reversion Target:** `close[k] > SMA5[k]` **OR** `RSI_2[k] > 70.0`
2. **Hard Stop-Loss:** `close[k] / entry_price - 1.0 <= -0.07` (`-7%` stop)
3. **Time Exit:** `k - entry_idx >= 10` trading days

### Portfolio Construction & Benchmark
- **Equal-risk per trade** for per-trade statistics: `r_i = (exit_price / entry_price - 1.0) - 0.0010`.
- **Matched SPY Benchmark (`H_i`):** For every trade `i` open from `entry_date` to `exit_date`, compute `b_i = SPY_close[exit_date] / SPY_close[entry_date] - 1.0`.
- **Excess Return:** `x_i = r_i - b_i` (answers: *"did buying this pullback beat holding SPY over the exact same days?"*).
- **Capacity-Constrained Book Simulation (Secondary Diagnostic):** Max `5` concurrent positions (`20%` of equity each; if >5 signals fire on day `t`, select the 5 deepest pullbacks by lowest `RSI_2[t]` — pre-declared tie-breaker, never tuned).

---

## Observation Unit, Clustering & Inference

- **Cluster unit:** **ISO week of entry** (`(iso_year, iso_week)`). Signals cluster heavily during market-wide dips; treating simultaneous trades as independent would inflate t-stats by 3–5×.
- **Effective N check:** Compute `icc_oneway`, `design_effect`, and `effective_n` via `src/prereg_ranker.py` and `src/alloc/validate.py::effective_n`.
- **Minimum Sample Floor:** `MIN_CLUSTERS = 40` distinct entry weeks in the holdout window (if `< 40`, verdict is `INSUFFICIENT`).
- **Bootstrap:** `10,000` resamples of ISO entry weeks with replacement, fixed seed `20261124`, percentile 95% CI.
- **Promotion Gate:** Run through `src/alloc/report.py::summarise` and `promotion_verdict` (`n_trials = 1` for this single rule, `MIN_DSR = 0.5`, `MIN_TSTAT = 3.0` on clustered t-stat).

---

## Hypotheses & Binding Verdicts

### H-A (Primary — Absolute Net Edge on Holdout)
- **Statistic:** Mean net per-trade return `mean(r)` and Profit Factor `PF(r) = sum(max(r,0)) / |sum(min(r,0))|` on the **2023–2026 locked holdout** (broad survivorship-free universe), week-clustered 95% CI.
- **Pass:** `mean(r)` 95% CI lower `> 0` **AND** `PF(r)` 95% CI lower `> 1.0` **AND** same positive sign in the 2018–2022 train window.
- **Fail:** Either CI crosses null or train/holdout signs disagree.

### H-B (Secondary — Beats Matched SPY Exposure on Holdout)
- **Statistic:** Mean excess return `mean(x = r - b)` on the **2023–2026 locked holdout**, week-clustered 95% CI.
- **Pass:** `mean(x)` 95% CI lower `> 0`.
- *(Note: Even if H-B is near zero, if H-A passes with materially higher Sharpe / lower drawdown in the capacity-constrained book than SPY buy-and-hold, record that honestly without overriding H-B's fail.)*

### Sensitivities (Reported Alongside — Zero Decision Authority)
1. Same-day close execution (`t` close entry, `k` close exit) vs next-day close (`t+1` / `k+1`).
2. Friction stress at `20 bps` (`0.0020`) round-trip.
3. Sector-filter arm: restricting entries to tickers mapped in `SECTOR_MAP` whose sector ETF is `FAVOR` or `NEUTRAL` (excluding `AVOID`) under `src/outlook/engine.py`.
4. Per-year breakdown (`2018` through `2026`) calling out **2020** and **2022** bear/crash regimes.
5. `earnings_coverage = known_clear` vs `unknown` split.
6. Capacity-constrained 5-slot equity curve vs SPY buy-and-hold (CAGR, Sharpe, Max Drawdown, time in market %).

---

## Authority & Prior

- **Authority:** A pass promotes `equity_pullback` from `[RESEARCH]` to `[VALIDATED]` in `configs/playbooks.json` and surfaces it on the daily Radar (`src/radar/`) for **paper / `own_picks` forward tracking** (minimum 8 forward weeks before any real-capital pilot per `docs/DECISION_RULE_20261015.md`). It authorizes **no real money** and modifies no existing options gate.
- **Prior (stated before running):** Moderate-to-high chance H-A passes on MOC fills; tighter under conservative `t+1` close delay (classic 1-day bounce decay). Key risk is **2022 bear-market whipsaw** when `SMA200` lags regime turns — which is why 2022 is explicitly reported.

**Estimator:** `scripts/backtest_equity_pullbacks.py` (backed by `src/equity_playbooks/backtest.py`).
**Result file:** `docs/RESULT_EQUITY_PULLBACKS_20261124.md`.
