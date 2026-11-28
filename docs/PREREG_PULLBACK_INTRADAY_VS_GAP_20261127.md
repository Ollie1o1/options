# Preregistration — Study 5B: Single-Stock Intraday Selloff vs. Overnight Gap-Down Filter on `equity_pullback`

**Registered:** 2026-11-27 (frozen before running `scripts/backtest_pullback_intraday_vs_gap.py`).

## Motivation

In Study 2E (`E2b`, ETF 5-minute bars 2016–2026), uptrend ETFs selling off **intraday** (`open → close <= -1.5%`) rebounded reliably (`+30.5 bps [+0.7, +58.5]`, positive in both halves), whereas **overnight gap-downs** failed (`+14.1 [-36.8, +60.9]`). On single stocks (`equity_pullback`, `LEAD`), overnight gap-downs are disproportionately earnings/guidance shocks, downgrades, or structural breaks, whereas intraday flushes often reflect forced liquidity selling. This study tests at `$0` cost whether separating intraday selloffs from overnight gap-downs improves `equity_pullback` on our `375`-stock liquid universe (`2018–2026`, `10 bps` round-trip).

## Frozen Universe, Baseline Parity & Split

* **Base Strategy (exact parity with `scripts/backtest_equity_pullbacks.py`):**
  * `data/equity_ohlcv.db` (`375` liquid stocks, official split-adjusted closes/highs/lows/volumes).
  * Entry criteria on signal day `t`: `Close >= $10`, 20d median dollar volume `>= $20M`, `Close > SMA200`, `12-1m > 0`, `RSI(2) < 10`, `3d return <= -3%`, no bad jumps (`|r| > 45%`), no earnings within 10d when covered.
  * Fill & Exit: Next-day close fill (`t+1`), exit on `Close > SMA5`, `RSI(2) > 70`, `-7%` stop, or `10d` time stop (filled next close), **`10 bps` round-trip cost**.
* **Train / Holdout Split (identical to `PREREG_EQUITY_PULLBACKS_20261124.md`):**
  * `TRAIN`: `2018-10-01` .. `2022-12-31`
  * `HOLDOUT`: `2023-01-24` .. latest (`15` trading-day purge embargo)

## Frozen Signal-Day Selloff Classification (Zero Lookahead — Known at Close of Day `t`)

1. **Primary Classification (`OHLCV`-native, 100% offline in `data/equity_ohlcv.db`):**
   * **`TRUE_GAP_DOWN`:** `high[t] < close[t-1]` (stock gapped down overnight and never traded back to prior close all day).
   * **`INTRADAY_FLUSH`:** `high[t] >= close[t-1]` AND `(close[t] / high[t] - 1) <= -0.025` (traded at/above prior close, then sold off ≥ 2.5% intraday into the close).
   * **`MILD_DRIFT`:** Remaining triggers (`high[t] >= close[t-1]` and intraday fade `> -2.5%`).
2. **Secondary Exact-Open Verification (Free Alpaca Daily Bars `open`/`close`, split-consistent within-day ratios):**
   * **`OVERNIGHT_GAP_TRAP`:** `open[t] / prev_close[t] - 1 <= -0.015` (gapped down ≥ 1.5% overnight).
   * **`INTRADAY_CLEAN`:** `open[t] / prev_close[t] - 1 > -0.015` AND `close[t] / open[t] - 1 <= -0.015` (clean intraday selloff ≥ 1.5% without a large overnight gap).

## Binding Hypotheses & Promotion Gate (Bonferroni `α = 0.025`, `97.5%` ISO-Week Cluster Bootstrap CI)

Two primary hypotheses evaluated on **`HOLDOUT` (`2023-01-24+`)** with `10,000` ISO-week cluster bootstrap draws (`seed = 20261127`, `min_clusters = 40`):

* **H5B-1 (Exclude Gap-Down Traps — `OHLCV`-native):** `NON_GAP` (`high[t] >= close[t-1]`) beats `TRUE_GAP_DOWN` (`high[t] < close[t-1]`) in paired/pooled holdout mean net return (`Δ_mean > 0` with `97.5% CI lo > 0`) AND same positive sign in `TRAIN` (`2018–2022`).
* **H5B-2 (Intraday Flush Focus — `OHLCV`-native):** `INTRADAY_FLUSH` outperforms `TRUE_GAP_DOWN` (`97.5% CI lo > 0` on holdout, positive in train) AND achieves standalone `97.5% CI lo > 0` and `PF > 1.0` on holdout.

Exact Alpaca `Open`-based counterparts (`H5B-1-OPEN`, `H5B-2-OPEN`) are reported identically as confirmatory robustness checks. Only a filter passing its `97.5%` holdout CI and `TRAIN` sign check is wired into `src/radar/scanners.py` / `[6] DISCOVER` (`equity_pullback`).
