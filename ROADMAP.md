# Options Screener — Profitability Roadmap

This document covers three areas in priority order:

1. **System fixes** — bugs and disconnected feedback loops that are silently costing you money
2. **Platform improvements** — features the audit found half-built or missing entirely
3. **Alpha research** — mathematical and research-based approaches for finding profitable single long/short options

Each section has a status tag, implementation notes pointing at specific files, and enough context to execute immediately.

---

## How to use this document

Bring it into a Claude Code session and say:  
_"Execute ROADMAP.md section [N]"_  

Each section is self-contained. Sections are ordered by ROI — do them in order if unsure where to start.

---

## Status legend

- `[DONE]` — already implemented in the current codebase
- `[PARTIAL]` — infrastructure exists, logic disconnected or broken
- `[BUILD]` — net-new feature, nothing in place yet
- `[RESEARCH]` — mathematical concept to implement as a new scoring signal

---

---

# Part 1 — System Fixes

## 1.1 Close the Kelly sizing feedback loop `[PARTIAL]`

**Why it matters:** Your position sizer (`src/trade_analysis.py:315`) computes Kelly fraction using hardcoded 50% TP / 25% SL estimates rather than your actual historical win rates. The win rates already exist in `paper_manager.py:1354` (`get_strategy_breakdown()`). The two pieces have never been wired together, so sizing is essentially random relative to your actual edge.

**What to build:**
- In `src/trade_analysis.py`, call `PaperTradeManager.get_strategy_breakdown()` at startup
- Map strategy names to the Kelly inputs: `win_rate`, `avg_win_pct`, `avg_loss_pct`
- Fall back to hardcoded defaults when fewer than 20 closed trades exist for a strategy (insufficient sample)
- Add a minimum sample guard: don't update Kelly inputs until N≥20 closed trades per strategy
- Re-run the Kelly calculation per strategy, not globally

**Files:** `src/trade_analysis.py`, `src/paper_manager.py`  
**Config key to add:** `"kelly_min_sample": 20`

---

## 1.2 Enforce portfolio Greeks limits at entry `[PARTIAL]`

**Why it matters:** `config.json` defines `portfolio_gex_limit: 50000` and portfolio Greeks are computed in `src/portfolio_risk.py:215`. But nothing in the trade-logging path checks these limits before accepting a new position. You can unknowingly build a portfolio with 10 correlated long calls and no hedge.

**What to build:**
- In `src/paper_manager.py`, before `log_trade()` commits to SQLite, call `RiskAggregator.get_portfolio_greeks()`
- Check: `abs(portfolio_delta) < config["max_portfolio_delta"]`, `abs(portfolio_gex) < config["portfolio_gex_limit"]`, `portfolio_vega < config["max_portfolio_vega"]`
- If any limit is breached, print a warning and require `--force` flag or confirmation prompt to proceed
- Add config keys: `"max_portfolio_delta": 50`, `"max_portfolio_vega": 5000`

**Files:** `src/paper_manager.py`, `src/portfolio_risk.py`, `config.json`

---

## 1.3 Fix the backtest signal mismatch `[PARTIAL]`

**Why it matters:** `src/backtester.py:329` uses `0.5×momentum + 0.3×hv_rank + 0.2×rsi` as its predictive signal. Your live screener uses a 27-component `quality_score`. These are different models. When the backtest says "IC = 0.35" it is testing the proxy signal, not the real one. You have no idea if your actual scoring works historically.

**What to build:**
- In `src/backtester.py`, replace the proxy signal (lines 329–334) with the actual `quality_score` column if present in the historical DataFrame, falling back to the proxy only when `quality_score` is unavailable
- Add a parity check: compute IC between proxy signal and quality_score on the same date range; log a warning if IC < 0.30 (they diverge)
- Add a `--signal proxy|quality` CLI flag so you can compare both

**Files:** `src/backtester.py`

---

## 1.4 Fix backtester / live exit rule mismatch `[PARTIAL]`

**Why it matters:** The backtester uses static 50% TP / 25% SL. Live paper trades use DTE-tiered profit targets (50% TP at ≥21 DTE, 35% at 7–21 DTE, 25% at <7 DTE). These are two different exit systems, so backtested win rates do not match live performance. The validation blind spot means you can't trust backtested Sharpe ratios.

**What to build:**
- In `src/backtester.py`, import the DTE-tiered target logic from `config.json` (`profit_target_tiers`)
- Apply the same DTE-tiered targets during backtest simulation
- Add a `--exit-rules live|static` flag for comparison

**Files:** `src/backtester.py`, `config.json`

---

## 1.5 Regime-aware component weight calibration `[PARTIAL]`

**Why it matters:** The IC calibrator (`src/backtester.py:1334`) blends component scores globally across all market regimes. But IV rank is a strong signal in high-VIX and near-useless in low-VIX. Momentum is the opposite. Blending them globally dilutes both signals.

**What to build:**
- In `src/backtester.py`, segment the IC calculation by VIX regime (Low/Normal/High — already tagged in `vrp_regime` and `volatility_regime` columns)
- Produce three separate weight recommendation sets: `composite_weights_low_vix`, `composite_weights_normal_vix`, `composite_weights_high_vix`
- In `src/options_screener.py`, select the appropriate weight set based on current `volatility_regime` before scoring
- Add all three weight sets to `config.json`

**Files:** `src/backtester.py`, `src/options_screener.py`, `config.json`

---

## 1.6 Greeks P&L attribution at close `[BUILD]`

**Why it matters:** When a trade closes you record profit/loss but not *why*. Without decomposition you cannot tell if you are genuinely making money from theta decay or accidentally from directional bets. One is a repeatable edge; the other is luck.

**What to build:**
- In `src/paper_manager.py`, when `close_trade()` is called, re-price the option using Black-Scholes with the exit-date underlying price, IV, and DTE
- Compute delta P&L (`entry_delta × ΔS`), theta P&L (`entry_theta × Δt`), vega P&L (`entry_vega × ΔIV`), gamma P&L (`0.5 × entry_gamma × ΔS²`), residual (slippage + model error)
- Store these as columns in the `paper_trades` SQLite table: `delta_pnl`, `theta_pnl`, `vega_pnl`, `gamma_pnl`, `residual_pnl`
- Add a `pnl_attribution` view to the portfolio dashboard

**Files:** `src/paper_manager.py`, `src/utils.py` (Black-Scholes already there)  
**New SQLite columns:** `delta_pnl REAL, theta_pnl REAL, vega_pnl REAL, gamma_pnl REAL, residual_pnl REAL`

---

---

# Part 2 — Platform Improvements

## 2.1 Pre-scan earnings calendar `[BUILD]`

**Why it matters:** Currently earnings dates are fetched per-ticker on demand. Scanning 50 tickers means you discover earnings conflicts mid-scan — sometimes after you've already scored and ranked a play. A batch pre-fetch at scan start lets you flag, skip, or specifically target earnings plays before any scoring runs. It also unlocks the **earnings IV crush scanner** described in section 3.3.

**What to build:**
- New function `src/data_fetching.py`: `batch_fetch_earnings(tickers, horizon_days=14) -> dict[str, date]`
  - Use `yfinance.Ticker(symbol).calendar` for each ticker (parallelised with `ThreadPoolExecutor`)
  - Cache results in memory for the session (no repeated lookups mid-scan)
- In `src/options_screener.py` `run_scan()`, call `batch_fetch_earnings` before the main scan loop
- Attach results to the scan context so `enrich_and_score()` can use pre-fetched dates instead of re-fetching
- Print a pre-scan earnings calendar summary: which tickers have earnings in the next 14 days

**Files:** `src/data_fetching.py`, `src/options_screener.py`

---

## 2.2 Win rate feedback into scoring display `[PARTIAL]`

**Why it matters:** `paper_manager.get_strategy_breakdown()` tracks win rate per strategy. This data is computed but never shown alongside scan results, so you have no reminder of which setups have actually worked for you historically.

**What to build:**
- In `src/cli_display.py`, `print_report()`, add a "Your historical edge" footer panel
- Call `get_strategy_breakdown()` and filter to strategies matching current mode
- Show: win rate, average P&L, trade count, and a confidence indicator (dim if N<10)
- Example: `Short Put: 11/15 wins (73.3%) · avg +3.2% · 15 trades`

**Files:** `src/cli_display.py`, `src/paper_manager.py`

---

## 2.3 Intraday entry timing filter `[BUILD]`

**Why it matters:** Options bid-ask spreads are widest in the first 30 minutes after open and the last 15 minutes before close. Entering during those windows costs 10–30% more in slippage. This compounds across every trade.

**What to build:**
- In `src/options_screener.py`, add `get_entry_timing_score()`:
  - 09:30–10:00 ET: score = 0.3 (wide spreads, price discovery still running)
  - 10:00–15:30 ET: score = 1.0 (optimal window)
  - 15:30–16:00 ET: score = 0.5 (liquidity withdrawing, MM hedging)
- Apply a timing multiplier to `quality_score` only when auto-logging is enabled (not for research scans)
- Print a coloured banner at scan start if current time is in a suboptimal window
- Add a `--ignore-timing` flag for off-hours research use

**Files:** `src/options_screener.py`, `src/cli_display.py`  
**Config key:** `"entry_timing_enabled": true`

---

## 2.4 Sector flow aggregator `[BUILD]`

**Why it matters:** When scanning a watchlist you compute `option_rvol` per ticker but never aggregate it. You cannot see that "semiconductors have elevated call flow today" until individual tickers are already scoring high. A sector-level dashboard surfaces themes before they're obvious.

**What to build:**
- New file `src/sector_flow.py`:
  - `compute_sector_flow(picks_df, sector_map) -> pd.DataFrame`
  - Groups tickers by GICS sector, aggregates: mean option_rvol, call/put ratio, % unusual, net IV trend
  - Returns ranked sector summary
- Sector map: hardcode the 37 liquid_large_cap names to their GICS sectors (or load from config)
- In `src/cli_display.py`, add `print_sector_flow(sector_df)` — a compact table shown at the top of multi-ticker scan output
- In `run_scan()`, call after all tickers are processed but before the main report

**Files:** `src/sector_flow.py` (new), `src/cli_display.py`, `src/options_screener.py`

---

---

# Part 3 — Alpha Research: Finding Super-Profitable Single Options

This section covers mathematical and research-based methods for finding options with genuine edge before they move. These are approaches used by quant desks and academic research that a retail screener can approximate with available data.

Each entry describes the concept, the mathematical formula, what to build, and which files to touch.

---

## 3.1 Implied vs Realised Volatility Arbitrage `[BUILD]`

**The concept:**  
The core edge in buying options is when implied volatility (IV) underprices how much the stock will actually move — i.e. when realised volatility (RV) exceeds IV. You already compute `iv_vs_hv` but use it as a linear signal. The better approach is to compute **variance risk premium (VRP)** in variance space and use a forward-looking RV forecast rather than backward-looking HV.

**The mathematics:**

```
VRP = IV² - RV²          (variance form, not vol form)

If VRP < 0  →  IV is cheap relative to realised vol  →  buying options has positive EV
If VRP > 0  →  IV is rich  →  selling options has positive EV

Forward RV estimate using HAR model:
RV_forecast = α + β₁×RV_1d + β₂×RV_5d + β₃×RV_22d

Where RV_Nd = annualised realised variance over last N days
```

The HAR (Heterogeneous Autoregression) model is the most accurate simple RV forecaster for equities. It captures the clustering of volatility at different timescales (daily traders, weekly rebalancers, monthly allocators). Comparing `IV²` to `HAR_forecast²` gives a more accurate "cheap/expensive" signal than raw `IV vs HV`.

**What to build:**
- In `src/utils.py`, add `har_rv_forecast(daily_returns: np.ndarray) -> float`
  - Requires minimum 22 days of daily returns
  - Coefficients can be OLS-fitted on each stock's own history or use academic defaults: `α=0.0, β₁=0.35, β₂=0.30, β₃=0.35`
- In `src/data_fetching.py`, store the HAR forecast as `har_rv_forecast` alongside existing `hv_30d`
- In `src/options_screener.py`, compute `vrp_har = IV² - har_rv_forecast²` and use it in place of the current `iv_vs_hv` for VRP scoring
- New column: `vrp_har_score` — sigmoid-scaled, inverted for buyers (negative VRP = cheap option)

**Files:** `src/utils.py`, `src/data_fetching.py`, `src/options_screener.py`  
**New score column:** `vrp_har_score`  
**Config weight key:** add `"vrp_har"` to `composite_weights`

---

## 3.2 Merton Jump-Diffusion Theoretical Value vs Market Price `[PARTIAL]`

**The concept:**  
Black-Scholes assumes log-normal returns with no jumps. Real stocks jump. OTM options on volatile or event-prone stocks are systematically underpriced by BS because the model ignores the fat-tailed jump distribution. The Merton Jump-Diffusion (MJD) model adds a Poisson jump process to GBM. When `MJD_price > market_price`, the market is undervaluing jump risk — a buy signal particularly powerful for:
- Biotech before FDA decisions
- Earnings reporters with history of large moves
- High-short-interest stocks prone to squeezes

**The mathematics:**

```
MJD call price = Σ_{n=0}^{∞} [ (e^{-λ'T} × (λ'T)^n / n!) × BS(S, K, T, r_n, σ_n) ]

Where:
  λ'   = λ × (1 + k)          adjusted jump intensity
  k    = E[J-1]                mean percentage jump (e.g., -2%)
  r_n  = r - λk + n×ln(1+k)/T  risk-neutral drift for n jumps
  σ_n  = sqrt(σ² + n×δ²/T)    vol including n jumps
  δ    = jump size std dev

Truncate sum at n=20 (converges quickly)
```

Your Monte Carlo in `src/simulation.py` already uses MJD for PoP/PoT. The theoretical MJD *option price* is not yet computed — only the probability paths.

**What to build:**
- In `src/utils.py`, add `mjd_option_price(S, K, T, r, sigma, option_type, lambda_=2.0, k=-0.02, delta=0.04, n_terms=20) -> float`
  - Implements the MJD closed-form series above
  - Uses the existing `bs_call` / `bs_put` functions for each term
- In `src/options_screener.py`, compute `mjd_value` for each contract and compare to `premium`
- New column: `mjd_discount = (mjd_value - premium) / mjd_value`
  - Positive = market underprices jump risk = option is cheap on MJD basis
  - Negative = option is overpriced on MJD basis
- Add `mjd_discount_score` to the composite scoring: high positive discount = strong buy signal for long options

**Files:** `src/utils.py`, `src/options_screener.py`  
**New columns:** `mjd_value`, `mjd_discount`, `mjd_discount_score`  
**Config weight key:** add `"mjd_discount"` to `composite_weights`

---

## 3.3 Earnings Straddle Systematic Scanner `[BUILD]`

**The concept:**  
Academic research (Goyal & Saretto 2009, Patell & Wolfson 1979) consistently finds that buying ATM straddles 5–7 days before earnings and closing at earnings open generates positive returns when:
1. The stock has a history of moving more than implied (hist_move > implied_move)
2. IV has not already spiked to price in the event (IV percentile < 60th)
3. The stock has beaten earnings estimates in 3 of last 4 quarters

The screener already computes all three of these signals individually but does not combine them into an **earnings straddle score** or a dedicated scanner that proactively finds the best upcoming earnings across the whole watchlist.

**What to build:**
- New file `src/earnings_scanner.py`:
  - `scan_upcoming_earnings(tickers, horizon_days=14, config) -> pd.DataFrame`
  - Uses `batch_fetch_earnings()` (from 2.1) to get all upcoming earnings dates
  - For each ticker with earnings in horizon: fetches ATM straddle cost, IV percentile, hist/implied move ratio, earnings beat rate
  - Computes `earnings_straddle_score`:
    ```
    score = 0.40 × hist_beat_ratio          # hist_move / implied_move, capped at 2.0, norm to 0-1
          + 0.30 × (1 - iv_percentile)      # prefer IV not already spiked
          + 0.20 × earnings_beat_rate        # 0–1 fraction of beats
          + 0.10 × (1 - dte_score)          # prefer 5–7 days out
    ```
  - Returns ranked table of best earnings straddle setups
- New menu option: `[E] EARNINGS` — runs the earnings scanner across the default watchlist
- In `src/cli_display.py`, add `print_earnings_scanner_report(df)`

**Files:** `src/earnings_scanner.py` (new), `src/cli_display.py`, `src/options_screener.py`  
**Depends on:** section 2.1 (batch earnings calendar)

---

## 3.4 Post-Earnings Drift Signal for Directional Options `[BUILD]`

**The concept:**  
Post-Earnings Announcement Drift (PEAD) is one of the most robust and replicated anomalies in academic finance (Ball & Brown 1968, Bernard & Thomas 1989). When a stock beats earnings estimates by more than the implied move, the stock continues to drift in the direction of the surprise for 3–10 trading days. Buying OTM calls (or puts) immediately after an earnings beat (or miss) captures this drift at high leverage before the market fully processes the information.

The key filter: the earnings surprise must exceed the implied move. A 5% beat on a stock that moves 3% implied is not a surprise; a 5% beat on a stock that moves 3% implied with 2% historical beat rate is.

**What to build:**
- In `src/data_fetching.py`, add `get_earnings_surprise(symbol) -> dict`:
  - Fetches last reported EPS actual vs estimate from yfinance `Ticker.earnings_history`
  - Computes `surprise_pct = (actual - estimate) / abs(estimate)`
  - Computes `surprise_vs_implied = surprise_pct / implied_move_pct` (normalised surprise)
- In `src/options_screener.py`, add `pead_score`:
  ```
  If earnings were within last 3 trading days:
    pead_score = clip(surprise_vs_implied / 2.0, 0, 1)   # normalize: 2× implied = max score
    Directional: positive surprise → call signal, negative surprise → put signal
  Else:
    pead_score = 0.0
  ```
- New column: `pead_score`, `pead_direction` ("call" / "put" / "none")
- In the single-stock report, flag options aligned with PEAD direction with a `DRIFT` tag

**Files:** `src/data_fetching.py`, `src/options_screener.py`, `src/cli_display.py`  
**New columns:** `earnings_surprise_pct`, `surprise_vs_implied`, `pead_score`, `pead_direction`

---

## 3.5 VVIX as Options-on-Options Entry Signal `[BUILD]`

**The concept:**  
VVIX (the VIX of VIX, ticker ^VVIX) measures the expected volatility of VIX itself. When VVIX is high relative to its own history, it means market participants are paying up for VIX options — a sign that the options market itself is uncertain about the near-term vol regime. High VVIX historically precedes periods where buying single-stock options generates outsized returns because:
1. Market makers are hedging aggressively (wider spreads, but also bigger moves)
2. Stock correlations tend to break down (alpha opportunity rather than beta)
3. Realised vol often exceeds IV on individual names even when index IV is "high"

**The mathematics:**  
```
VVIX_percentile = percentile_rank(VVIX_today, VVIX_rolling_252d)

Signal thresholds:
  VVIX_pct > 0.80  →  strong long-option environment (adjust quality score × 1.15)
  VVIX_pct 0.50–0.80  →  neutral
  VVIX_pct < 0.20  →  expensive vol regime, prefer selling (adjust quality score × 0.85 for buyers)
```

**What to build:**
- In `src/data_fetching.py`, add `fetch_vvix() -> dict`:
  - Fetch ^VVIX via yfinance (1-year daily close)
  - Compute rolling percentile rank
  - Return `{"vvix": float, "vvix_percentile": float}`
- In `src/options_screener.py` `get_market_context()`, call `fetch_vvix()` alongside VIX
- Apply a `vvix_multiplier` to `quality_score` for buyer-mode scans:
  - Multiplier scales linearly from 0.85 (VVIX pct 0) to 1.15 (VVIX pct 1.0)
- Print VVIX reading in the market context banner at scan start

**Files:** `src/data_fetching.py`, `src/options_screener.py`, `src/cli_display.py`  
**Config key:** `"vvix_enabled": true, "vvix_buyer_boost_max": 0.15`

---

## 3.6 IV Skew Percentile as Directional Signal `[PARTIAL]`

**The concept:**  
Put-call skew (25-delta risk reversal) measures how much more expensive OTM puts are vs OTM calls. When skew is at an extreme (puts very expensive, calls cheap), it often signals crowded hedging — and when the crowd is already hedged, a market rally can be violent (short squeeze dynamics at the index level). Conversely, calls richly priced vs puts signals excessive optimism.

For single stocks, skew extremes are even more predictive than for indices. A stock with:
- Calls cheap relative to puts (negative skew) AND high short interest = classic squeeze setup
- Puts cheap relative to calls (positive skew) AND downtrend momentum = distribution top

You already compute `iv_skew_rank` but it is used only as a mild component in `skew_combined_score`. The percentile ranking is not separately surfaced as a directional signal.

**What to build:**
- In `src/options_screener.py`, compute `skew_percentile_signal`:
  ```
  skew_rv = iv_skew_rank     (already exists, 0=calls cheap, 1=puts cheap)
  
  call_cheap_signal = 1 - skew_rv    # high when calls are cheap vs puts
  put_cheap_signal  = skew_rv        # high when puts are cheap vs calls
  
  For calls: skew_directional_score = call_cheap_signal × (1.0 + short_interest_norm)
  For puts:  skew_directional_score = put_cheap_signal
  ```
- New column: `skew_directional_score`
- Add to the composite weights with key `"skew_directional"` (distinct from existing `"skew_align"`)
- In single-stock report, add a "Skew Signal" line: `Calls cheap (skew pct: 8th)` or `Puts cheap (skew pct: 92nd)`

**Files:** `src/options_screener.py`, `src/cli_display.py`  
**New column:** `skew_directional_score`  
**Config weight key:** `"skew_directional"` in `composite_weights`

---

## 3.7 Gamma Scalping Efficiency Score `[BUILD]`

**The concept:**  
When you own a long option, you can delta-hedge daily to extract the difference between realised vol and implied vol. The profit per delta-hedge is:

```
Daily P&L from gamma scalp ≈ 0.5 × Gamma × (ΔS)² - Theta × Δt

Breakeven daily move = sqrt(2 × Theta / Gamma)  =  σ_IV × S × sqrt(Δt)

If σ_realised × S × sqrt(Δt) > breakeven_move  →  gamma scalp is profitable

Efficiency ratio = σ_realised / σ_IV  (> 1.0 = you'd profit from scalping)
```

This is the mathematical underpinning of why IV < RV options are worth buying even if the stock might not reach the strike. A stock that moves 2% per day with 25% IV options is profitable to own on a gamma-scalp basis even if it never goes ITM.

**What to build:**
- In `src/utils.py`, add `gamma_scalp_efficiency(gamma, theta, S, hv_realised) -> dict`:
  - Returns `breakeven_daily_move`, `efficiency_ratio`, `daily_theta_cost`, `expected_daily_gamma_pnl`
- In `src/options_screener.py`, compute `gamma_scalp_score`:
  ```
  efficiency = hv_ewma / impliedVolatility   (EWMA vol vs IV, more responsive than 30d HV)
  gamma_scalp_score = sigmoid(efficiency - 1.0, scale=5.0)
  # > 0.5 when stock realises more than implied
  ```
- New columns: `gamma_scalp_efficiency`, `gamma_scalp_score`
- Display in single-stock detail: `Gamma scalp breakeven: ±$X.XX/day · Efficiency: 1.23×`

**Files:** `src/utils.py`, `src/options_screener.py`, `src/cli_display.py`  
**New columns:** `gamma_scalp_efficiency`, `breakeven_daily_move`, `gamma_scalp_score`  
**Config weight key:** `"gamma_scalp"` in `composite_weights`

---

## 3.8 Dark Pool + Options Sweep Informed Flow Detector `[BUILD]`

**The concept:**  
Institutional informed trading — ahead of earnings surprises, M&A announcements, and major product news — shows up predictably in the options market before it appears in the stock price. The two clearest signals:

1. **Options sweep**: a single large order that hits multiple exchanges simultaneously at ask price, urgently accumulating a position. This almost always indicates directional conviction.
2. **OI growth ahead of catalyst**: when open interest in a specific strike/expiration grows 50%+ in a single day with no obvious reason, a large player is positioning.

You already track `Unusual_Whale` (Vol/OI > 1.5 AND volume > 500) and `option_rvol`. The gap is:
- You do not track *speed* of accumulation (single-day OI jump vs gradual build)
- You do not separate buy-side vs sell-side dominance in the flow (premium paid at ask vs bid)
- You do not flag when multiple strikes in the same expiration are bought simultaneously (sweep pattern)

**What to build:**
- In `src/data_fetching.py`, add `detect_sweep_pattern(option_chain_df) -> pd.Series`:
  - For each expiration, check if ≥3 consecutive OTM strikes all have Vol/OI > 2.0 simultaneously → sweep flag
  - `oi_jump_pct`: compare today's OI to prior day's OI snapshot (from `src/oi_snapshot.py`, already exists)
  - `bid_ask_trade_location`: if `lastPrice >= ask × 0.97`, flag as buyer-initiated (urgency signal)
- New columns: `sweep_flag` (bool), `oi_jump_pct` (float), `buyer_initiated` (bool)
- In `src/options_screener.py`, compute `informed_flow_score`:
  ```
  informed_flow_score = 0.40 × sweep_flag
                      + 0.35 × clip(oi_jump_pct / 1.0, 0, 1)   # 100% OI jump = max score
                      + 0.25 × buyer_initiated
  ```
- Flag qualifying contracts in the report with an `INFORMED FLOW` tag

**Files:** `src/data_fetching.py`, `src/oi_snapshot.py`, `src/options_screener.py`, `src/cli_display.py`  
**New columns:** `sweep_flag`, `oi_jump_pct`, `buyer_initiated`, `informed_flow_score`  
**Config weight key:** `"informed_flow"` in `composite_weights` (suggested initial weight: 0.08)

---

## 3.9 Short-Term Reversal Signal for Oversold/Overbought Options `[BUILD]`

**The concept:**  
Academic research (Muravyev & Ni 2020) shows that options on stocks experiencing extreme short-term price reversals (large 1–3 day moves with high volume) have significantly higher returns when the option direction aligns with the expected reversal. The mechanism: market makers delta-hedge aggressively during the initial move, creating mean-reversion pressure. Buying a call on a stock that dropped 5%+ in a single day with elevated put volume often captures a rebound.

```
Reversal signal conditions for CALL buy:
  1. ret_1d < -4%              (stock dropped hard yesterday)
  2. volume > 2× 20d avg vol   (high volume = capitulation not distribution)
  3. put_call_ratio > 1.5       (puts were heavily bought = crowded short)
  4. rsi_14 < 30               (oversold)

Reversal signal conditions for PUT buy:
  1. ret_1d > +4%              (stock ran hard)
  2. volume > 2× 20d avg vol
  3. put_call_ratio < 0.5       (calls crowded)
  4. rsi_14 > 70               (overbought)

reversal_score = number of conditions met / 4
```

**What to build:**
- In `src/options_screener.py`, compute `reversal_score` and `reversal_direction`:
  - All input signals already exist: `ret_5d` (use 1d subset), `volume`, `pcr`, `rsi_14`
  - Add `ret_1d` fetch in `src/data_fetching.py` (1-day return, separate from `ret_5d`)
- Filter: only apply reversal score when `ret_1d` magnitude > 3%
- New column: `reversal_score`, `reversal_direction`
- Add `"reversal"` to `composite_weights`

**Files:** `src/data_fetching.py`, `src/options_screener.py`  
**New columns:** `ret_1d`, `reversal_score`, `reversal_direction`  
**Config weight key:** `"reversal"` in `composite_weights` (suggested: 0.05, condition-gated)

---

## 3.10 Options-Implied Earnings Move vs Historical — Systematic Edge Finder `[PARTIAL]`

**The concept:**  
You already compute `implied_earnings_move` and `hist_earnings_move`. The edge is in stocks where:
- `hist_earnings_move` consistently exceeds `implied_earnings_move` → straddle buying edge
- `implied_earnings_move` consistently exceeds `hist_earnings_move` → straddle selling edge

But the current implementation computes an average across all historical quarters. The better approach uses:
1. **Weighted recency** — recent quarters matter more (EWMA weighting)
2. **Consistency measure** — a stock that beats IV by 5% every quarter is more valuable than one that averages 5% but varies wildly
3. **Regime filtering** — earnings moves that occurred during COVID-like high-VIX periods are less predictive than normal-regime quarters

```
edge_consistency = mean(hist_move - implied_move) / std(hist_move - implied_move)
                   → Sharpe-like ratio of the earnings straddle edge

If edge_consistency > 1.0  →  reliable buyer edge (stock consistently surprises)
If edge_consistency < -1.0 →  reliable seller edge (IV consistently overstates move)
```

**What to build:**
- In `src/data_fetching.py`, modify `get_earnings_history()` to compute:
  - EWMA-weighted hist moves (recent quarters weight × 1.5)
  - `edge_ratio = (hist_move_ewma - implied_move) / implied_move`
  - `edge_consistency = mean_edge / std_edge` (Sharpe of the spread)
  - `implied_move_beat_rate` = fraction of quarters where stock moved more than implied
- New columns: `edge_ratio`, `edge_consistency`, `implied_move_beat_rate`
- In scoring, use `edge_consistency` as a multiplier on `earnings_iv_cheap` signal
- In the earnings scanner (3.3), surface this as the primary ranking signal

**Files:** `src/data_fetching.py`, `src/options_screener.py`  
**New columns:** `edge_ratio`, `edge_consistency`, `implied_move_beat_rate`

---

## 3.11 Butterfly Mispricing — IV Smile Arbitrage `[BUILD]`

**The concept:**  
A butterfly spread (buy ITM, sell 2× ATM, buy OTM) has a theoretical value derived from the local volatility smile. If the market prices a butterfly significantly below its smile-implied theoretical value, it means the market's vol smile is locally too flat — and you can extract the convexity cheaply by buying the butterfly.

For a single-option buyer, the practical implication is simpler: when IV smile is locally flat around a strike but historical realised smile shows convexity, OTM options are cheap relative to ATM options. This gives you a mathematical basis for preferring OTM options when the smile is flat.

```
Butterfly value = C(K-w) - 2×C(K) + C(K+w)   (for width w)

Theoretical butterfly from smile = (σ_K-w + σ_K+w - 2×σ_K) × vega_K × w²

Mispricing = (market_butterfly_value - theoretical_butterfly_value) / theoretical_butterfly_value

If mispricing < -0.10  →  butterfly is cheap = OTM options underpriced vs ATM
```

**What to build:**
- In `src/options_screener.py`, add `compute_butterfly_mispricing(option_chain_df) -> pd.Series`:
  - For each ATM strike, find the ±1 and ±2 strike wings using the available chain
  - Compute market butterfly value from bid/ask midpoints
  - Compute theoretical value from the local vol surface (use the existing SVI surface if available, else linear interpolation of IV smile)
  - New column: `butterfly_mispricing` (-1 to +1, negative = OTM cheap)
- Expose as `butterfly_score` in composite weights for buyer modes
- Show in single-stock report: "IV smile mispricing: OTM options X% cheap/rich vs ATM"

**Files:** `src/options_screener.py`, `src/cli_display.py`  
**New columns:** `butterfly_mispricing`, `butterfly_score`  
**Config weight key:** `"butterfly"` in `composite_weights` (suggested: 0.04)

---

## 3.12 PCR Extreme Reversal as Contrarian Signal `[PARTIAL]`

**The concept:**  
Put/call ratio at extremes is one of the most reliable contrarian signals in options. You already compute `pcr` per expiration but use it only modestly in the scoring. The more powerful version:

- Stock-level PCR at multi-month extreme (not just today's reading): `pcr_percentile_252d`
- When `pcr_percentile > 0.95` (put-heavy) → contrarian call signal (market over-hedged, ready for relief)
- When `pcr_percentile < 0.05` (call-heavy) → contrarian put signal (market complacent)
- Combining PCR extreme with RSI extreme (both oversold) gives a higher-conviction signal

```
pcr_reversal_score_for_calls = pcr_percentile_252d    
  # high PCR = lots of put buying = contrarian bullish

Combined signal:
  pcr_reversal_score = 0.5 × pcr_percentile + 0.3 × (1 - rsi_norm) + 0.2 × volume_surge
  (for calls; invert for puts)
```

**What to build:**
- In `src/data_fetching.py`, fetch 252 days of PCR history per ticker (via yfinance options chain history or compute from OI data)
- Compute `pcr_percentile_252d` (rolling percentile rank of today's PCR)
- In `src/options_screener.py`, compute `pcr_reversal_score` as above
- Separate from existing `pcr_score` which is a flow signal; this is a contrarian reversal signal
- New column: `pcr_percentile_252d`, `pcr_reversal_score`
- Config weight key: `"pcr_reversal"` in `composite_weights`

**Files:** `src/data_fetching.py`, `src/options_screener.py`  
**New columns:** `pcr_percentile_252d`, `pcr_reversal_score`

---

---

# Part 4 — Quick Wins (Small Changes, Real Impact)

## 4.1 Historical win rate on the entry screen `[PARTIAL]` *(~2 hours)*
Display your actual closed-trade win rate per strategy before you enter a new one. One line in `cli_display.py`. Data already exists in SQLite.

## 4.2 Score your own past trades `[BUILD]` *(~1 day)*
Re-score every closed paper trade using the current model and compare predicted quality_score to actual outcome. This immediately shows if the model has predictive power or is decorative.

## 4.3 Minimum IV rank filter for buyer modes `[BUILD]` *(~2 hours)*
Add a config flag `"buyer_min_iv_percentile": 0` that blocks entering long options when IV is above the Nth percentile. Buying options when IV is at a 90th percentile is almost never profitable. A hard filter here prevents a common mistake.

## 4.4 Auto-log threshold display `[BUILD]` *(~1 hour)*
When a scan finishes, print "Auto-log threshold: quality_score ≥ 0.65 · 3 contracts eligible" so you know exactly which picks would be auto-logged and can adjust the threshold before running again.

## 4.5 Spread cost as % of expected move `[BUILD]` *(~2 hours)*
For every option, add column `spread_cost_vs_em = bid_ask_spread / expected_move`. If your entry friction is 15% of the expected move, you need the stock to move significantly just to break even on fills. Flag any option where this ratio > 0.20.

---

---

# Execution Order

| Priority | Section | Estimated effort | Profit impact |
|----------|---------|-----------------|---------------|
| 1 | 1.1 Kelly feedback loop | 2 days | High — direct sizing improvement |
| 2 | 1.2 Portfolio limits at entry | 1 day | High — prevents correlated blowups |
| 3 | 2.1 Pre-scan earnings calendar | 2 days | High — unlocks 3.3 and 3.10 |
| 4 | 3.1 HAR realised vol + VRP² | 2 days | High — better IV cheap/rich signal |
| 5 | 3.2 MJD theoretical value | 1 day | High — catches underpriced jump risk |
| 6 | 3.3 Earnings straddle scanner | 3 days | High — systematic earnings edge |
| 7 | 1.3 Fix backtest signal mismatch | 1 day | Medium — validates live model |
| 8 | 1.4 DTE-tiered exit in backtest | 1 day | Medium — accurate Sharpe reporting |
| 9 | 3.7 Gamma scalp efficiency | 1 day | Medium — better entry filter |
| 10 | 3.6 Skew percentile directional | 1 day | Medium — directional conviction |
| 11 | 3.8 Informed flow detector | 3 days | Medium — catch smart money |
| 12 | 3.4 PEAD drift signal | 2 days | Medium — post-earnings alpha |
| 13 | 3.5 VVIX entry signal | 1 day | Medium — regime-aware timing |
| 14 | 1.5 Regime-aware calibration | 3 days | Medium — better weight tuning |
| 15 | 1.6 Greeks P&L attribution | 2 days | Medium — understand your edge source |
| 16 | 2.2 Win rate in scan output | 0.5 days | Low — awareness |
| 17 | 2.3 Intraday timing filter | 1 day | Low — fill quality |
| 18 | 3.9 Short-term reversal | 2 days | Low-medium — condition-gated |
| 19 | 3.10 Earnings edge consistency | 1 day | Medium — improves 3.3 ranking |
| 20 | 3.11 Butterfly mispricing | 2 days | Low-medium — smile arb |
| 21 | 3.12 PCR extreme reversal | 2 days | Low-medium — contrarian |
| 22 | 2.4 Sector flow aggregator | 3 days | Low — awareness feature |
| 23 | 4.1–4.5 Quick wins | 1 day total | Low individually, good hygiene |

---

*Last updated: 2026-05-14*
