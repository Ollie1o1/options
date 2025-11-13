# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Commands

- Install dependencies
  ```bash
  pip install -r requirements.txt
  ```

- Run the screener (interactive CLI)
  ```bash
  python options_screener.py
  ```

- Close expired trades and calculate realized P/L
  ```bash
  python options_screener.py --close-trades
  ```

- Run backtesting engine on historical logs
  ```bash
  python backtest_screener.py
  ```

Notes
- Python 3.7+ is required.
- The main screener is interactive and prompts for mode, expirations, DTE, etc.
- The `--close-trades` flag allows updating the trade log with exit prices and P/L.
- There is no configured linter or test suite in this repo.

## High-level architecture

- Multi-module application:
  - **`options_screener.py`**: Main interactive CLI that fetches options chains from Yahoo Finance (via `yfinance`), computes analytics, and prints a formatted report. Results can be exported to CSV and logged for future P/L tracking. Now includes adaptive VIX-based scoring, IV Rank/Percentile, earnings awareness, Monte Carlo simulations, and strike filtering.
  - **`config.json`**: Configuration file for scoring weights, VIX regime thresholds, and parameters.
  - **`simulation.py`**: Monte Carlo simulation module using Geometric Brownian Motion for probability calculations.
  - **`visualize_results.py`**: Visualization module for generating IV/HV scatter plots, risk/reward histograms, and expected move charts.
  - **`backtest_screener.py`**: Backtesting engine that loads historical JSONL logs, fetches expiration prices, calculates realized P/L, and outputs performance metrics.

- Operating modes (selected by the first prompt):
  - Single-stock: enter a ticker (e.g., AAPL). Quantile-based LOW/MEDIUM/HIGH premium buckets on that one symbol.
  - Budget scan: enter `ALL`. You supply a per-contract budget and a list of tickers. Contracts are filtered to fit the budget and bucketed by percentage of budget.
  - Discovery scan: enter `DISCOVER` (or leave blank). Scans a predefined list of highly traded tickers (you choose how many to scan) with quantile-based buckets and diversified picks.

- Data flow (big picture):
  1) Configuration & market context:
     - `load_config()` reads `config.json` for scoring weights and parameters.
     - `get_vix_level()` fetches current VIX to determine volatility regime.
     - `determine_vix_regime()` selects adaptive weights (low/normal/high vol regimes).
  2) Input & mode selection: `main()` reads interactive inputs (mode, max expirations, DTE bounds, optional budget/tickers).
  3) Data fetch: `fetch_options_yfinance()` calls `yfinance` to pull options chains, computes 30‑day historical volatility (HV), calculates IV Rank/Percentile, and fetches next earnings date.
  4) Risk-free rate: `get_risk_free_rate()` fetches the 13‑week Treasury (^IRX) via `yfinance`; falls back to 4.5% if unavailable.
  5) Enrichment & scoring: `enrich_and_score()` applies:
     - Strike filtering (±15% moneyness band)
     - Standard metrics: Probability of Profit, Expected Move, Probability of Touch, Risk/Reward
     - Monte Carlo simulation: `monte_carlo_pop()` provides simulation-based PoP and PoT
     - IV Rank/Percentile integration
     - Earnings awareness: flags contracts expiring within 5 days of earnings
     - Adaptive quality scoring using VIX-based weights with earnings penalty and IV Rank bonus
  6) Categorization & selection:
     - `categorize_by_premium()` assigns LOW/MEDIUM/HIGH buckets (quantiles for single/discovery, budget‑based for budget mode).
     - `pick_top_per_bucket()` selects the top N per bucket, optionally diversifying across tickers for multi‑ticker modes.
  7) Reporting & output:
     - `print_report()` renders the main report with summary stats and rationale.
     - A "Top Overall Pick" is computed using weighted scores.
     - Optional outputs: 
       - `export_to_csv()` to `exports/` with all metrics including IV Rank, Monte Carlo results, and event flags
       - `create_visualizations()` generates charts in `reports/`
       - `log_trade_entry()` to `trades_log/entries.csv` with exit tracking fields
  8) Trade closing: `python options_screener.py --close-trades` updates trade log with realized P/L for expired positions.
  9) Backtesting: `backtest_screener.py` analyzes historical performance, calculates metric correlations, and generates reports.

- External dependencies and assumptions:
  - `yfinance` provides all market data (options chains, prices, and ^IRX); network access is required and some data may be delayed or incomplete.
  - `pandas` is used for data shaping and ranking.

- Key extension points (within `options_screener.py`):
  - Scoring weights and target delta inside `enrich_and_score()`.
  - Number of picks per bucket in `pick_top_per_bucket()`.
  - Discovery-mode ticker universe and budget-mode default tickers in `main()`.

## Important README highlights

- Quick start: clone, `pip install -r requirements.txt`, then `python options_screener.py`.
- Three modes (Single, Budget, Discovery) with interactive prompts and sensible defaults.
- CSV export path pattern: `exports/options_picks_<Mode>_<timestamp>.csv`.
- Trade logging path: `trades_log/entries.csv`.
- The tool provides additional analytics such as Probability of Profit, Expected Move, Probability of Touch, Risk/Reward, IV vs HV, and IV Skew, and uses a composite quality scoring system to rank candidates.
