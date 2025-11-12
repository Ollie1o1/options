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

Notes
- Python 3.7+ is required.
- The tool is fully interactive and does not accept CLI flags; it prompts for mode, expirations, DTE, etc.
- There is no configured linter or test suite in this repo.

## High-level architecture

- Single-module application: all logic lives in `options_screener.py`. The program is an interactive CLI that fetches options chains from Yahoo Finance (via `yfinance`), computes analytics, and prints a formatted report. Results can be exported to CSV and logged for future P/L tracking.

- Operating modes (selected by the first prompt):
  - Single-stock: enter a ticker (e.g., AAPL). Quantile-based LOW/MEDIUM/HIGH premium buckets on that one symbol.
  - Budget scan: enter `ALL`. You supply a per-contract budget and a list of tickers. Contracts are filtered to fit the budget and bucketed by percentage of budget.
  - Discovery scan: enter `DISCOVER` (or leave blank). Scans a predefined list of highly traded tickers (you choose how many to scan) with quantile-based buckets and diversified picks.

- Data flow (big picture):
  1) Input & mode selection: `main()` reads interactive inputs (mode, max expirations, DTE bounds, optional budget/tickers).
  2) Data fetch: `fetch_options_yfinance()` calls `yfinance` to pull options chains for each selected ticker and computes 30‑day historical volatility (HV). It also resolves the current underlying price and collects expirations up to the chosen count.
  3) Risk-free rate: `get_risk_free_rate()` fetches the 13‑week Treasury (^IRX) via `yfinance`; falls back to 4.5% if unavailable.
  4) Enrichment & scoring: `enrich_and_score()` standardizes and filters contracts by DTE, computes premiums (mid or last), liquidity, spread percentage, Black‑Scholes delta, and advanced analytics:
     - Probability of Profit, Expected Move, Probability of Touch
     - Risk/Reward metrics (max loss, break-even, R/R)
     - IV vs HV and IV skew (put vs call at same strike/expiry)
     - Composite `quality_score` (liquidity, IV advantage, R/R, PoP, spread, delta)
  5) Categorization & selection:
     - `categorize_by_premium()` assigns LOW/MEDIUM/HIGH buckets (quantiles for single/discovery, budget‑based for budget mode).
     - `pick_top_per_bucket()` selects the top N per bucket, optionally diversifying across tickers for multi‑ticker modes.
  6) Reporting & output:
     - `print_report()` renders the main report (per bucket listings with summary stats and rationale).
     - A “Top Overall Pick” is computed using a weighted “overall_score” derived from quality, liquidity, spread, delta, and IV quality.
     - Optional outputs: `export_to_csv()` to `exports/options_picks_<mode>_<timestamp>.csv` and `log_trade_entry()` to `trades_log/entries.csv` (directories auto‑created).

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
