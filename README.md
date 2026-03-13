# Options Screener — Professional Edition

A Python-based options screening tool that identifies high-probability trading opportunities through advanced analytics, institutional-level metrics, and dynamic safety filters. Includes an AI-powered scoring layer that enriches top candidates with qualitative analysis and produces a final weighted ranked list.

---

## Table of Contents

1. [Core Philosophy](#core-philosophy)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [The Technical Screener](#the-technical-screener)
   - [Scan Modes](#scan-modes)
   - [CLI Reference](#cli-reference)
   - [Terminal Output](#terminal-output)
5. [The AI Ranking Layer](#the-ai-ranking-layer)
   - [Setup](#setup)
   - [Usage](#usage)
   - [CLI Reference (ai_rank.py)](#cli-reference-ai_rankpy)
   - [How It Works](#how-it-works)
6. [Analytics Engine](#analytics-engine)
7. [Paper Trading](#paper-trading)
8. [Configuration](#configuration)
9. [Project Structure](#project-structure)
10. [Roadmap](#roadmap)

---

## Core Philosophy

A high-quality trade occurs when multiple independent factors align simultaneously:

1. **Action** — The market is interested: high volume, tight spreads, unusual flow.
2. **Edge** — The probabilities favour you: IV vs. HV, seasonality, risk/reward.
3. **Structure** — The trade is not fighting a technical barrier: support/resistance, OI walls.
4. **Trend** — The trade aligns with the stock's momentum and broader market context.

The screener finds opportunities where all four forces converge. Consistency comes from saying "no" to good volume on a bad chart.

---

## Installation

**Prerequisites:** Python 3.9+, pip

```bash
git clone https://github.com/Ollie1o1/options.git
cd options
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The screener works out of the box — no API keys needed for the technical screener (it uses Yahoo Finance). The AI layer requires an Anthropic API key (see [Setup](#setup) below).

---

## Quick Start

```bash
# Interactive technical screener (CLI)
python -m src.options_screener

# Streamlit web dashboard
python -m src.options_screener --ui

# AI-enhanced ranked output for specific tickers
python ai_rank.py AAPL TSLA NVDA

# AI ranking with no API key (technical scores only)
python ai_rank.py AAPL --no-ai
```

---

## The Technical Screener

### Scan Modes

Launch the screener with `python -m src.options_screener` and enter one of these commands at the prompt:

| Command | Mode | What it does |
|---------|------|--------------|
| `AAPL` (any ticker) | Single-stock | Deep analysis of one symbol across all expiries |
| `ALL` | Budget scan | Multi-ticker scan filtered to a per-contract dollar budget |
| `DISCOVER` | Discovery | Scans the top 100 most-liquid tickers, no budget limit |
| `SELL` | Premium Selling | Short put candidates ranked by return-on-risk |
| `SPREADS` | Credit Spreads | Bull Put and Bear Call spread opportunities |
| `IRON` | Iron Condors | Delta-neutral range-bound strategies |
| `PORTFOLIO` | Portfolio | View P/L on all open paper trades |

### CLI Reference

```
python -m src.options_screener [OPTIONS]

Options:
  --no-color      Disable ANSI color output (useful for piping to a log file)
  --close-trades  Update the trade log with closing prices and realised P/L
  --ui            Launch the Streamlit web dashboard
  -h, --help      Show help and exit
  --version       Show version string and exit
```

### Terminal Output

The CLI renders a fully colour-coded, responsive interface that adapts to your terminal width (60–120 chars):

**Startup**
- Double-box banner with current date/time
- Colour-coded mode menu
- Market context ticker: trend (Bull/Bear), VIX regime, macro risk flag, 10Y yield change

**Scan progress**
- Single-line tqdm progress bar — no noise from third-party libraries
- Clean per-ticker summary after completion: `✓ AAPL  12 contract(s)`

**Report — per pick**
```
  🐋 CALL   262.50  2026-03-20   $4.03   31.0%    1494    1095   +0.38   OTM  ★★★☆☆
    ↳ Mechanics: Vol: 1095 OI: 1494 | Spread: 3.7% | Delta: +0.38 | Greeks: Γ … | Cost: $402.50
    ↳ Analysis:  IV: 31.0% (below median) | PoP: 55.2% | RR: 1.4x | EV: $18 | Sentiment: Bullish
    ↳ Thesis:    High probability (>65%) • Trend aligned | ⚠ Wide spread - use limits
    ↳ Entry:     ≤$3.95  |  Target: $5.93 (+50%)  |  Stop: $2.96 (-25%)
         Breakeven: $266.45  |  Max Loss: $395  |  Confidence: HIGH (87%)
         Risks: High time decay - monitor closely
```

**Executive Summary** — printed after all picks, showing top 3 opportunities with entry levels and portfolio warnings (concentration, correlation, earnings risk, negative EV).

---

## The AI Ranking Layer

`ai_rank.py` runs the technical screener then sends the candidates to the Claude API for qualitative analysis. AI scores are combined with technical scores to produce a single ranked list.

**What the AI evaluates for each candidate:**
- Whether current IV is justified given upcoming catalysts
- Trend and momentum alignment with the trade direction
- Earnings / macro event risk relative to expiry
- Overall risk/reward quality
- Returns a score 0–100 plus a 1–2 sentence rationale and risk flags

### Setup

1. Dependencies are already in `requirements.txt`. If installing manually:
   ```bash
   pip install anthropic rich
   ```

2. Set your Anthropic API key as an environment variable:
   ```bash
   # macOS / Linux
   export ANTHROPIC_API_KEY=sk-ant-...

   # Windows CMD
   set ANTHROPIC_API_KEY=sk-ant-...

   # Windows PowerShell
   $env:ANTHROPIC_API_KEY="sk-ant-..."
   ```
   > The key is only needed when running `ai_rank.py` without `--no-ai`.

### Usage

```bash
# Score and rank candidates — basic usage
python ai_rank.py AAPL
python ai_rank.py AAPL TSLA NVDA

# Change screener mode
python ai_rank.py --mode "Premium Selling" SPY QQQ
python ai_rank.py --mode "Discovery scan"

# Control DTE window
python ai_rank.py --dte-min 14 --dte-max 45 AAPL

# Set a per-contract budget cap (Budget scan mode)
python ai_rank.py --mode "Budget scan" --budget 500 AAPL MSFT GOOG

# Change trader profile (adjusts scoring weights)
python ai_rank.py --profile position SPY   # swing | day | position

# Adjust how much the AI score matters
python ai_rank.py --ai-weight 0.5 AAPL    # 50% AI, 50% technical

# Skip AI entirely — rank by technical quality_score only (no API key needed)
python ai_rank.py --no-ai AAPL TSLA

# Show top 10 instead of default 20
python ai_rank.py --top 10 AAPL

# Save ranked results to a file
python ai_rank.py AAPL --output ranked.csv
python ai_rank.py AAPL --json ranked.json

# Suppress the screener's verbose progress output
python ai_rank.py AAPL --quiet

# Enable debug logging for the AI scorer
python ai_rank.py AAPL --verbose

# Full help
python ai_rank.py --help
```

### CLI Reference (ai_rank.py)

```
python ai_rank.py [TICKERS ...] [OPTIONS]

Positional:
  TICKERS             One or more ticker symbols (e.g. AAPL TSLA SPY)

Screener:
  --mode MODE         Screener mode. Choices:
                        Single-stock (default), Budget scan, Discovery scan,
                        Premium Selling, Credit Spreads, Iron Condor
  --dte-min DAYS      Minimum days to expiration (default: 7)
  --dte-max DAYS      Maximum days to expiration (default: 45)
  --budget $          Per-contract budget cap (Budget scan mode only)
  --expiries N        Max expiration dates fetched per ticker (default: 4)
  --profile PROFILE   Trader profile: swing (default) | day | position

AI scoring:
  --no-ai             Skip AI — rank by technical quality_score only
  --ai-weight W       AI contribution to final_score, 0–1 (default: 0.30)
  --top N             Candidates to display in ranked table (default: 20)

Output:
  --output FILE       Save ranked results to CSV
  --json FILE         Save ranked results to JSON
  --quiet             Suppress screener verbose output
  --verbose           Enable debug logging for the AI scorer
```

### How It Works

```
Tickers
   │
   ▼
[Technical Screener]  ─────────────────────────────────────────────────────
   │  src/options_screener.py                                              │
   │  • Fetches 1-year daily history (single fetch per ticker)             │
   │  • Calculates Greeks, PoP, EV, IV rank, HV, momentum, warnings       │
   │  • Scores each contract → quality_score [0, 1]                       │
   ▼                                                                       │
[AI Scorer]           ─────────────────────────────────────────────────────
   │  src/ai_scorer.py                                                     │
   │  • Extracts key fields (IV, HV, delta, PoP, earnings, warnings…)     │
   │  • Batches candidates (default 5 per API call)                        │
   │  • Sends structured prompt to Claude API                              │
   │  • Receives: ai_score (0–100), reasoning, flags, catalyst_risk        │
   ▼                                                                       │
[Ranking]             ─────────────────────────────────────────────────────
   │  src/ranking.py
   │  final_score = 0.70 × quality_score + 0.30 × (ai_score / 100)
   │  (weights configurable in src/config_ai.py or via --ai-weight)
   ▼
[Ranked Table]
   • Colour-coded rich table: Rank | Symbol | Type | Strike | Expiry | DTE
   •   Premium | IV% | PoP% | Tech | AI | Final | Catalyst | Flags
   • Falls back to plain text if rich is not installed
```

---

## Analytics Engine

### Scoring

Each contract is scored across 13+ weighted factors:

| Factor | What it measures |
|--------|-----------------|
| Probability of Profit | Blended Black-Scholes + Monte Carlo PoP |
| Expected Move Realism | How achievable the breakeven is vs. the 1σ expected move |
| Risk/Reward | Payoff at 0.75× EM target vs. premium paid |
| Momentum | RSI, 5-day return, ATR trend |
| IV Rank | IV percentile vs. 30-day range; buyers rewarded for low IV |
| IV vs. HV Edge | HV-adjusted EV — positive = options cheap vs. realised vol |
| Liquidity | Volume + open interest, rank-normalised |
| Catalyst | Earnings proximity bonus/penalty |
| Theta Efficiency | Time decay pressure relative to delta |
| Skew Alignment | Put/call IV skew directional bias |
| Gamma/Theta Ratio | Explosive payoff potential per unit of daily bleed |
| Trader Profile | Liquidity-weighted for day traders; DTE-weighted for swing |
| Expected Value | HV-adjusted BS value minus market price minus spread cost |

Scores are further adjusted for trend alignment (+0.15), decay risk (−0.20), gamma squeeze setup (+0.25), OI wall (−0.10), macro risk (−0.10), and seasonal win rate.

**VIX regime adaptation** — scoring weights shift automatically based on the current VIX level (low / normal / high), emphasising different factors in different volatility environments.

### Greeks

All Greeks are computed analytically via Black-Scholes, vectorised with NumPy across the full chain in a single batch: delta, gamma, vega, theta, rho, charm, vanna.

### Monte Carlo

Probability of Profit is blended: **60% Monte Carlo** (Merton Jump Diffusion GBM, captures path-dependency and tail risk) + **40% analytical PoP**, using `numpy.random.default_rng` for reproducible, thread-safe results.

### Trade Plan Generation

For each pick the screener generates:
- **Thesis** — plain-English explanation of why the trade ranks highly
- **Entry price** — bid-to-mid improvement based on spread width
- **Profit target and stop loss** — from `exit_rules` in `config.json`
- **Breakeven and max loss**
- **Confidence score** — penalises wide spreads, low liquidity, short DTE; rewards unusual flow
- **Risk list** — flags wide spreads, negative EV, earnings risk, low liquidity, OI walls, macro

---

## Paper Trading

Log any pick directly from the CLI and track it going forward:

- Positions auto-update on every launch (fetches live quotes via yfinance)
- Take Profit and Stop Loss thresholds enforced from `config.json`
- Win rate, total P/L, and average return tracked in a local SQLite database
- Close expired positions with `python -m src.options_screener --close-trades`

---

## Configuration

### config.json — Screener settings

```json
{
  "filters": {
    "min_volume": 50,
    "min_open_interest": 10,
    "max_bid_ask_spread_pct": 0.40,
    "delta_min": 0.15,
    "delta_max": 0.35,
    "min_days_to_expiration": 7,
    "max_days_to_expiration": 45,
    "min_iv_percentile": 20
  },
  "exit_rules": {
    "take_profit": 0.50,
    "stop_loss": -0.25,
    "time_exit_dte": 21
  },
  "composite_weights": {
    "pop": 0.22,
    "em_realism": 0.10,
    "rr": 0.15,
    "momentum": 0.08,
    "iv_rank": 0.06,
    "liquidity": 0.12,
    "ev": 0.14
  },
  "vix_regimes": {
    "low":    { "threshold": 15,  "weights": { "..." : "..." } },
    "normal": { "threshold_min": 15, "threshold_max": 25, "weights": { "...": "..." } },
    "high":   { "threshold": 25,  "weights": { "...": "..." } }
  }
}
```

### src/config_ai.py — AI layer settings

```python
AI_CONFIG = {
    "provider":        "anthropic",          # API provider
    "model":           "claude-sonnet-4-6",  # Model used for scoring
    "api_key_env":     "ANTHROPIC_API_KEY",  # Env var that holds your key
    "ai_weight":       0.30,   # AI score share of final_score
    "technical_weight": 0.70,  # Screener quality_score share
    "batch_size":      5,      # Candidates per API call (cost control)
    "max_tokens":      2048,
    "temperature":     0.1,    # Lower = more consistent scoring
    "timeout":         60,
    "fields_to_include": [...] # Fields forwarded to the AI prompt
}
```

Key levers:

| Setting | Effect |
|---------|--------|
| `ai_weight` / `technical_weight` | Shift how much the AI opinion matters vs. the quant model |
| `batch_size` | Smaller = more API calls but easier to debug; larger = cheaper |
| `model` | Swap to a faster/cheaper/smarter model at any time |
| `fields_to_include` | Trim to reduce token cost; expand for richer AI context |

---

## Project Structure

```
options/
├── ai_rank.py                # AI-enhanced entry point — runs screener + AI scoring
├── options_screener.py       # Thin wrapper for backward compatibility
├── config.json               # All screener thresholds, weights, and exit rules
├── requirements.txt
├── README.md
└── src/
    ├── options_screener.py   # Core scan engine, report printing, spread finders
    ├── data_fetching.py      # yfinance single-fetch, technical indicators, caching
    ├── filters.py            # Chain filtering, premium bucketing, top-N selection
    ├── scoring.py            # Composite quality score re-exports
    ├── utils.py              # Vectorised Black-Scholes Greeks (NumPy/SciPy)
    ├── simulation.py         # Monte Carlo PoP / PoT (Merton Jump Diffusion GBM)
    ├── formatting.py         # ANSI colours, box drawing, metric formatters
    ├── trade_analysis.py     # Thesis generation, entry/exit levels, confidence
    ├── paper_manager.py      # SQLite paper trade logging and position tracking
    ├── dashboard.py          # Streamlit web interface
    ├── vol_analytics.py      # Volatility cone, IV surface, regime classification
    ├── backtester.py         # Walk-forward backtester with realistic costs
    ├── stress_test.py        # Scenario P/L analysis
    ├── regime_dashboard.py   # Market regime visualisation
    ├── ai_scorer.py          # AIScorer — batches candidates to Claude API
    ├── ranking.py            # combine_scores(), print_ranked_table(), to_csv/json()
    └── config_ai.py          # AI layer configuration constants
```

---

## Roadmap

- [x] Vectorised Black-Scholes Greeks engine
- [x] Monte Carlo PoP blending (Merton Jump Diffusion)
- [x] HV-adjusted expected value
- [x] Paper trading with automated exit tracking
- [x] Streamlit dashboard
- [x] Full colour CLI — responsive width, trade plan per pick, executive summary
- [x] Credit spread and iron condor screeners
- [x] Volatility analytics — cone, IV surface, regime dashboard
- [x] Walk-forward backtester with realistic slippage and commissions
- [x] AI scoring and ranking layer
- [ ] Real-time alerts (email / SMS)
- [ ] Multi-leg spread support in paper manager
- [ ] Backtesting UI improvements
- [ ] News API integration for live catalyst detection

---

## Disclaimer

For educational and informational purposes only. Not financial advice. Options trading involves substantial risk of loss. Always do your own research.

## License

Personal use only. Not licensed for commercial distribution.
