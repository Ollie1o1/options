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
   - [Choosing a Model](#choosing-a-model)
   - [Setup](#setup)
   - [Usage](#usage)
   - [CLI Reference (ai_rank.py)](#cli-reference-ai_rankpy)
   - [How It Works](#how-it-works)
   - [Output Columns Explained](#output-columns-explained)
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

The technical screener works out of the box — no API keys needed (uses Yahoo Finance). The AI layer requires an OpenRouter or Anthropic API key (see [Setup](#setup)).

---

## Quick Start

```bash
# Interactive technical screener (CLI)
python -m src.options_screener

# Streamlit web dashboard
streamlit run src/dashboard.py

# AI-enhanced ranked output for specific tickers
python ai_rank.py AAPL TSLA NVDA

# AI ranking with full reasoning shown per pick
python ai_rank.py AAPL --detail

# Skip AI entirely — technical scores only, no API key needed
python ai_rank.py AAPL --no-ai
```

---

## The Technical Screener

### Scan Modes

Launch with `python -m src.options_screener` and enter one of these commands at the prompt:

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
  CALL   262.50  2026-03-20   $4.03   31.0%    1494    1095   +0.38   OTM  ★★★☆☆
    ↳ Mechanics: Vol: 1095 OI: 1494 | Spread: 3.7% | Delta: +0.38 | Greeks: Γ … | Cost: $402.50
    ↳ Analysis:  IV: 31.0% (below median) | PoP: 55.2% | RR: 1.4x | EV: $18 | Sentiment: Bullish
    ↳ Thesis:    High probability (>65%) • Trend aligned | Wide spread - use limits
    ↳ Entry:     <=3.95  |  Target: $5.93 (+50%)  |  Stop: $2.96 (-25%)
         Breakeven: $266.45  |  Max Loss: $395  |  Confidence: HIGH (87%)
         Risks: High time decay - monitor closely
```

**Executive Summary** — printed after all picks, showing top 3 opportunities with entry levels and portfolio warnings (concentration, correlation, earnings risk, negative EV).

---

## The AI Ranking Layer

`ai_rank.py` runs the technical screener then sends candidates through a two-pass AI analysis. The AI enriches each candidate with qualitative context that the quant model can't see — news sentiment, catalyst proximity, IV justification, and directional bias — then produces a combined final score.

**Two-pass architecture:**
1. **Pass 1 — Ticker context:** For each symbol, the AI reads IV regime, earnings date, implied earnings move vs. historical average, term structure, news sentiment, short interest, and recent headlines. It returns a regime summary and catalyst risk rating.
2. **Pass 2 — Contract scoring:** Each contract is scored 0–100 with a 1–2 sentence rationale. The ticker summary from Pass 1 is injected into each contract's prompt, so the AI knows the broader context when evaluating individual strikes and expiries.

**What the AI scores:**
- Whether IV is justified given upcoming catalysts
- Trend and momentum alignment with the trade direction
- Earnings / macro event risk relative to expiry
- Risk/reward quality beyond what the raw numbers show
- Returns `ai_score` (0–100), `ai_confidence` (0–10), `catalyst_risk` (low/medium/high), and reasoning

**Same-day cache:** Scores are cached in a local SQLite database (`.ai_score_cache.db`) keyed on symbol, strike, expiry, and IV regime. Re-running on the same tickers within the same trading day skips the API for already-scored contracts.

### Choosing a Model

The AI layer uses [OpenRouter](https://openrouter.ai) by default, which gives you access to dozens of models through a single API key. You can also point it directly at the Anthropic API.

#### Free tier (OpenRouter)

These models are free with an OpenRouter account and no credit card required:

| Model | ID for config_ai.py | Quality | Notes |
|-------|---------------------|---------|-------|
| **Llama 3.3 70B** (default) | `meta-llama/llama-3.3-70b-instruct:free` | Good | Best free option; rate-limited by upstream providers |
| Gemma 3 12B (fallback) | `google/gemma-3-12b-it:free` | Decent | Used automatically after 2 Llama failures |
| DeepSeek R1 (free) | `deepseek/deepseek-r1:free` | Very good | Slower; strong reasoning |
| Mistral 7B | `mistralai/mistral-7b-instruct:free` | Basic | Fast but weaker at nuanced analysis |

> The free tier is rate-limited. If you hit 429 errors, the screener retries automatically with exponential backoff and switches to the fallback model. For production use, add a small OpenRouter credit balance to lift rate limits.

#### Paid models (recommended for serious use)

Adding credit to your OpenRouter account ($5–10 goes a long way) unlocks faster, higher-quality models:

| Model | ID for config_ai.py | Cost | Notes |
|-------|---------------------|------|-------|
| **Claude Sonnet 4.6** | `anthropic/claude-sonnet-4-6` | ~$3/M tok | Best reasoning on options context |
| GPT-4o | `openai/gpt-4o` | ~$5/M tok | Strong; good at structured JSON output |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4-5` | ~$0.80/M tok | Fast and cheap; good enough for screening |
| Llama 3.3 70B (paid) | `meta-llama/llama-3.3-70b-instruct` | ~$0.59/M tok | Same model, no rate limits |
| DeepSeek V3 | `deepseek/deepseek-chat` | ~$0.27/M tok | Excellent value; strong analyst-style output |

#### Direct Anthropic API

If you prefer to use your Anthropic API key directly (bypassing OpenRouter):

```python
# in src/config_ai.py
AI_CONFIG = {
    "provider": "anthropic",
    "model": "claude-sonnet-4-6",
    "api_key_env": "ANTHROPIC_API_KEY",
    ...
}
```

To change model at any time, edit the `model` field in `src/config_ai.py`. The `fallback_model` field is used automatically after 2 failed retries.

### Setup

1. Create a free account at [openrouter.ai](https://openrouter.ai) and copy your API key from the dashboard.

2. Create a `.env` file in the project root (it is gitignored — never committed):
   ```
   OPENROUTER_API_KEY=sk-or-v1-your-key-here
   ```
   See `.env.example` for the template.

3. That's it. The screener loads the key automatically via `python-dotenv`.

> If you prefer Anthropic directly, set `ANTHROPIC_API_KEY=sk-ant-...` in `.env` and update `provider` + `api_key_env` in `src/config_ai.py`.

### Usage

```bash
# Score and rank candidates
python ai_rank.py AAPL
python ai_rank.py AAPL TSLA NVDA

# Show full AI reasoning for each pick
python ai_rank.py AAPL --detail

# Run AI portfolio coherence check on top 5 picks
python ai_rank.py AAPL MSFT TSLA --portfolio-check

# Change screener mode
python ai_rank.py --mode "Premium Selling" SPY QQQ
python ai_rank.py --mode "Credit Spreads" AAPL MSFT

# Control DTE window
python ai_rank.py --dte-min 14 --dte-max 45 AAPL

# Set a per-contract budget cap (Budget scan mode)
python ai_rank.py --mode "Budget scan" --budget 500 AAPL MSFT GOOG

# Change trader profile (adjusts scoring weights)
python ai_rank.py --profile position SPY   # swing | day | position

# Adjust how much the AI score matters (default: 0.30)
python ai_rank.py --ai-weight 0.5 AAPL    # 50% AI, 50% technical

# Skip AI entirely — rank by technical quality_score only (no API key needed)
python ai_rank.py --no-ai AAPL TSLA

# Show top 10 instead of default 20
python ai_rank.py --top 10 AAPL

# Save ranked results to a file
python ai_rank.py AAPL --output ranked.csv
python ai_rank.py AAPL --json ranked.json

# Quiet mode (suppress screener progress output)
python ai_rank.py AAPL --quiet

# Debug logging for the AI scorer
python ai_rank.py AAPL --verbose
```

### CLI Reference (ai_rank.py)

```
python ai_rank.py [TICKERS ...] [OPTIONS]

Positional:
  TICKERS              One or more ticker symbols (e.g. AAPL TSLA SPY)

Screener:
  --mode MODE          Screener mode. Choices:
                         Single-stock (default), Budget scan, Discovery scan,
                         Premium Selling, Credit Spreads, Iron Condor
  --dte-min DAYS       Minimum days to expiration (default: 7)
  --dte-max DAYS       Maximum days to expiration (default: 45)
  --budget $           Per-contract budget cap (Budget scan mode only)
  --expiries N         Max expiration dates fetched per ticker (default: 8)
  --profile PROFILE    Trader profile: swing (default) | day | position

AI scoring:
  --no-ai              Skip AI — rank by technical quality_score only
  --ai-weight W        AI contribution to final_score, 0–1 (default: 0.30)
  --top N              Candidates to display in ranked table (default: 20)
  --detail             Show full AI reasoning for each pick
  --portfolio-check    Run AI portfolio coherence check on top 5 picks

Output:
  --output FILE        Save ranked results to CSV
  --json FILE          Save ranked results to JSON
  --quiet              Suppress screener verbose output
  --verbose            Enable debug logging for the AI scorer
```

### How It Works

```
Tickers
   │
   ▼
[Technical Screener]
   │  src/options_screener.py
   │  - Fetches 1-year daily history (single fetch per ticker)
   │  - Calculates Greeks, PoP, EV, IV rank, HV, momentum, warnings
   │  - Scores each contract -> quality_score [0, 1]
   │
   ▼
[Pass 1 — Ticker Context]
   │  src/ai_scorer.py
   │  - Per symbol: IV regime, earnings date, implied vs. historical move,
   │    term structure, news sentiment, short interest, headlines
   │  - Returns: regime, catalyst_risk, directional_bias, summary
   │
   ▼
[Pass 2 — Contract Scoring]
   │  src/ai_scorer.py
   │  - Narrative context built per contract (IV rank narrative, seller/buyer
   │    edge, PoP strength, RR quality, theta burn, score drivers, warnings)
   │  - Ticker summary from Pass 1 prepended to each contract prompt
   │  - Batches candidates (default 5 per API call)
   │  - Returns: ai_score (0-100), ai_confidence (0-10), reasoning, flags,
   │             catalyst_risk, iv_justified
   │  - Results cached in .ai_score_cache.db for the trading day
   │
   ▼
[Ranking]
   │  src/ranking.py
   │  final_score = tw * quality_score + aw * (ai_score / 100)
   │
   │  ai_weight (aw) is dynamic per row:
   │    base_aw (0.30) x VIX regime mult x AI confidence mult x liquidity adj
   │    capped at 0.55
   │
   │  Divergence flagged when |ai_score/100 - quality_score| > 0.20
   │
   ▼
[Ranked Table]
   - Colour-coded rich table (180-char wide):
     Rank | Symbol | Type | Strike | Expiry | DTE | Premium | IV%
     PoP% | Tech | AI | Conf | Final | Cat | Div | Flags/Reasoning
   - Divergence summary block below table for flagged picks
   - --detail expands full reasoning per row
   - Falls back to plain text if rich is not installed
```

### Output Columns Explained

| Column | Description |
|--------|-------------|
| **Tech** | Technical quality score × 100 (0–100). Pure quant — Greeks, PoP, EV, liquidity, momentum |
| **AI** | AI score (0–100). Qualitative — catalyst, IV justification, narrative context |
| **Conf** | AI confidence in its own score (0–10). Higher = more AI weight applied |
| **Final** | Weighted blend: `tw × Tech/100 + aw × AI/100`. This is the ranking signal |
| **Cat** | Catalyst risk: LOW / MED / HIGH. Earnings or macro events near expiry |
| **Div** | Divergence: `AI>TECH` (AI bullish vs quant) or `TECH>AI` (quant bullish vs AI). Highlighted in cyan/magenta |
| **Flags/Reasoning** | Short flags in normal mode; full 1–2 sentence AI rationale with `--detail` |

**Divergence** is the most actionable signal — when the two models strongly disagree, it's worth reading the reasoning before trading.

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

**VIX regime adaptation** — scoring weights shift automatically based on the current VIX level (low / normal / high), emphasising different factors in different volatility environments. The AI weight also scales with VIX regime (×0.80 in low VIX, ×1.30 in high VIX).

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
  }
}
```

### src/config_ai.py — AI layer settings

```python
AI_CONFIG = {
    # API provider: "openrouter" (default) or "anthropic"
    "provider":        "openrouter",
    "model":           "meta-llama/llama-3.3-70b-instruct:free",
    "fallback_model":  "google/gemma-3-12b-it:free",   # used after 2 failed retries
    "api_key_env":     "OPENROUTER_API_KEY",

    # Scoring weights
    "ai_weight":        0.30,
    "technical_weight": 0.70,

    # Dynamic weight multipliers by VIX regime
    "regime_weight_multipliers": {
        "low":    0.80,   # quant signals dominate in calm markets
        "normal": 1.00,
        "high":   1.30,   # AI catalyst awareness worth more in volatile markets
    },

    # Feature flags
    "cache_enabled":      True,   # same-day SQLite score cache
    "confidence_enabled": True,   # AI returns ai_confidence 0-10
    "two_pass_enabled":   True,   # ticker-level context before contract scoring
    "news_enabled":       True,   # inject recent headlines into ticker context

    # API call settings
    "batch_size":   5,
    "max_tokens":   2048,
    "temperature":  0.1,
    "timeout":      60,
}
```

Key levers:

| Setting | Effect |
|---------|--------|
| `model` / `fallback_model` | Swap to any OpenRouter model ID at any time |
| `ai_weight` / `technical_weight` | Shift how much AI opinion matters vs. quant model |
| `two_pass_enabled` | Disable to skip ticker context pass (faster, fewer API calls) |
| `batch_size` | Smaller = easier to debug; larger = fewer API calls |
| `cache_enabled` | Disable to force fresh AI scores on every run |
| `fields_to_include` | Trim to reduce token cost; expand for richer AI context |

---

## Project Structure

```
options/
├── ai_rank.py                # AI-enhanced entry point — runs screener + AI scoring
├── options_screener.py       # Thin wrapper for backward compatibility
├── config.json               # Screener thresholds, weights, and exit rules
├── requirements.txt
├── .env                      # Your API keys (gitignored — never committed)
├── .env.example              # Template showing required env vars
├── README.md
└── src/
    ├── options_screener.py   # Core scan engine, report printing, spread finders
    ├── data_fetching.py      # yfinance single-fetch, technical indicators, news headlines
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
    ├── ai_scorer.py          # Two-pass AI scoring with retry, fallback, narrative context
    ├── ai_cache.py           # Same-day SQLite cache for AI scores
    ├── ranking.py            # combine_scores(), print_ranked_table(), divergence flags
    └── config_ai.py          # AI layer configuration (model, weights, thresholds)
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
- [x] AI scoring and ranking layer (two-pass, dynamic weights, divergence detection)
- [x] Same-day AI score cache (SQLite)
- [x] Narrative context enrichment (IV/HV edge, PoP, RR, theta burn, warnings)
- [x] Portfolio coherence check (`--portfolio-check`)
- [ ] Real-time alerts (email / SMS)
- [ ] Multi-leg spread support in paper manager
- [ ] Backtesting UI improvements
- [ ] Live news API integration (replace yfinance headlines)

---

## Disclaimer

For educational and informational purposes only. Not financial advice. Options trading involves substantial risk of loss. Always do your own research.

## License

Personal use only. Not licensed for commercial distribution.
