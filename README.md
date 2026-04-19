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
8. [Weight-Profile Auto-Logging](#weight-profile-auto-logging)
9. [Configuration](#configuration)
10. [API + Discord & Telegram Bots](#api--discord--telegram-bots)
11. [Project Structure](#project-structure)
12. [Roadmap](#roadmap)

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

**Prerequisites:** Python 3.10+, pip (Compatible with Python 3.14+)

```bash
git clone https://github.com/Ollie1o1/options.git
cd options
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

The technical screener works out of the box — no API keys needed (uses Yahoo Finance). The AI layer requires an OpenRouter API key (see [Setup](#setup)).

**Optional — Polygon.io enrichment:** Set `POLYGON_API_KEY` in your `.env` file to enable higher-quality ticker-filtered news, real-time VWAP, and unusual options flow detection. Free API keys are available at [polygon.io/dashboard/signup](https://polygon.io/dashboard/signup). The screener runs identically without this key.

---

## Quick Start

> **Important:** Always run from the project root directory (`cd options`).

### Recommended (works every time)

```bash
# Activate the venv first, then run normally
source venv/bin/activate
python -m src.options_screener
python ai_rank.py AAPL TSLA NVDA
```

### Auto-venv launchers (no activation needed)

If you don't want to activate the venv every time, use these launchers — they detect the venv automatically and re-launch under it:

```bash
# Interactive screener (auto-activates venv)
python3 run.py

# AI-enhanced ranking (auto-activates venv)
python3 ai_rank.py AAPL TSLA NVDA

# Package-style entry point (auto-activates venv)
python3 -m src
```

> **Why does this matter?** The project has ~30 dependencies (pandas, openai, rich, etc.) installed in the `venv/` directory. If you run `python3 -m src.options_screener` without activating the venv first, Python uses your system interpreter which doesn't have these packages — you'll get missing module errors, no colors in the AI table, and broken API calls. The launchers above handle this automatically.

### All commands

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

# Paper trade portfolio viewer
python -m src.check_pnl

# Walk-forward backtest
python -m src.backtester AAPL SPY NVDA
```

> Commands in this section assume the venv is activated. If not, either activate it (`source venv/bin/activate`) or use the auto-venv launchers above.

---

## The Technical Screener

### Scan Modes

Launch with `python -m src.options_screener` (venv active) or `python3 run.py` (auto-venv) and enter one of these commands at the prompt:

| Command | Mode | What it does |
|---------|------|--------------|
| `AAPL` (any ticker) | Single-stock | Deep analysis of one symbol across all expiries |
| `ALL` | Budget scan | Multi-ticker scan filtered to a per-contract dollar budget |
| `DISCOVER` | Discovery | Scans the top 100 most-liquid tickers, no budget limit |
| `SELL` | Premium Selling | Short put candidates ranked by return-on-risk |
| `SPREADS` | Credit Spreads | Bull Put and Bear Call spread opportunities |
| `IRON` | Iron Condors | Delta-neutral range-bound strategies |
| `PORTFOLIO` | Portfolio | View P/L on all open paper trades |
| `MY LIST` | Watchlist | Scan your personal watchlist (type `ADD AAPL` to build it) |

### CLI Reference

```
python -m src.options_screener [OPTIONS]    # requires venv active
python3 run.py [OPTIONS]                   # auto-activates venv
python3 -m src [OPTIONS]                   # auto-activates venv

Options:
  --no-color             Disable ANSI color output (useful for piping to a log file)
  --no-ai                Skip AI analysis after scan
  --close-trades         Update the trade log with closing prices and realised P/L
  --ui                   Launch the Streamlit web dashboard
  --mode MODE            Skip mode menu: ticker | all | discover | sell | spreads | iron | portfolio | mylist
  --ticker SYM           Ticker symbol (implies --mode ticker)
  --watchlist NAME       Use named watchlist (liquid_large_cap, sector_etfs, high_iv, income)
  --top N                Cross-ticker top-N scan (default 10), grouped by DTE bucket
  --export csv           Export top scan results to scan_results_YYYYMMDD_HHMM.csv
  --auto                 Skip interactive prompts, use config defaults
  --compact              Compact per-pick output (3 lines per pick)
  --no-cache             Disable all caching (requests, AI scores, IV history)
  --weights NAME         Weight profile name (in configs/weights/) or path to JSON; tags logged trades
  --auto-log             Auto-log top-N picks after scan (skips save-menu prompt)
  --log-top N            With --auto-log: how many top picks to log (default 5)
  --list-profiles        List available weight profiles and exit
  --surface              Show 3D P&L risk surface for the top pick (single-stock mode)
  --surface-mode M       Render mode: braille (default, hi-res Unicode) or ascii
  --surface-greek G      Show greek sensitivity surface: delta, gamma, vega, theta
  --no-contours          Disable contour lines on surface
  --viz                  Open interactive 3D visualizer (Plotly, opens in browser)
  -h, --help             Show help and exit
  --version              Show version string and exit
```

### Terminal Output

The CLI renders a fully colour-coded, responsive interface that adapts to your terminal width (60–120 chars):

**Startup**
- Market regime dashboard: VIX level, VIX3M term structure, PCR, SPY momentum, IV premium
- Double-box banner with current date/time
- Colour-coded mode menu

**Report — per pick**
```
  CALL   262.50  2026-03-20   $4.03   31.0%    1494    1095   +0.38   OTM  ★★★☆☆
    ↳ Mechanics: Vol: 1095 OI: 1494 | Spread: 3.7% | Delta: +0.38 | Greeks: Γ … | Cost: $402.50
    ↳ Analysis:  IV: ██████████████░░░░░░ 72%ile | PoP: 55.2% | RR: 1.4x | EV: $18
    ↳ Thesis:    High probability (>65%) • Trend aligned | CHEAP vs surface
    ↳ Entry:     <=3.95  |  Target: $5.93 (+50%)  |  Stop: $2.96 (-25%)
         Breakeven: $266.45  |  Max Loss: $395  |  Confidence: HIGH (87%)
         Exec:     LIMIT @ $3.95 (mid-point) | Vol 1095 vs OI 1,494
         Score:    + PoP +0.14  + RR +0.10  + Vol +0.08
```

**Comparison table** — compact ranked summary after detailed output:
```
  QUICK COMPARISON  —  Top Picks
    #  Tick  Strike        Exp  Score   PoP   R/R   IV%      EV  Sprd    SVI
    1  AAPL  $215C       04/17   0.82   62%  2.1x   65%  $  +45  5.0%  CHEAP
    2  MSFT  $400P       04/17   0.71   55%  1.8x   72%  $  +30  8.0%   RICH
```

**3D Risk Surface** — high-resolution braille or ASCII surface (activated with `--surface`):
```
  3D P&L Risk Surface  —  AAPL $215 CALL
  Price shock: -25% <- -> +25%   |   IV shock: -50% <- -> +50%
  P&L range: $-4.50  to  $+18.20

  [Braille-rendered isometric 3D surface with truecolor gradient]
  [Contour lines: white = breakeven, yellow = iso-value levels]

  Legend: ████████████████████████████████  $-5 to $+18
```

Supports P&L and Greek sensitivity surfaces (`--surface-greek delta|gamma|vega|theta`).
Uses Unicode braille characters (2x4 dots per character) for ~8x pixel density vs ASCII.
Falls back to ASCII shading automatically if the terminal doesn't support Unicode.

**Interactive 3D Visualizer** — browser-based multi-view dashboard (activated with `--viz` or `[V]` at the post-scan menu):

Six tabbed views, all interactive with rotation, zoom, pan, and hover tooltips:

| Tab | What it shows |
|-----|---------------|
| **Contract Explorer** | 3D scatter of all screened contracts. Four dropdown menus remap X/Y/Z axes and colour to any of 20 numeric fields (moneyness, DTE, score, IV, Greeks, EV, volume, etc.). Top picks highlighted as gold diamonds. |
| **IV Surface** | SVI-fitted implied volatility surface with market IVs overlaid as scatter points. Points coloured by residual: blue = CHEAP vs surface, red = RICH. |
| **Greek Landscape** | Delta/gamma/vega/theta sensitivity surfaces for the top pick. Toggle between Greeks with buttons. Full BS repricing across price and IV shocks. |
| **P&L Scenarios** | P&L surface for top 3 picks across ±25% price and ±50% IV shocks. Red = loss, green = profit. Breakeven plane at $0. Toggle individual contracts. |
| **Score Decomposition** | Parallel coordinates of 12 most-weighted score components. Drag on any axis to filter contracts. Lines coloured by quality score. |
| **Risk Radar** | Spider chart comparing top 5 contracts across 7 risk dimensions (PoP, EV, Liquidity, IV Edge, VRP, Momentum, Greeks). |

Dark financial theme. No extra dependencies — uses Plotly (already in requirements). Opens as a standalone HTML page in your default browser.

**Stress test** — 7×3 scenario matrix with full Black-Scholes repricing:
```
  PORTFOLIO STRESS TEST  —  3 open position(s)  [full BS repricing]
  IV Shock       -20%      -10%       -5%       +0%       +5%      +10%      +20%
  IV flat       -1,245      -580      -267        +0      +215      +390      +640
  IV +10%       -1,400      -735      -422      -155       +60      +235      +485
  IV +20%       -1,555      -890      -577      -310       -95       +80      +330
```

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

| Model | ID for config_ai.py | Role | Notes |
|-------|---------------------|------|-------|
| **Arcee Trinity** (primary) | `arcee-ai/trinity-large-preview:free` | Primary model | Good reasoning, free |
| **StepFun 3.5 Flash** | `stepfun/step-3.5-flash:free` | 1st fallback | Fast reasoning model |
| **Nemotron 120B** | `nvidia/nemotron-3-super-120b-a12b:free` | 2nd fallback | Large model, strong analysis |
| Llama 3.3 70B | `meta-llama/llama-3.3-70b-instruct:free` | 3rd fallback | Solid general-purpose |

> The free tier is rate-limited. If you hit 429 errors, the screener retries automatically with exponential backoff and switches to the fallback model. For production use, add a small OpenRouter credit balance to lift rate limits.

#### Paid models (recommended for serious use)

Adding credit to your OpenRouter account ($5–10 goes a long way) unlocks faster, higher-quality models:

| Model | ID for config_ai.py | Cost | Notes |
|-------|---------------------|------|-------|
| **Claude Sonnet 4.6** | `anthropic/claude-sonnet-4-6` | ~$3/M tok | Best reasoning on options context |
| GPT-4o | `openai/gpt-4o` | ~$5/M tok | Strong; good at structured JSON output |
| Claude Haiku 4.5 | `anthropic/claude-haiku-4-5` | ~$0.80/M tok | Fast and cheap; good enough for screening |
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

   # Optional: per-model keys for specialised models
   # OPENROUTER_ARCEE_KEY=sk-or-v1-...
   # OPENROUTER_STEPFUN_KEY=sk-or-v1-...
   # OPENROUTER_NVIDIA_KEY=sk-or-v1-...

   # Optional: Polygon.io for higher-quality news, VWAP, unusual flow
   # POLYGON_API_KEY=your_polygon_key_here
   ```
   See `.env.example` for the full template (also includes Discord/Telegram bot tokens).

3. That's it. The screener loads the key automatically from `.env` at startup (works with or without `python-dotenv` installed).

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
   │  - Fetches options chains + 1-year daily history per ticker
   │  - Calculates Greeks, PoP, EV, IV rank, HV, momentum, warnings
   │  - SVI IV surface fitting → mispricing detection
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
   │  - Batches candidates (default 3 per API call)
   │  - Model fallback chain: arcee → stepfun → nvidia → llama
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
[Rich Table]
   - Summary panel: pick count, avg scores, divergence count, top pick
   - Colour-coded table: scores, types, DTE, PoP, confidence, catalyst risk
   - Alternating row stripes, top-3 highlighting
   - Divergence detail panel for flagged picks
   - Falls back to plain text if rich is not installed
```

### Output Columns Explained

| Column | Description |
|--------|-------------|
| **Tech** | Technical quality score × 100 (0–100). Pure quant — Greeks, PoP, EV, liquidity, momentum |
| **AI** | AI score (0–100). Qualitative — catalyst, IV justification, narrative context |
| **Conf** | AI confidence in its own score (0–10). Higher = more AI weight applied |
| **Final** | Weighted blend: `tw × Tech/100 + aw × AI/100`. This is the ranking signal |
| **Cat** | Catalyst risk: LOW / MED / HI!. Earnings or macro events near expiry |
| **Div** | Divergence: `AI>TECH` (AI bullish vs quant) or `TECH>AI` (quant bullish vs AI). Highlighted in cyan/magenta |
| **Reasoning** | Short flags in normal mode; full 1–2 sentence AI rationale with `--detail` |

**Divergence** is the most actionable signal — when the two models strongly disagree, it's worth reading the reasoning before trading.

---

## Analytics Engine

### Understanding Probability of Profit (PoP)

PoP is arguably the most important metric in the screener. Unlike simple "Delta" (which estimates the probability of the option being 1 cent In-the-Money), **PoP calculates the probability that the trade will be profitable at expiration after accounting for the premium paid.**

*   **Buying Options (Aggressive):** You start with a "premium tax." If you buy a $100 Call for $5.00, your breakeven is $105.00. The PoP will be significantly lower than the Delta because the stock must move +5% just for you to break even.
    *   *Standard PoP:* ~25% – 35%
    *   *Expectation:* High risk, high reward. You expect to lose more often than you win, but wins are outsized (100%+).
*   **Selling Options (Conservative):** You "become the casino." When you sell an option, time is your friend. You win if the stock goes up, stays flat, or even if it drops slightly (less than the premium collected).
    *   *Standard PoP:* ~65% – 85%
    *   *Expectation:* Consistent income, lower risk, capped reward.

**Why don't I see 50%+ PoP?** If you are buying "Out-of-the-Money" (OTM) options, you are intentionally taking a low-probability, high-payoff bet. To see 50%+ PoP, switch the screener to **"SELL" (Premium Selling) mode** or buy **Deep In-the-Money (ITM)** options.

### Scoring

Each contract is scored across 27 weighted factors including:

| Factor | What it measures |
|--------|-----------------|
| Probability of Profit | Blended Black-Scholes + Monte Carlo PoP |
| Expected Move Realism | How achievable the breakeven is vs. the 1σ expected move |
| Risk/Reward | Payoff at 0.75× EM target vs. premium paid |
| Momentum | RSI, 5-day return, ATR trend, confluence scoring |
| IV Rank | IV percentile vs. 30-day range; buyers rewarded for low IV |
| IV vs. HV Edge | HV-adjusted EV — positive = options cheap vs. realised vol |
| IV Mispricing | SVI surface residual — contract rich or cheap vs fitted vol surface |
| Liquidity | Volume + open interest, rank-normalised |
| Catalyst | Crush-magnitude-adjusted earnings penalty |
| Theta Efficiency | Time decay pressure relative to delta |
| Skew Alignment | Put/call IV skew directional bias |
| Gamma/Theta Ratio | Explosive payoff potential per unit of daily bleed |
| Expected Value | HV-adjusted BS value minus market price minus spread cost |

Scores are further adjusted for trend alignment, decay risk, gamma squeeze setup, OI wall proximity, macro risk, seasonal win rate, and VIX regime multipliers.

**VIX regime adaptation** — scoring weights shift automatically based on the current VIX level (low / normal / high), emphasising different factors in different volatility environments. The AI weight also scales with VIX regime (×0.80 in low VIX, ×1.30 in high VIX).

### Pricing

- **European options** — full Black-Scholes with continuous dividend yield
- **American options** — Barone-Adesi-Whaley (1987) approximation with dividend yield flow-through
- **IV surface** — SVI (Stochastic Volatility Inspired) parameterisation per expiry; mispricing = market IV vs fitted surface
- **Stress test** — full BS repricing across 7×3 stock-move × IV-shock scenarios (delta-gamma fallback for edge cases)

### Greeks

All Greeks are computed analytically via Black-Scholes with continuous dividend yield, vectorised with NumPy across the full chain in a single batch: delta, gamma, vega, theta, rho, charm, vanna.

### Monte Carlo

Probability of Profit is blended: **60% Monte Carlo** (Merton Jump Diffusion GBM, captures path-dependency and tail risk) + **40% analytical PoP**, using `numpy.random.default_rng` for reproducible, thread-safe results.

### Trade Plan Generation

For each pick the screener generates:
- **Thesis** — plain-English explanation of why the trade ranks highly, with CHEAP/RICH vs surface flags
- **Entry price** — limit order guidance based on bid-ask spread width
- **Profit target and stop loss** — from `exit_rules` in `config.json`
- **Breakeven and max loss**
- **Confidence score** — penalises wide spreads, low liquidity, short DTE; rewards unusual flow
- **Execution guidance** — limit price, volume vs OI ratio
- **Quality score breakdown** — per-component contribution to the composite score
- **Risk list** — flags wide spreads, negative EV, earnings risk, low liquidity, OI walls, macro

---

## Paper Trading

Log any pick directly from the CLI and track it going forward:

- Positions auto-update on every launch (fetches live quotes via yfinance)
- Entry IV and Greeks stored per trade (schema v8)
- All 27 per-component scores stored at entry — enables per-component IC analysis after trades close
- Take Profit and Stop Loss thresholds enforced from `config.json`
- P&L attribution for closed trades (delta/gamma/theta/vega breakdown)
- Full BS repricing stress test on open portfolio
- Win rate, total P/L, and average return tracked in a local SQLite database
- Close expired positions with `python -m src.options_screener --close-trades`
- View portfolio with `python -m src.check_pnl` (or press `7` / `PORTFOLIO` at the mode menu)

---

## Weight-Profile Auto-Logging

Collect optimization data for the 27 composite weights by running the screener under different weight configurations and auto-logging the best picks. Every logged trade is tagged with a `weight_profile` column so you can later compute per-profile win rate, IC, and P&L and pick the best weight set.

### The default "mimic a full scan and log the best trades" command

```bash
python -m src.options_screener --mode discover --weights baseline --auto-log --log-top 5 --auto
```

This is the recommended daily data-collection command. It:

- Runs a **Discovery scan** across the top 100 most-liquid tickers (thousands of contracts screened).
- Applies the **baseline** weight profile (a snapshot of `config.json`'s `composite_weights` — the control).
- After scoring, takes the **top 5 picks** by `quality_score` and logs them to `paper_trades.db`.
- **Deduplicates**: if you re-run the same day under the same profile, duplicates are skipped so noise doesn't pollute the dataset.
- **No prompts** (`--auto`): runs end-to-end unattended.

Each auto-logged trade is identical to a manually-logged one — it appears in the portfolio view (`python -m src.check_pnl` or `7` at the mode menu), contributes to portfolio Greeks/stress test, and can be closed with `--close-trades`.

### A/B-testing weight sets

Create a variant in `configs/weights/` (copy `baseline.json`, tweak weights), then run both:

```bash
python -m src.options_screener --mode discover --weights baseline --auto-log --auto
python -m src.options_screener --mode discover --weights exp_high_pop --auto-log --auto
```

Same scan, different weights → different rankings → different logged picks. Over weeks, the DB builds up per-profile P&L that you can compare directly.

```bash
# List available profiles
python -m src.options_screener --list-profiles

# Inspect per-profile counts
sqlite3 paper_trades.db "SELECT weight_profile, COUNT(*) FROM trades GROUP BY weight_profile"

# Per-profile P&L on closed trades
sqlite3 paper_trades.db "SELECT weight_profile, COUNT(*), ROUND(AVG(pnl_pct)*100,1) AS avg_pct FROM trades WHERE status='CLOSED' GROUP BY weight_profile"
```

### Creating a new profile

```bash
cp configs/weights/baseline.json configs/weights/my_experiment.json
# Edit the file — tweak any of the 27 weights under "composite_weights"
python -m src.options_screener --weights my_experiment --auto-log --auto --mode discover
```

Unknown keys log a warning; non-numeric values error out. If `--weights` is omitted, behaviour is unchanged from pre-profile runs (trades logged with `weight_profile = NULL`).

### Dedup rule

`(calendar_day, ticker, strike, expiration, type, weight_profile)` is the unique key. Meaning:

- Same contract re-scanned on the same day under the same profile → **skipped**.
- Same contract under a different profile → **inserted** (so profiles can diverge).
- Same contract the next day → **inserted** (re-entry signal preserved).

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
  "vix_regime_multipliers": {
    "low":    { "pop": 1.0, "ev": 1.2, "rr": 1.1 },
    "normal": { "pop": 1.0, "ev": 1.0, "rr": 1.0 },
    "high":   { "pop": 1.2, "ev": 0.8, "rr": 1.3 }
  }
}
```

### src/config_ai.py — AI layer settings

```python
AI_CONFIG = {
    # API provider: "openrouter" (default) or "anthropic"
    "provider":              "openrouter",
    "model":                 "arcee-ai/trinity-large-preview:free",
    "fallback_model":        "stepfun/step-3.5-flash:free",
    "second_fallback_model": "nvidia/nemotron-3-super-120b-a12b:free",
    "third_fallback_model":  "meta-llama/llama-3.3-70b-instruct:free",
    "api_key_env":           "OPENROUTER_ARCEE_KEY",

    # Per-model API key overrides
    "model_key_map": {
        "arcee-ai/trinity-large-preview:free":    "OPENROUTER_ARCEE_KEY",
        "stepfun/step-3.5-flash:free":            "OPENROUTER_STEPFUN_KEY",
        "nvidia/nemotron-3-super-120b-a12b:free": "OPENROUTER_NVIDIA_KEY",
    },

    # Scoring weights
    "ai_weight":        0.30,
    "technical_weight": 0.70,

    # Dynamic weight multipliers by VIX regime
    "regime_weight_multipliers": {
        "low":    0.80,
        "normal": 1.00,
        "high":   1.30,
    },

    # Feature flags
    "cache_enabled":      True,
    "confidence_enabled": True,
    "two_pass_enabled":   True,
    "news_enabled":       True,

    # API call settings
    "batch_size":   3,
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

## API + Discord & Telegram Bots

The screener can be exposed as a local HTTP API so that Discord and Telegram bots can query it from anywhere — live options data from your phone or in a chat app without opening a terminal.

### Architecture

```
Discord Bot  ──┐
               ├──► FastAPI server :8000 ──► run_scan / get_market_context
Telegram Bot ──┘
```

### Setup

```bash
# Install bot dependencies
pip install -r requirements_bots.txt

# Set up tokens
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux

# Edit .env with real values:
#   DISCORD_BOT_TOKEN=...
#   DISCORD_GUILD_ID=...    (server ID for instant slash-command sync)
#   TELEGRAM_BOT_TOKEN=...
#   API_BASE_URL=http://127.0.0.1:8000
```

### Running

```bash
# Start everything at once (API + both bots)
python start_all.py

# Or start individually:
python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
python -m src.bots.discord_bot
python -m src.bots.telegram_bot
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /market` | VIX level, regime, trend, macro risk (cached 60s) |
| `GET /top?n=10` | Top N picks from `liquid_large_cap` watchlist |
| `GET /scan/{symbol}` | Scan a single ticker |
| `GET /watchlist/{name}` | Scan any named watchlist from config.json |

### Bot Commands

Both bots support the same four commands:

| Command | What it does |
|---------|-------------|
| `/market` | Current market context: trend, VIX, macro risk |
| `/top` | Top picks from the liquid_large_cap watchlist |
| `/scan AAPL` | Scan a single ticker |
| `/watchlist high_iv` | Scan a named watchlist (`liquid_large_cap`, `sector_etfs`, `high_iv`, `income`) |

---

## Project Structure

```
options/
├── run.py                    # Auto-venv launcher: python3 run.py (no activation needed)
├── ai_rank.py                # AI-enhanced entry point — runs screener + AI scoring
├── options_screener.py       # Thin wrapper for backward compatibility
├── backtest_screener.py      # Backtester entry point
├── config.json               # Screener thresholds, weights, and exit rules
├── configs/
│   └── weights/              # Weight profiles for A/B optimization (baseline.json + experiments)
├── watchlist.json            # Personal ticker list (auto-created by ADD command)
├── requirements.txt
├── requirements_bots.txt     # Bot/API dependencies (fastapi, discord.py, python-telegram-bot)
├── start_all.py              # Launch API + both bots in one command
├── .env                      # Your API keys (gitignored — never committed)
├── .env.example              # Template showing required env vars (screener + bots)
├── README.md
└── src/
    ├── __main__.py           # Auto-venv launcher for python3 -m src
    ├── options_screener.py   # Core scan engine, report printing, spread/condor finders
    ├── cli_display.py        # Terminal display: per-pick detail, comparison table, report
    ├── data_fetching.py      # yfinance single-fetch, technical indicators, chain cache
    ├── filters.py            # Chain filtering, IV smile outlier removal, premium bucketing
    ├── scoring.py            # Composite quality score re-exports
    ├── trade_analysis.py     # Thesis, entry/exit levels, confidence, execution guidance
    ├── risk_engine.py        # OI wall detection, gamma ramp flags, structural risk
    ├── utils.py              # Vectorised BS Greeks + BAW American pricing (NumPy/SciPy)
    ├── iv_surface.py         # SVI IV surface fitting, mispricing detection
    ├── simulation.py         # Monte Carlo PoP / PoT (Merton Jump Diffusion GBM)
    ├── formatting.py         # ANSI colours, box drawing, metric formatters
    ├── stress_test.py        # Full BS repricing scenario P/L matrix
    ├── paper_manager.py      # SQLite paper trade logging, schema migration (v8), dedup, weight_profile tagging
    ├── weight_profiles.py    # Load/list weight profiles from configs/weights/
    ├── check_pnl.py          # Standalone portfolio P/L viewer with Greeks
    ├── news_fetcher.py       # Concurrent multi-source news (yfinance, Finviz, Polygon)
    ├── macro_analyzer.py     # Macro risk gating (10Y yield, VIX, DXY, credit spreads)
    ├── portfolio_risk.py     # Portfolio VaR, correlation analysis, position sizing
    ├── sector_analyzer.py    # Sector-level analysis and rotation signals
    ├── polygon_client.py     # Polygon.io integration (news, VWAP, unusual flow)
    ├── vol_analytics.py      # Vol cone, concurrent IV surface, regime classification
    ├── regime_dashboard.py   # VIX/PCR/SPY market regime dashboard at startup
    ├── backtester.py         # Walk-forward backtester with spread cost simulation
    ├── backtest_optimizer.py # Weight optimizer for composite_weights via IC analysis
    ├── calc_expected_move.py # Implied expected move calculator
    ├── oi_snapshot.py        # OI change tracking between runs
    ├── watchlist.py          # Watchlist management (ADD/REMOVE commands)
    ├── visual_surface.py     # 3D risk surface: braille/ASCII, P&L + Greek surfaces, contours
    ├── visualizer_3d.py      # Interactive 3D Plotly visualizer (6-tab browser dashboard)
    ├── visualize_results.py  # Matplotlib/Plotly charts for scan results
    ├── dashboard.py          # Streamlit web interface
    ├── ai_scorer.py          # Two-pass AI scoring with retry, fallback, narrative context
    ├── ai_cache.py           # Same-day SQLite cache for AI scores
    ├── ranking.py            # combine_scores(), Rich ranked table, divergence detection
    ├── prompts.py            # AI system/scoring prompt templates
    ├── config_ai.py          # AI layer configuration (models, weights, thresholds)
    ├── config_validator.py   # Config.json validation at module load
    ├── types.py              # Shared type definitions and data classes
    ├── api.py                # FastAPI server (/market, /top, /scan, /watchlist)
    └── bots/
        ├── __init__.py
        ├── discord_bot.py    # Discord slash commands (/market, /top, /scan, /watchlist)
        └── telegram_bot.py   # Telegram command handlers (same 4 commands)
```

---

## Roadmap

- [x] Vectorised Black-Scholes Greeks engine with dividend yield
- [x] Barone-Adesi-Whaley American option pricing
- [x] SVI IV surface fitting with mispricing detection
- [x] Monte Carlo PoP blending (Merton Jump Diffusion)
- [x] HV-adjusted expected value
- [x] Paper trading with entry IV/Greeks + all 27 component scores stored, P&L attribution, schema v8
- [x] Weight-profile auto-logging system — `--weights NAME --auto-log` for A/B optimization with per-profile dedup
- [x] Full BS repricing stress test (7×3 scenario matrix)
- [x] Streamlit dashboard
- [x] Full colour CLI — responsive width, trade plan per pick, comparison table
- [x] Execution guidance (limit price, volume/OI analysis)
- [x] Quality score breakdown per component
- [x] Credit spread and iron condor screeners
- [x] Volatility analytics — cone, concurrent IV surface, regime dashboard
- [x] Walk-forward backtester with realistic slippage, commissions, and spread cost simulation
- [x] AI scoring and ranking layer (two-pass, dynamic weights, divergence detection)
- [x] Rich coloured AI ranking table with summary panel and divergence detail
- [x] Same-day AI score cache (SQLite)
- [x] Narrative context enrichment (IV/HV edge, PoP, RR, theta burn, warnings)
- [x] Portfolio coherence check (`--portfolio-check`)
- [x] Polygon.io integration — ticker-filtered news, real-time VWAP, unusual options flow
- [x] Multi-source concurrent news fetcher (yfinance RSS, Finviz, Polygon)
- [x] Macro risk gating (10Y yield, VIX regime, DXY, credit spreads)
- [x] Portfolio VaR and correlation analysis
- [x] OI wall and gamma ramp structural risk checks
- [x] In-session options chain cache (eliminates redundant yfinance fetches)
- [x] VIX regime weight multipliers for scoring and AI weight
- [x] Historical IV crush estimation per ticker
- [x] Config validation at module load
- [x] 3D risk surface — braille hi-res + ASCII fallback, truecolor gradient, Greek surfaces, contour lines
- [x] Interactive 3D visualizer — 6-tab Plotly browser dashboard (contract explorer, IV surface, Greek landscape, P&L scenarios, score decomposition, risk radar)
- [x] FastAPI local HTTP server (market context, top picks, single-symbol scan, watchlist scan)
- [x] Discord bot — slash commands: /market, /top, /scan, /watchlist
- [x] Telegram bot — same 4 commands, Markdown formatting
- [x] Single-launcher (start_all.py) for all 3 processes with clean Ctrl+C shutdown
- [ ] Real-time alerts (email / SMS)
- [ ] Multi-leg spread support in paper manager
- [ ] Backtesting UI improvements

---

## Disclaimer

For educational and informational purposes only. Not financial advice. Options trading involves substantial risk of loss. Always do your own research.

## License

Personal use only. Not licensed for commercial distribution.
