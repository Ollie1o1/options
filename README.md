# Options Screener — Professional Edition

A Python-based options screening tool that identifies high-probability trading opportunities through advanced analytics, institutional-level metrics, and dynamic safety filters — delivered through a fully color-coded, professional terminal interface.

## Core Philosophy: Hunting for Alignment

A high-quality trade occurs when multiple independent factors align simultaneously:

1. **Action** — The market is interested: high volume, tight spreads, unusual flow.
2. **Edge** — The probabilities favour you: IV vs. HV, seasonality, risk/reward.
3. **Structure** — The trade is not fighting a technical barrier: support/resistance, OI walls.
4. **Trend** — The trade aligns with the stock's momentum and broader market context.

The screener finds opportunities where all four forces converge. Consistency comes from saying "no" to good volume on a bad chart.

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Ollie1o1/options.git
cd options
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run

```bash
# Interactive CLI
python -m src.options_screener

# Streamlit dashboard
python -m src.options_screener --ui
```

> The screener must be run as a module (`-m`) because it uses relative imports.

---

## CLI Reference

```
python -m src.options_screener [OPTIONS]

Options:
  --no-color      Disable all ANSI color output (useful for logging to file)
  -h, --help      Show help and exit
  --version       Show version string and exit
  --close-trades  Update the trade log with closing prices and realized P/L
  --ui            Launch the Streamlit web dashboard
```

### Scan Modes

| Input | Mode | Description |
|---|---|---|
| `AAPL` (any ticker) | Single-stock | Deep analysis of one symbol |
| `ALL` | Budget scan | Multi-ticker scan within a dollar budget per contract |
| `DISCOVER` | Discovery | Scan the top 100 most-liquid tickers, no budget limit |
| `SELL` | Premium Selling | Short put candidates ranked by return-on-risk |
| `SPREADS` | Credit Spreads | Bull Put and Bear Call spread opportunities |
| `IRON` | Iron Condors | Delta-neutral range-bound strategies |
| `PORTFOLIO` | Portfolio | View P/L on open paper trades |

---

## Terminal Output

The CLI renders a fully color-coded, responsive interface that adapts to your terminal width (60–120 chars):

**Startup**
- Double-box banner with current date/time
- Color-coded mode menu: yellow index, bold white command, dim description
- Market context: trend (green/red/yellow), VIX regime, macro risk warning

**Scan progress**
- Single-line tqdm bar that updates in place — no noise from third-party libraries
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

**Executive Summary** — printed after all bucket picks, showing top 3 opportunities with entry levels and portfolio warnings (wide spreads, negative EV, earnings risk, low liquidity).

---

## Analytics Engine

### Scoring

Each contract is scored across 13 weighted factors:

| Factor | What it measures |
|---|---|
| Probability of Profit | Blended Black-Scholes + Monte Carlo PoP |
| Expected Move Realism | How achievable the breakeven move is vs. 1σ EM |
| Risk/Reward | Payoff at 0.75× EM target vs. premium paid |
| Momentum | RSI, 5-day return, ATR trend |
| IV Rank (mode-aware) | IV percentile vs. 30-day range; buyers rewarded for low IV |
| IV vs. HV Edge | HV-adjusted EV — positive means options are cheap vs. realised vol |
| Liquidity | Volume + open interest, rank-normalised |
| Catalyst | Earnings proximity bonus/penalty |
| Theta Efficiency | Time decay pressure relative to delta |
| Skew Alignment | Put/call IV skew directional bias |
| Gamma/Theta Ratio | Explosive payoff potential per unit of daily bleed |
| Trader Profile | Liquidity-weighted for day traders; DTE-weighted for swing |
| Expected Value | HV-adjusted BS value minus market price minus spread cost |

Scores are then adjusted for trend alignment (+0.15), decay risk (−0.20), gamma squeeze setup (+0.25), OI wall (−0.10), macro risk (−0.10), and seasonality.

### Greeks

All Greeks are computed analytically via Black-Scholes (vectorized NumPy):
delta, gamma, vega, theta, rho — calculated per contract across the full chain in a single batch.

### Monte Carlo

Probability of Profit is blended: 60% Monte Carlo simulation (captures path-dependency and jump risk) + 40% analytical PoP, when simulation data is available.

### Trade Plan Generation

For each pick, the screener generates:
- **Thesis** — plain-English explanation of why the trade ranks highly
- **Entry price** — bid-to-mid improvement based on spread width
- **Profit target and stop loss** — from `exit_rules` in `config.json`
- **Breakeven and max loss**
- **Confidence score** — penalises wide spreads, low liquidity, short DTE; rewards unusual flow
- **Risk list** — liquidity, spread, time decay, IV crush, earnings, EV, OI walls, macro

---

## Paper Trading

Log any pick directly from the CLI and track it forward:

- Positions auto-update on every launch (fetches live quotes)
- Take Profit and Stop Loss thresholds enforced from `config.json`
- Win rate, total P/L, and average return tracked over time
- Close expired positions with `--close-trades`

---

## Project Structure

```
options/
├── src/
│   ├── options_screener.py   # CLI entrypoint, scan engine, report printing
│   ├── formatting.py         # ANSI colors, box drawing, metric formatters
│   ├── trade_analysis.py     # Thesis generation, entry/exit levels, confidence
│   ├── data_fetching.py      # yfinance wrappers, caching, market context
│   ├── filters.py            # Configuration-driven chain filtering
│   ├── scoring.py            # Composite quality score components
│   ├── simulation.py         # Monte Carlo PoP / PoT engine
│   ├── paper_manager.py      # Paper trade logging and position tracking
│   ├── dashboard.py          # Streamlit web interface
│   ├── backtest_screener.py  # Historical backtest runner
│   └── utils.py              # Vectorized BS math, formatting helpers
├── config.json               # All thresholds, weights, and exit rules
└── requirements.txt
```

---

## Configuration

`config.json` controls filters, scoring weights, and exit rules:

```json
{
  "filters": {
    "min_volume": 50,
    "min_open_interest": 10,
    "max_bid_ask_spread_pct": 0.40,
    "delta_min": 0.15,
    "delta_max": 0.35,
    "min_days_to_expiration": 7,
    "max_days_to_expiration": 45
  },
  "exit_rules": {
    "take_profit": 0.50,
    "stop_loss": -0.25
  },
  "composite_weights": {
    "pop": 0.20,
    "em_realism": 0.10,
    "rr": 0.12,
    "liquidity": 0.12,
    "ev": 0.10
  }
}
```

---

## Roadmap

- [x] Vectorized Black-Scholes Greeks engine
- [x] Monte Carlo PoP blending
- [x] HV-adjusted expected value
- [x] Paper trading with automated exit tracking
- [x] Streamlit dashboard
- [x] Full color CLI — responsive width, trade plan per pick, executive summary
- [x] Credit spread and iron condor screeners
- [ ] Real-time alerts (email/SMS)
- [ ] Multi-leg spread support in paper manager
- [ ] Backtesting UI improvements

---

---

## AI-Enhanced Scoring Layer

The `ai_rank.py` wrapper adds an AI scoring layer on top of the technical screener.
It batches candidates and sends them to Claude for qualitative analysis, then
combines the AI score with the screener's `quality_score` into a single ranked list.

### Setup

1. Install the extra dependencies (already in `requirements.txt`):
   ```bash
   pip install anthropic rich
   ```

2. Export your Anthropic API key:
   ```bash
   export ANTHROPIC_API_KEY=sk-ant-...   # macOS / Linux
   set  ANTHROPIC_API_KEY=sk-ant-...     # Windows CMD
   $env:ANTHROPIC_API_KEY="sk-ant-..."   # Windows PowerShell
   ```

### Usage

```bash
# Score and rank candidates for one or more tickers
python ai_rank.py AAPL TSLA NVDA

# Premium Selling mode, custom DTE window
python ai_rank.py --mode "Premium Selling" --dte-min 14 --dte-max 45 SPY QQQ

# Skip AI — rank by technical quality_score only (no API key needed)
python ai_rank.py AAPL --no-ai

# Increase AI weight to 50 % and save results
python ai_rank.py SPY --ai-weight 0.5 --output ranked.csv

# All options
python ai_rank.py --help
```

### How it works

| Step | Module | Description |
|------|--------|-------------|
| 1 | `src/options_screener.py` | Technical scan — Greeks, PoP, IV rank, EV, warnings |
| 2 | `src/ai_scorer.py` | Batches candidates → Claude API → `ai_score` (0–100) + reasoning + flags |
| 3 | `src/ranking.py` | Weighted merge: `final_score = 0.70 × quality_score + 0.30 × (ai_score / 100)` |
| 4 | `src/ranking.py` | Ranked table via `rich` (or plain text fallback) |

### Configuration

Edit `src/config_ai.py` to change:

| Key | Default | Purpose |
|-----|---------|---------|
| `model` | `claude-sonnet-4-6` | Claude model used for scoring |
| `ai_weight` | `0.30` | AI score contribution to `final_score` |
| `technical_weight` | `0.70` | `quality_score` contribution |
| `batch_size` | `5` | Candidates per API call (cost control) |
| `temperature` | `0.1` | Lower = more consistent scores |
| `fields_to_include` | (list) | Candidate fields forwarded to the AI |

### New files

```
ai_rank.py            CLI wrapper — entry point
src/ai_scorer.py      AIScorer class — batched Claude API scoring
src/ranking.py        combine_scores(), print_ranked_table(), to_csv(), to_json()
src/config_ai.py      AI configuration constants
```

---

## Disclaimer

For educational and informational purposes only. Not financial advice. Options trading involves substantial risk of loss. Always do your own research.

## License

Personal use only. Not licensed for commercial distribution.
