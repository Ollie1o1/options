# Options Screener - Professional Edition

A comprehensive Python-based options screening tool with advanced analytics, probability calculations, and trade tracking capabilities.

## 🎯 Overview

This professional-grade screener analyzes options chains with **institutional-level metrics** and outputs actionable trading opportunities across different price ranges.

### **Key Features:**
- **Three Operating Modes:**
  - **Single-Stock Mode**: Deep dive analysis of one ticker
  - **Budget-Based Multi-Stock Mode**: Scan multiple tickers within budget constraints
  - **Discovery Mode**: Scan top 100 most-traded tickers for best overall opportunities
- **Advanced Options Analytics:**
  - Probability of Profit (PoP)
  - Expected Move (1 SD)
  - Probability of Touch
  - Risk/Reward Ratios
  - Max Loss & Break-even calculations
- **Volatility Analysis:**
  - 30-day Historical Volatility (HV) fetching
  - IV vs HV comparison
  - IV Skew (put IV vs call IV)
- **Quality Scoring System:**
  - Multi-factor weighted algorithm
  - 25% Liquidity + 20% IV Advantage + 20% R/R + 15% PoP + 10% Spread + 10% Delta
- **Trade Management:**
  - CSV export with all metrics
  - Trade logging for P/L tracking
  - Timestamp-based file organization
- Real-time data via Yahoo Finance (yfinance API)
- Automatic risk-free rate fetching from 13-week Treasury
- Black-Scholes delta calculations
- **Top Overall Pick** with intelligent justification
- ITM/OTM moneyness indicators
- Liquidity & spread quality flags
- Category summary statistics

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/Ollie1o1/options.git
cd options
```

2. **(Recommended) Create and activate a virtual environment:**
```bash
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
```

3. **Install required dependencies:**
```bash
pip install -r requirements.txt
```

### Usage

Run the screener:
```bash
python options_screener.py
```

---

## 📊 Operating Modes

### Mode 1: Single-Stock Analysis

**Interactive Prompts:**
```
Enter stock ticker, 'ALL', or 'DISCOVER': AAPL
How many nearest expirations to scan [4]: 4
Minimum days to expiration (DTE) [7]: 7
Maximum days to expiration (DTE) [120]: 60
Fetching current risk-free rate...
Using risk-free rate: 4.35% (13-week Treasury)
```

**What You Get:**
- Deep analysis of one ticker
- 15 picks (5 low/5 medium/5 high premium)
- All advanced metrics for focused trading

---

### Mode 2: Budget-Based Multi-Stock Scan

**Interactive Prompts:**
```
Enter stock ticker, 'ALL', or 'DISCOVER': ALL
Enter your budget per contract in USD (e.g., 500) [500]: 750
Enter comma-separated tickers to scan [AAPL,MSFT,NVDA,AMD,TSLA,SPY,QQQ,AMZN,GOOGL,META]: 
How many nearest expirations to scan [4]: 4
Minimum days to expiration (DTE) [7]: 7
Maximum days to expiration (DTE) [120]: 60
```

**Budget Categorization:**
- **LOW**: 0-33% of budget (e.g., $0-$250 for $750 budget)
- **MEDIUM**: 33-66% of budget (e.g., $250-$500 for $750 budget)
- **HIGH**: 66-100% of budget (e.g., $500-$750 for $750 budget)

---

### Mode 3: Discovery Mode

**Interactive Prompts:**
```
Enter stock ticker, 'ALL', or 'DISCOVER' []: DISCOVER
How many tickers to scan (1-100) [50]: 50
How many nearest expirations to scan [4]: 4
Minimum days to expiration (DTE) [7]: 7
Maximum days to expiration (DTE) [120]: 90
```

**Features:**
- Scans top 100 most-traded tickers
- No budget limit
- Quantile-based categorization
- Auto-diversifies across tickers

---

## 📈 Advanced Metrics Explained (v3+)

The screener now focuses on **probability-aware, scenario-based metrics**. This
section walks through what each metric means and how to read the output.

### 1. Expected Move (EM)
**Formula:** `EM ≈ Stock Price × IV × √(DTE / 365)`

This is the **1 standard deviation price move** the options market is
implying between now and expiration.

- EM is shown per contract as `Expected Move: ±$X.YZ`.
- In the CSV export it appears as the `expected_move` column.

**Use cases:**
- Sanity-check whether your strike is *inside* or *outside* the expected move.
- Compare required move to EM (see next section) to filter unrealistic plays.

### 2. Break-even Realism – Required Move vs EM
For each contract we calculate:

- **Required move**: how far the stock must move from current price to
  reach break-even.
  - Calls: `breakeven = strike + premium` → `required_move = max(0, breakeven - spot)`
  - Puts:  `breakeven = strike - premium` → `required_move = max(0, spot - breakeven)`
- **EM Realism score (0–1)** – compares required move to EM:
  - `required_move ≤ 0.5 × EM` → score ≈ 1.0 (very realistic)
  - `0.5–1.0 × EM` → score ≈ 0.7 (reasonable)
  - `> 1.0 × EM` → score decays toward 0.1 (unrealistic / “needs hero move”).

**Where you’ll see it:**
- In the rationale line: `req 3.20 vs EM 8.50`.
- In CSV: `required_move`, `em_realism_score`.

**How to use it:**
- Prefer trades where required move is **well inside EM**.
- Avoid trades where `required_move` is **much larger** than EM – they are
  unlikely to break even within the expected move envelope.

### 3. Probability of Profit (PoP)
PoP is now **delta-based and EM-aware**:

- Base approximation: `PoP ≈ 1 − |delta|`.
- If the strike sits **outside the EM band** (too far from spot), PoP is
  reduced (e.g. multiplied by ~0.7).

**Where you’ll see it:**
- Screen output: `PoP 58.3%` in the rationale line.
- CSV: `prob_profit`.

**Interpretation:**
- **50–60%**: balanced probability / reward.
- **60–70%**: high probability, usually lower RR.
- **>70%**: very conservative, often better for premium-selling structures.

### 4. Theta Decay Pressure (TDP)
Theta Decay Pressure estimates **how much you are paying per day relative to
how sensitive the option is** (delta).

- Raw measure (contract-level): `premium × 100 / DTE`.
- Adjusted for delta so low-delta, high-premium, short-dated options are
  penalized more.
- For DTE ≤ 7 days, the impact of high TDP is increased.

**Where you’ll see it:**
- Rationale: `TDP 14.2/day`.
- CSV: `theta_decay_pressure`, `theta_score` (0–1 where higher is better).

**How to use it:**
- As a buyer, avoid contracts with **very high TDP** unless you have a strong
  near-term catalyst.
- As a short-term trader, compare candidates – lower TDP for similar setups
  generally offers more breathing room.

### 5. Risk / Reward (RR)
Risk/Reward is now tied directly to **Expected Move**:

- Target price is set using EM:
  - Call: `target = spot + 0.75 × EM`
  - Put:  `target = spot − 0.75 × EM`
- Reward is the profit per share at that target **after** paying the premium.
- **RR** = `max_gain_if_target_hit / premium`.

**Where you’ll see it:**
- Output: `RR 3.4x`.
- CSV: `rr_ratio`.

**Interpretation:**
- **RR < 2** → “avoid” region.
- **2–3** → solid.
- **3–4** → strong.
- **≥4** → excellent, but sanity-check EM realism and PoP so it’s not a
  lottery ticket.

### 6. Volatility Context: HV, IV Rank & Percentiles
The screener now computes **short-term and medium-term IV context**:

- 30-day historical volatility (`hv_30d`).
- 30-day & 90-day IV rank and percentile (using realized vol as a proxy):
  - `iv_rank_30`, `iv_percentile_30`.
  - `iv_rank_90`, `iv_percentile_90`.

**Simple interpretation:**
- **Low percentile (0–20%)** → IV is cheap → better for **buying** options.
- **Mid (20–60%)** → IV is fair.
- **High (60–100%)** → IV is expensive → better for **selling** premium.

These feed into the `iv_rank_score` used by the composite quality score.

### 7. Momentum Indicators
Momentum is computed once per underlying and applied to all its contracts:

- **5-day return (`ret_5d`)** – short-term price momentum.
- **14-day RSI (`rsi_14`)** – mean-reversion vs. trend strength.
- **ATR trend (`atr_trend`)** – how current ATR compares to its recent
  average (measures volatility expansion/ contraction).

**Where you’ll see it:**
- Rationale: `5d +3.2%, RSI 54`.
- CSV: `ret_5d`, `rsi_14`, `atr_trend`, `momentum_score`.

**How to use it:**
- Strong positive `ret_5d` + healthy RSI (40–60) with mildly rising ATR can
  support trend-following call buys.
- Overbought RSI (>70) + high ATR might call for caution on new longs.

### 8. Catalyst Awareness (Earnings)
The screener now tries to detect the **next earnings date** via yfinance and
marks options whose expiration is close to that date.

- Column `event_flag`:
  - `OK` → no major catalyst detected.
  - `EARNINGS_NEARBY` → expiration within a small buffer around earnings
    (configurable, default 5 days).

**Where you’ll see it:**
- Rationale: includes `earnings soon` when flagged.
- CSV: `event_flag`, `catalyst_score`.

**How to use it:**
- For **directional earnings bets**, you may *prefer* `EARNINGS_NEARBY`.
- For **theta strategies** (selling options), you may choose to avoid these
  unless that risk is intentional.

---

---

## 🏆 Composite Quality Score (0–1.0)

The old 6-factor score has been replaced by a **probability-based composite
score** that better reflects real-world trade quality.

### High-level formula

The final `quality_score` is a weighted blend of the following normalized
components (all 0–1):

- **PoP score** – from `prob_profit`.
- **EM Realism** – from `em_realism_score`.
- **Risk/Reward score** – from `rr_score`.
- **Momentum score** – from `momentum_score`.
- **IV Rank score** – from `iv_rank_score`.
- **Liquidity score** – from `liquidity_score`.
- **Catalyst score** – from `catalyst_score`.
- **Theta score** – from `theta_score`.
- **EV score** – from `ev_score` (expected value per contract).
- **Trader preference** – from `trader_pref_score` (day vs swing).

Default weights (configurable in `config.json` under `composite_weights`):

```text
pop:         0.18
em_realism:  0.12
rr:          0.15
momentum:    0.10
iv_rank:     0.10
liquidity:   0.15
catalyst:    0.05
theta:       0.10
ev:          0.05
trader_pref: 0.10
```

Weights are automatically normalized to sum to 1.0.

### Reading the Score

- **0.80 – 1.00**: A+ / A setup – strong across most dimensions.
- **0.65 – 0.80**: Solid, worth consideration.
- **0.50 – 0.65**: Mixed bag – inspect the rationale carefully.
- **< 0.50**: Low quality – usually fails on realism, liquidity, or RR.

The qualitative rationale line (under each contract) is designed to tell you
*why* the score is high/low:

- Liquidity, spread, and delta context.
- EM realism, PoP, and RR.
- Momentum and RSI snapshot.
- Catalyst warning (earnings soon) and theta pressure.

### Day trader vs Swing trader modes

After choosing DTE bounds, the CLI asks for a trading style profile:

```text
Select trading style profile:
  1. Swing trader (default) - balanced delta + more DTE
  2. Day / short-term trader - prioritize liquidity & tight spreads
```

This choice influences `trader_pref_score`:

- **Swing trader (default)**
  - Prefers contracts with **healthy |delta|** and **more time to expiry**.
- **Day / short-term trader**
  - Prefers **higher liquidity** and **tighter spreads**, even if DTE is short.

This does **not** override your other filters – it simply nudges similarly
scored trades toward those that match your style.

---

## 💾 Export & Logging Features

### CSV Export
After each scan, optionally export all picks with complete metrics:

**Columns Include:**
- symbol, type, strike, expiration, premium
- delta, IV, HV, IV vs HV
- volume, OI, spread %
- prob_profit, expected_move, prob_touch
- max_loss, breakeven, rr_ratio
- quality_score, liquidity_flag, spread_flag

**Location:** `exports/options_picks_[mode]_[timestamp].csv`

### Trade Logging
Enable P/L tracking by logging entry data:

**Logged Data:**
- Entry timestamp
- Option details (symbol, type, strike, expiry)
- Entry premium & underlying price
- All key metrics (IV, HV, PoP, R/R, quality score)
- Status: OPEN

**Location:** `trades_log/entries.csv`

**Future Use:** Compare entry data with exit data to calculate realized P/L and track strategy performance over time.

---

## 📋 Output Format

### Sample Output:

```
================================================================================
  OPTIONS SCREENER REPORT - AAPL
================================================================================
  Stock Price: $192.50
  Risk-Free Rate: 4.35% (13-week Treasury)
  Expirations Scanned: 4
  DTE Range: 7 - 60 days
  Chain Median IV: 38.5%
  Mode: Single-stock
================================================================================

────────────────────────────────────────────────────────────────────────────────
  LOW PREMIUM (Top 5 Picks)
────────────────────────────────────────────────────────────────────────────────
  Summary: Avg IV 36.2% | Avg Spread 2.1% | Median |Δ| 0.42

  Type  Strike   Exp          Prem     IV      OI       Vol      Δ       Tag
  ----------------------------------------------------------------------------
  CALL   195.00  2025-02-21   $2.85    36.5%     8420    2250    +0.41  OTM
    → liquidity vol 2250, OI 8420; spread 2.1%; delta +0.41 | DTE: 42d
      PoP: 58.3% | R/R: 1.8:1 | Max Loss: $285 | IV vs HV: +2.3%

...

================================================================================
  ⭐ TOP OVERALL PICK
================================================================================

  AAPL CALL | Strike $195.00 | Exp 2025-02-21 (42d) | OTM

  Premium: $2.85
  IV: 36.5% | Delta: +0.41 | Quality: 0.89
  Volume: 2250 | OI: 8420 | Spread: 2.1%
  
  Advanced Metrics:
    • Probability of Profit: 58.3%
    • Risk/Reward Ratio: 1.8:1
    • Max Loss: $285 | Break-even: $197.85
    • Expected Move: ±$8.20
    • IV vs HV: +2.3% (slight premium)

  💡 Rationale: Chosen for excellent liquidity, balanced IV near chain median,
     tight bid-ask spread. Also offers optimal delta range, short-term play.

================================================================================
  SCAN SUMMARY
================================================================================
  Total Picks Displayed: 15
  Chain Median IV: 38.5%
  Expirations Scanned: 4
  Risk-Free Rate Used: 4.35%
  DTE Filter: 7-60 days
  Mode: Single-stock
================================================================================

  ⚠️  Not financial advice. Verify all data before trading.
================================================================================

Export results to CSV? (y/n) [n]: y
  📄 Results exported to: exports/options_picks_Single-stock_20250112_045532.csv

Log trades for P/L tracking? (y/n) [n]: y
  💾 Trade entries logged to trades_log/entries.csv

👋 Done! Happy trading!
```

---

## 🔄 Modes Comparison

| Feature | Single-Stock | Budget Multi-Stock | Discovery Mode |
|---------|--------------|-------------------|----------------|
| **Trigger** | Enter ticker (e.g., "AAPL") | Enter "ALL" | Enter "DISCOVER" or blank |
| **Tickers Scanned** | 1 | 1-10+ (custom) | Up to 100 (top liquid) |
| **Budget Filter** | None | Yes (cost ≤ budget) | None |
| **Categorization** | Quantile (33% splits) | Budget-based (% of budget) | Quantile (33% splits) |
| **Ticker Diversity** | Single ticker | Diversified | Diversified |
| **HV Fetching** | Yes | Yes (per ticker) | Yes (per ticker) |
| **Best Use Case** | Deep dive one stock | Budget-constrained search | Find absolute best opportunities |

---

## ⚠️ Important Disclaimers

### Legal & Compliance
- **Not Financial Advice:** This tool is for educational and informational purposes only
- **Personal Use:** Review Yahoo Finance Terms of Service for data usage limitations
- **No Warranties:** Data accuracy depends on upstream provider; use at your own risk
- **Market Risk:** Options trading involves substantial risk of loss. Never risk more than you can afford to lose.

### Data Limitations
- **Delayed Data:** Yahoo Finance may have 15-20 minute delays
- **Coverage:** Not all tickers have complete options data
- **Accuracy:** Implied volatility, Greeks, and probability calculations are estimates
- **Market Hours:** Data freshness varies outside trading hours
- **Historical Volatility:** 30-day HV may not reflect current market conditions

### Probability Disclaimers
- **Model Risk:** Probability calculations assume log-normal distribution and constant volatility
- **Not Guarantees:** PoP and PoT are statistical estimates, not predictions
- **Market Events:** Black swan events, earnings, news can invalidate probability models
- **Slippage:** Actual fill prices may differ from mid-price used in calculations

### Recommended Validation
Before trading any screened option:
1. ✅ Verify current bid/ask on your broker platform
2. ✅ Check recent news/events affecting the underlying
3. ✅ Confirm expiration date and contract specifications
4. ✅ Review upcoming earnings dates and ex-dividend dates
5. ✅ Assess personal risk tolerance and position sizing
6. ✅ Consider transaction costs and taxes
7. ✅ Validate probability assumptions match your outlook
8. ✅ Review tax implications with a professional

---

## 🐛 Troubleshooting

### "No options expirations available"
- Ticker may not have listed options
- Try a more liquid symbol (e.g., SPY, AAPL, MSLA)

### "No contracts passed filters"
- DTE range may be too restrictive
- Try widening min/max DTE or scanning more expirations
- Check if budget is too low (for budget mode)

### "Could not fetch historical volatility"
- Normal for some tickers with limited history
- Screener will use IV-only scoring

### Slow Performance
- Scanning 50+ tickers takes 2-5 minutes
- Reduce to 20-30 tickers for faster results
- Network speed affects data fetch time

### CSV Export Fails
- Ensure you have write permissions
- Check disk space
- `exports/` directory is auto-created

---

## 📚 Resources

### Options Education
- [CBOE Options Institute](https://www.cboe.com/education/)
- [Options Playbook](https://www.optionsplaybook.com/)
- [TastyTrade Options Basics](https://www.tastytrade.com/concepts-strategies)
- [Option Alpha Probability Guide](https://optionalpha.com/lessons/)

### Technical References
- Black-Scholes Model
- Log-Normal Distribution
- Volatility Surface Analysis
- Greeks and Risk Management

### API Documentation
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

### Market Data Providers (Alternatives)
- [Alpha Vantage](https://www.alphavantage.co/) - Free tier available
- [Polygon.io](https://polygon.io/) - Real-time options data
- [Tradier](https://tradier.com/) - Developer-friendly brokerage API
- [Interactive Brokers TWS API](https://www.interactivebrokers.com/en/trading/ib-api.php)

---

## 📝 License

This project is provided as-is for educational purposes. No warranty or guarantee of fitness for any particular purpose is provided.

**Data Attribution:** Market data provided by Yahoo Finance via yfinance library.

---

## 🤝 Contributing

Enhancements welcome! Areas for contribution:
- Additional probability models (binomial, Monte Carlo)
- More Greeks (gamma, theta, vega, rho)
- Portfolio-level analysis
- Backtesting engine enhancements
- ML-based quality scoring
- Real-time alerts
- Web UI

---

## 📧 Support

For issues or questions:
- Open an issue on GitHub: [Ollie1o1/options](https://github.com/Ollie1o1/options)
- Review the Troubleshooting section above

---

## 🆕 Changelog

### v3.0.0 - Probability-Based Upgrade (2025-11-13)
- ✨ Added EM-based break-even realism scoring (required move vs expected move).
- ✨ Reworked Probability of Profit to use delta + EM band adjustments.
- ✨ Implemented theta decay pressure and a theta risk score.
- ✨ Redesigned Risk/Reward ratio around EM-based target prices.
- ✨ Added 30/90 day IV rank & percentile metrics with IV rank scoring.
- ✨ Introduced momentum analytics (5d return, 14d RSI, ATR trend).
- ✨ Re-enabled robust earnings awareness and catalyst scoring.
- ✨ Replaced the old 6-factor quality score with a composite 0–1 score
  combining PoP, EM realism, RR, momentum, IV rank, liquidity, catalysts,
  theta, EV, and trader style preference.
- ✨ Added day trader vs swing trader profile to influence ranking.
- ✨ Exported all new metrics (EM, realism, theta, momentum, IV ranks, sub-scores)
  in CSV output.

### v2.0.0 - Professional Edition (2025-01-12)
- ✨ Added Probability of Profit calculations
- ✨ Added Expected Move (1 SD) calculations
- ✨ Added Probability of Touch calculations
- ✨ Added Risk/Reward ratio analysis
- ✨ Added 30-day Historical Volatility fetching
- ✨ Added IV vs HV comparison and advantage scoring
- ✨ Added IV Skew calculations (put vs call IV)
- ✨ Added liquidity quality flags (GOOD/FAIR/POOR)
- ✨ Added spread quality flags (OK/WIDE/VERY_WIDE)
- ✨ Enhanced quality scoring with 6-factor weighted algorithm
- ✨ Added CSV export functionality with timestamps
- ✨ Added trade logging for P/L tracking
- ✨ Added max loss and break-even point calculations
- 🔧 Improved output formatting with advanced metrics
- 🔧 Enhanced error handling and traceback reporting

### v1.0.0 - Initial Release
- Basic options screening
- Single/Budget/Discovery modes
- Black-Scholes delta
- Quality scoring
- Terminal-friendly output

---

**Happy Screening! 📊✨**

*Remember: Past performance does not guarantee future results. Probability calculations are estimates based on assumptions that may not hold in real markets. Always perform your own due diligence before trading. Options trading involves significant risk and is not suitable for all investors.*
