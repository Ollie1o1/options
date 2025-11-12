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

2. **Install required dependencies:**
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

## 📈 Advanced Metrics Explained

### 1. Probability of Profit (PoP)
**Formula:** Uses log-normal distribution to calculate probability of being profitable at expiration.

**Interpretation:**
- **50-60%**: Moderate probability
- **60-75%**: Good probability
- **>75%**: High probability (but lower potential reward)

### 2. Expected Move
**Formula:** `Stock Price × IV × √(T)`

Represents 1 standard deviation price movement expected by expiration.

**Example:** Stock at $100, IV 40%, 30 DTE → Expected move ≈ $11

### 3. Probability of Touch
Likelihood that the strike price will be touched/reached before expiration.

**Use Case:** Important for selling options (you want low PoT) or setting profit targets.

### 4. Risk/Reward Ratio
**Calculation:**
- Max Loss = Premium × 100 (per contract)
- Potential Reward = Projected profit at target price
- R/R Ratio = Reward / Risk

**Interpretation:**
- **<1:1**: Unfavorable
- **1:1 to 2:1**: Reasonable
- **>2:1**: Attractive

### 5. Historical Volatility vs Implied Volatility

**IV > HV:** Options are relatively expensive (good for selling)
**IV < HV:** Options are relatively cheap (good for buying)
**IV ≈ HV:** Fairly priced

### 6. IV Skew
**Skew = Put IV - Call IV** (at same strike/expiry)

- **Positive skew**: Puts more expensive (normal, reflects downside risk)
- **Negative skew**: Calls more expensive (unusual, potential opportunity)

---

## 🏆 Quality Scoring System

### Enhanced Weighting (v2.0):
```
Quality Score = 25% Liquidity 
              + 20% IV Advantage 
              + 20% Risk/Reward 
              + 15% Probability of Profit 
              + 10% Spread Tightness 
              + 10% Delta Quality
```

### Component Details:

**Liquidity (25%)**
- Rank-normalized volume and open interest
- Ensures easy entry/exit

**IV Advantage (20%)**
- Prefers IV slightly above HV (but not extreme)
- Indicates good value

**Risk/Reward (20%)**
- Targets R/R ratio > 1.5:1
- Caps at 3:1 to avoid unrealistic scenarios

**Probability of Profit (15%)**
- Balances PoP with reward potential
- Optimal range: 55-70%

**Spread Tightness (10%)**
- Tight spreads = lower transaction costs
- Wide spreads penalized

**Delta Quality (10%)**
- Targets |delta| around 0.40
- Balanced directional exposure

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
