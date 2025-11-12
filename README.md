# Options Screener

A Python-based options screening tool that fetches real-time options data and identifies high-quality trading opportunities across different premium price ranges.

## 🎯 Overview

This screener analyzes options chains for any stock ticker and outputs **15 top picks**: 5 low-priced, 5 medium-priced, and 5 high-priced options. Each pick is evaluated on multiple quality dimensions including liquidity, bid-ask spread, delta characteristics, and implied volatility.

**Key Features:**
- Real-time data via Yahoo Finance (yfinance API)
- Automatic risk-free rate fetching from 13-week Treasury
- Multi-factor quality scoring algorithm
- Black-Scholes delta calculations
- User-friendly CLI with sensible defaults
- Automatic premium categorization (low/medium/high)
- Detailed rationale for each pick

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to this repository:**
```bash
cd C:\Users\Oliver\Documents\options
```

2. **Install required dependencies:**
```bash
pip install yfinance pandas
```

### Usage

Run the screener:
```bash
python options_screener.py
```

**Interactive Prompts:**
```
Enter stock ticker (e.g., AAPL): TSLA
How many nearest expirations to scan [4]: 6
Minimum days to expiration (DTE) [7]: 14
Maximum days to expiration (DTE) [120]: 90
Fetching current risk-free rate...
Using risk-free rate: 4.35% (13-week Treasury)
```

**Sample Output:**
```
============ LOW PREMIUM (top 5) ============
TSLA CALL  Strike 250.00  Exp 2025-02-21  Prem $1.25  IV 45.2%  OI   1250  Vol    320  Δ +0.38
  → liquidity vol 320, OI 1250; spread 3.2%; delta +0.38; IV 45.2% (≈ chain median 44.8%); quality 0.82

...

============ MEDIUM PREMIUM (top 5) ============
...

============ HIGH PREMIUM (top 5) ============
...
```

---

## 📊 How It Works

### 1. Data Collection
The script uses **yfinance** to fetch:
- Current underlying stock price
- Current risk-free rate from 13-week Treasury bill (^IRX)
- Options chains (calls and puts) for the nearest N expirations
- Bid/ask prices, volume, open interest, implied volatility, strike prices, expiration dates

### 2. Quality Scoring Algorithm

Each option contract receives a **quality score** (0.0 to 1.0) based on four weighted components:

| Factor | Weight | Description |
|--------|--------|-------------|
| **Liquidity** | 35% | Combined rank-normalized volume and open interest |
| **Spread Tightness** | 25% | Bid-ask spread as % of mid price (tighter = better) |
| **Delta Quality** | 20% | Proximity to target delta of ±0.40 (balanced risk/reward) |
| **IV Quality** | 20% | Moderate IV relative to chain median (avoids extremes) |

**Formula:**
```
Quality Score = 0.35×Liquidity + 0.25×Spread + 0.20×ΔQuality + 0.20×IVQuality
```

#### Component Details

**Liquidity Score:**
- Rank-normalizes volume and open interest across the entire chain
- Filters out completely dead contracts (zero volume AND zero OI)
- Higher volume/OI = easier to enter/exit positions

**Spread Score:**
- Calculates `(Ask - Bid) / Mid` as percentage
- Caps spread at 25% (worse spreads treated equally)
- Tight spreads (<5%) score near 1.0
- Wide spreads (>25%) score near 0.0

**Delta Quality:**
- Computes Black-Scholes delta using underlying price, strike, time to expiration, risk-free rate, and IV
- Targets absolute delta around 0.40 (moderate directional exposure)
- Penalizes deep ITM (delta near ±1.0) and far OTM (delta near 0) contracts

**IV Quality:**
- Identifies contracts with moderate IV relative to the chain median
- Avoids extremely high IV (overpriced risk) and extremely low IV (illiquid/stale)
- Contracts at the median score highest (1.0); extremes score lower

### 3. Premium Categorization

After scoring, contracts are divided into three price buckets by premium:
- **LOW**: Bottom 33rd percentile of premiums
- **MEDIUM**: Middle 33rd percentile
- **HIGH**: Top 33rd percentile

### 4. Selection & Ranking

Within each bucket:
1. Sort by quality score (descending)
2. Tie-breaker: spread percentage (ascending)
3. Tie-breaker: volume (descending)
4. Tie-breaker: open interest (descending)
5. Tie-breaker: time to expiration (ascending - prefer nearer)

Return top 5 from each bucket.

---

## 🔧 Technical Details

### Black-Scholes Delta Calculation

The script computes delta using the standard Black-Scholes formula:

**Call Delta:** `N(d₁)`  
**Put Delta:** `N(d₁) - 1`

Where:
```
d₁ = [ln(S/K) + (r + σ²/2)T] / (σ√T)
```
- **S** = Underlying price
- **K** = Strike price
- **r** = Risk-free rate (annualized)
- **σ** = Implied volatility (annualized)
- **T** = Time to expiration (years)
- **N(·)** = Standard normal cumulative distribution function

### Data Handling

**Premium Calculation:**
- Primary: Mid-price `(Bid + Ask) / 2`
- Fallback: Last traded price if mid is unavailable

**Missing IV Handling:**
- Group median by expiration + type (call/put)
- Chain-wide median as final fallback

**Filtering:**
- Removes contracts outside DTE bounds
- Removes contracts with no premium data
- Removes completely illiquid contracts (zero volume AND zero OI)

### Dependencies

| Package | Purpose |
|---------|---------|
| **yfinance** | Fetch Yahoo Finance options data |
| **pandas** | Data manipulation and analysis |
| **math** (stdlib) | Black-Scholes calculations, erf function |
| **datetime** (stdlib) | Time calculations and DTE filtering |

---

## ⚙️ Configuration

### Adjustable Parameters (via prompts)

| Parameter | Default | Description |
|-----------|---------|-------------|
| Ticker | (required) | Stock symbol (e.g., AAPL, TSLA) |
| Max Expirations | 4 | Number of nearest expiration dates to scan |
| Min DTE | 7 | Minimum days to expiration filter |
| Max DTE | 120 | Maximum days to expiration filter |
| Risk-Free Rate | Auto | Fetched from 13-week Treasury (^IRX), fallback 4.5% |

### Code-Level Customization

**Edit scoring weights** in `enrich_and_score()`:
```python
df["quality_score"] = (
    0.35 * liquidity +      # Adjust liquidity weight
    0.25 * spread_score +   # Adjust spread weight
    0.20 * delta_quality +  # Adjust delta weight
    0.20 * iv_quality       # Adjust IV weight
)
```

**Change delta target** (default 0.40):
```python
delta_target = 0.40  # Line ~180
```

**Adjust picks per bucket** (default 5):
```python
picks = pick_top_per_bucket(df_bucketed, per_bucket=5)  # Line ~447
```

---

## 📈 Interpreting Results

### Output Format
```
TICKER TYPE  Strike XXX.XX  Exp YYYY-MM-DD  Prem $X.XX  IV XX.X%  OI XXXXXX  Vol XXXXXX  Δ ±X.XX
  → liquidity vol XXX, OI XXX; spread X.X%; delta ±X.XX; IV XX.X% (relation chain median); quality X.XX
```

### Key Metrics

| Field | Meaning |
|-------|---------|
| **TYPE** | CALL or PUT |
| **Strike** | Contract strike price |
| **Exp** | Expiration date |
| **Prem** | Premium (mid-price or last) |
| **IV** | Implied volatility (annualized %) |
| **OI** | Open interest (existing contracts) |
| **Vol** | Today's volume traded |
| **Δ** | Delta (rate of change vs underlying) |
| **Quality** | Composite score (0.0–1.0) |

### Delta Interpretation

| Delta Range | Call Meaning | Put Meaning |
|-------------|--------------|-------------|
| **0.0 to 0.2** | Far OTM | Deep ITM |
| **0.2 to 0.4** | OTM | ITM |
| **0.4 to 0.6** | Near/At-the-money | Near/At-the-money |
| **0.6 to 0.8** | ITM | OTM |
| **0.8 to 1.0** | Deep ITM | Far OTM |

**Target (±0.40):** Balanced directional exposure with reasonable time value.

---

## ⚠️ Important Disclaimers

### Legal & Compliance
- **Not Financial Advice:** This tool is for educational and informational purposes only
- **Personal Use:** Review Yahoo Finance Terms of Service for data usage limitations
- **No Warranties:** Data accuracy depends on upstream provider; use at your own risk
- **Market Risk:** Options trading involves substantial risk of loss

### Data Limitations
- **Delayed Data:** Yahoo Finance may have 15-20 minute delays
- **Coverage:** Not all tickers have complete options data
- **Accuracy:** Implied volatility and Greeks are estimates
- **Market Hours:** Data freshness varies outside trading hours

### Recommended Validation
Before trading any screened option:
1. ✅ Verify current bid/ask on your broker platform
2. ✅ Check recent news/events affecting the underlying
3. ✅ Confirm expiration date and contract specifications
4. ✅ Assess personal risk tolerance and position sizing
5. ✅ Review tax implications with a professional

---

## 🐛 Troubleshooting

### Common Issues

**"No options expirations available"**
- Ticker may not have listed options
- Try a more liquid symbol (e.g., SPY, AAPL, TSLA)

**"No contracts passed filters"**
- DTE range may be too restrictive
- Try widening min/max DTE or scanning more expirations

**"Missing dependencies"**
- Run: `pip install yfinance pandas`
- Ensure Python 3.7+ is installed

**Slow Performance**
- Scanning 10+ expirations takes longer
- Reduce to 3-6 expirations for faster results
- Network speed affects data fetch time

**Invalid Ticker Error**
- Ensure ticker is alphanumeric (no special characters)
- Use proper exchange suffix if needed (e.g., "BRK-B" won't work, use "BRK.B")

---

## 🔮 Future Enhancements

Potential features for future versions:
- [ ] Export results to CSV/JSON
- [ ] Historical backtesting mode
- [ ] Greeks calculations (gamma, theta, vega)
- [ ] Multi-ticker batch processing
- [ ] Web UI with charts
- [ ] Integration with brokerage APIs (Tradier, Interactive Brokers)
- [ ] Custom scoring formula builder
- [ ] Alert system for new opportunities
- [ ] Machine learning quality predictor

---

## 📚 Resources

### Options Education
- [CBOE Options Institute](https://www.cboe.com/education/)
- [Options Playbook](https://www.optionsplaybook.com/)
- [TastyTrade Options Basics](https://www.tastytrade.com/concepts-strategies)

### API Documentation
- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [pandas Documentation](https://pandas.pydata.org/docs/)

### Market Data Providers (Alternatives)
- [Alpha Vantage](https://www.alphavantage.co/) - Free tier available
- [Polygon.io](https://polygon.io/) - Real-time options data
- [Tradier](https://tradier.com/) - Developer-friendly brokerage API

---

## 📝 License

This project is provided as-is for educational purposes. No warranty or guarantee of fitness for any particular purpose is provided.

**Data Attribution:** Market data provided by Yahoo Finance via yfinance library.

---

## 🤝 Contributing

Feel free to fork this repository and submit pull requests with improvements:
- Enhanced scoring algorithms
- Additional data sources
- Better error handling
- Performance optimizations
- Documentation improvements

---

## 📧 Support

For issues or questions:
- Open an issue on GitHub: [Ollie1o1/options](https://github.com/Ollie1o1/options)
- Review the Troubleshooting section above

---

**Happy Screening! 📊✨**

*Remember: Past performance does not guarantee future results. Always perform your own due diligence before trading.*
