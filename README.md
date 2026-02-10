# Options Screener - Professional Edition

A comprehensive Python-based options screening tool designed to identify high-probability trading opportunities by layering advanced analytics, institutional-level metrics, and dynamic safety filters. **Now with a professional Streamlit web dashboard!**

## 🎯 Core Philosophy: Hunting for Alignment

This is not just a tool that finds options with high volume. It's an intelligent screener built on a core philosophy of **Alignment**. A high-quality trade occurs when multiple independent factors align:

1.  **Action:** The market is interested (high volume, tight spreads).
2.  **Edge:** The probabilities are in your favor (IV vs. HV, seasonality, risk/reward).
3.  **Structure:** The trade is not positioned against a major technical barrier (support/resistance, OI Walls).
4.  **Trend:** The trade is aligned with the underlying stock's momentum and the broader market context.

This screener finds opportunities where these forces align, giving you a statistical edge. Consistency comes from saying "No" to good volume on a bad chart.

---

## 🚀 Quick Start

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Ollie1o1/options.git
    cd options
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv env
    
    # On Windows:
    env\Scripts\activate
    
    # On macOS/Linux:
    source env/bin/activate
    ```

3.  **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

#### 🖥️ CLI Mode (Terminal Interface)

**Run the screener in terminal mode (legacy interface):**
```bash
python -m src.options_screener
```

> **Important:** The screener must be run as a module using `-m` because it uses relative imports. Running `python src/options_screener.py` directly will cause an ImportError.

The script will guide you through interactive prompts to select a mode and configure your scan.

#### 🌐 Web Dashboard (NEW!)

**Launch the Streamlit web interface:**
```bash
python -m src.options_screener --ui
```

Your browser will automatically open to `http://localhost:8501` with a professional dashboard featuring:
- **Real-time market context** (SPY trend, VIX regime, macro indicators)
- **Interactive configuration** with sidebar controls
- **Advanced weight tuning** to customize the scoring algorithm
- **Three analysis tabs**: Results viewer, dynamic filtering, portfolio manager
- **CSV export** and in-app trade log editing

**Or launch Streamlit directly:**
```bash
streamlit run src/dashboard.py
```

#### 📊 Backtesting

**Run the backtester:**
```bash
python -m src.backtest_screener
```
This script runs automatically, analyzing past screener results saved in the `logs/` directory.

---

## ⚙️ Modes of Operation

The screener supports multiple scanning strategies:

1.  **Single-Stock Analysis** (e.g., `AAPL`): Deep dive into a single ticker with comprehensive metrics.
2.  **Discovery Scan** (`DISCOVER`): Scans top liquid tickers to find the best market-wide opportunities.
3.  **Budget Scan** (`ALL`): Filters options within a specified budget constraint per contract.
4.  **Premium Selling** (`SELL`): Optimized for high-probability short put opportunities.
5.  **Credit Spreads** (`SPREADS`): Finds bull put and bear call credit spread setups.
6.  **Iron Condor** (`IRON`): Identifies range-bound premium collection opportunities.
7.  **Portfolio Viewer** (`PORTFOLIO`): View open position P/L tracking.

### Mode Selection

**CLI Mode:**
- Follow interactive prompts after running `python -m src.options_screener`

**Web Dashboard:**
- Select from dropdown in sidebar
- Configure filters: Min/Max DTE, Max Expirations
- Adjust trader profile: scalp, swing, long-term
- Tune scoring weights with advanced sliders

---

## 📊 Features Deep Dive

The screener's power comes from its layered, multi-factor analysis. Features are grouped into four categories:

### 1. The Core Engine (Base Analytics)

These are the foundational metrics calculated for every option.

-   **Multi-Factor Quality Score:** A sophisticated weighted algorithm (configurable in `config.json` or via UI sliders) that combines dozens of metrics into a single `quality_score` from 0 to 100. This is the primary sorting mechanism.
-   **Greeks Calculation:** Provides standard Black-Scholes Greeks (`delta`, `gamma`, `vega`, `theta`) to assess risk and sensitivity.
-   **Volatility Analysis:**
    -   Calculates 30-day **Historical Volatility (HV)**.
    -   Compares **Implied Volatility (IV)** to HV to determine if options are statistically cheap or expensive.
    -   Calculates **IV Rank and Percentile** (30 & 90-day) to contextualize the current IV level.
-   **Probability Metrics:**
    -   **Probability of Profit (PoP):** A delta-based, expected-move-aware calculation of the likelihood of the trade being profitable at expiration.
    -   **Expected Move (EM):** The market-implied 1 standard deviation price move until expiration.
    -   **Monte Carlo Simulation:** Uses 10,000 simulations for enhanced PoP accuracy.
-   **Risk/Reward Analysis:** Calculates the break-even price, max loss, and a realistic risk/reward ratio based on a target price derived from the Expected Move.
-   **Expected Value (EV):** Theoretical edge calculation incorporating probabilities and spread costs.

### 2. Edge-Finding Features (The "Why")

These features are designed to find a statistical "edge" in the market.

-   **Institutional Flow ("Unusual Whales"):** Automatically flags contracts with unusually high `Volume` relative to their `Open Interest` (`Vol_OI_Ratio > 1.5`), indicating potential large, informed traders are entering a position. These trades are prioritized in the output.
-   **Spread Builders:**
    -   **Vertical Spread Finder:** Identifies potential debit spreads.
    -   **Credit Spread Engine:** Bull put and bear call spread opportunities.
    -   **Iron Condor Scanner:** Multi-leg neutral strategies with delta neutrality checks.
-   **Earnings Volatility:** Flags options as an **"Earnings Play"** if an earnings report is due before expiration. Checks if the option is **"Underpriced"** by comparing IV to recent HV.
-   **Sentiment Analysis:** Fetches recent news headlines and uses `TextBlob` for sentiment scoring from -1 (Bearish) to +1 (Bullish).
-   **Term Structure Analysis:** Compares front-month vs back-month IV to detect volatility skew opportunities.

### 3. Safety Filters (The "Why Not")

These filters act as a defensive layer, penalizing trades with hidden risks.

-   **Market Context Filter:**
    -   Fetches real-time **SPY** and **VIX** data.
    -   Determines **Market Trend** (Bull/Bear) based on SPY's 50-day SMA.
    -   Determines **Volatility Regime** (High/Low) based on VIX threshold (>20).
    -   **Macro Risk Detection:** Monitors DXY volatility for forex instability.
    -   **Yield Spike Guard:** Tracks 10Y Treasury (^TNX) for bond market stress.
-   **Trend Alignment Filter:** Rewards trades aligned with the stock's 20-day SMA trend.
-   **Theta (Time Decay) Safety Check:** Flags high decay risk (>6% daily) with score penalty.
-   **Support/Resistance Warning:** Penalizes trades near 20-day technical barriers.
-   **Portfolio Protection:** Correlation analysis warns against redundant positions (correlation >0.80).

### 4. Probability Enhancers (The "When")

These features look for specific market structures that increase success probability.

-   **Historical Seasonality Check:** Calculates monthly win rate over 5 years, adjusts scores for seasonal alignment.
-   **"OI Wall" Detection:** Identifies highest Call/Put Open Interest strikes (pin risk zones).
-   **Bollinger Band Squeeze:** Detects low volatility coiling + whale activity for explosive move setups.
-   **Max Pain Calculation:** Identifies the strike price where option sellers profit most.
-   **Sector Relative Strength:** Compares stock performance to its sector ETF.
-   **Institutional Metrics:**
    -   **Short Interest:** Tracks days to cover for squeeze potential.
    -   **RVOL (Relative Volume):** Abnormal volume detection.
    -   **GEX Flip Price:** Gamma exposure inflection point.
    -   **VWAP:** Volume-weighted average price for intraday context.

---

## 🎨 Web Dashboard Features

### Market Context (Always Visible)
- **SPY Trend:** Bull/Bear market detection
- **Volatility Regime:** VIX-based regime identification
- **Macro Risk:** Currency volatility alerts
- **10Y Yield:** Bond market stress indicator

### Sidebar Controls
- **Scan Mode Selection:** 6 specialized scanning strategies
- **Dynamic Inputs:** Context-aware ticker/budget/count inputs
- **Filter Sliders:** Min/Max DTE, Max Expirations
- **Trader Profile:** Scalp/Swing/Long-term optimization

### Advanced Weight Tuning
Customize the quality score algorithm in real-time:
- Probability of Profit (PoP)
- Expected Move Realism
- Risk/Reward Ratio
- Momentum Score
- Liquidity Score
- Catalyst Bonus
- Theta Decay
- Expected Value
- Trader Preference

### Analysis Tabs

**Tab 1: Results**
- Interactive dataframe with formatted columns
- Sortable and filterable display
- Download button for CSV export
- Scan timestamp and underlying price display

**Tab 2: Dynamic Filtering**
- Quality Score slider for post-scan filtering
- Filter results without re-running expensive scan
- Delta count visualization

**Tab 3: Portfolio Manager**
- Load `trades_log/entries.csv` in editable view
- Summary metrics: Open positions, Closed positions, Total P/L
- Inline editing with save functionality
- Track unrealized P/L

---

## 🔬 Backtesting Engine

The repository includes a backtesting engine to evaluate historical performance.

**Logic: Managed Trade Simulation**

Instead of a simple "hold to expiration" model, the backtester simulates a realistic managed trade strategy:
-   **Entry:** The entry price is the option's premium on the day the screener ran.
-   **Exit Rules:** For each day after entry, checks if:
    -   Daily `High` hits profit target (Entry × 1.5) → **WIN**
    -   Daily `Low` hits stop-loss (Entry × 0.5) → **LOSS**
-   **Output:** Overall Win Rate %, total P/L, and performance statistics.

---

## 📋 Example CLI Output

```
================================================================================
  OPTIONS SCREENER REPORT - AAPL
================================================================================
  Stock Price: $267.44
  Market Status: Trend is Bear | Volatility is High
  Risk-Free Rate: 3.77% (13-week Treasury)
  Expirations Scanned: 4
  DTE Range: 7 - 120 days
  Chain Median IV: 25.3%
  Mode: Single-stock
================================================================================

────────────────────────────────────────────────────────────────────────────────
  LOW PREMIUM (Top 5 Picks)
────────────────────────────────────────────────────────────────────────────────
  Summary: Avg IV 25.2% | Avg Spread 4.3% | Median |Δ| 0.47

  Whale Type  Strike   Exp          Prem     IV      OI       Vol      Δ       Tag
  -------------------------------------------------------------------------------
  🐋    PUT    265.00 2025-12-12 $5.53    25.0%       944     604  -0.41 OTM
    ↳ Mechanics: Vol: 604 OI: 944 | Spread: 4.5% | Delta: -0.41 | Cost: $552.50
    ↳ Analysis:  IV: 25.0% (≈ median) | PoP: 58.5% | RR: 0.8x | Quality: 73.2 | 🔥 SQUEEZE PLAY

================================================================================
  ⭐ TOP OVERALL PICK
================================================================================

  AAPL PUT | Strike $265.00 | Exp 2025-12-12 (22d) | OTM

  Premium: $5.53
  IV: 25.0% | Delta: -0.41 | Quality: 73.2
  Volume: 604 | OI: 944 | Spread: 4.5%

  💡 Rationale: Excellent liquidity, balanced IV, tight spread, optimal delta range.
```

---

## 📦 Dependencies

Core requirements (installed via `pip install -r requirements.txt`):
- `yfinance` - Market data fetching
- `pandas` - Data manipulation
- `numpy` - Numerical computing
- `finvizfinance` - Market screener integration
- `textblob` - Sentiment analysis
- `streamlit` - Web dashboard framework
- `watchdog` - File system monitoring

---

## 🗂️ Project Structure

```
options/
├── src/
│   ├── options_screener.py    # Main CLI engine (refactored for headless mode)
│   ├── dashboard.py            # Streamlit web interface (NEW)
│   ├── data_fetching.py        # Market data API wrappers
│   ├── backtest_screener.py    # Historical performance analysis
│   ├── check_pnl.py            # P/L tracking utilities
│   └── visualize_results.py    # Chart generation (optional)
├── trades_log/
│   └── entries.csv             # Trade log (editable via dashboard)
├── logs/                       # JSONL scan results for backtesting
├── exports/                    # CSV exports
├── config.json                 # Scoring weights and settings
└── requirements.txt            # Python dependencies
```

---

## ⚙️ Configuration

### Filtering Thresholds

Magic numbers and filtering criteria are now externalized in `config.json` under the `"filters"` section:

```json
{
  "filters": {
    "min_volume": 50,
    "min_open_interest": 10,
    "max_bid_ask_spread_pct": 0.40,
    "min_days_to_expiration": 7,
    "max_days_to_expiration": 45,
    "min_days_to_expiration_iron": 30,
    "max_days_to_expiration_iron": 60,
    "delta_min": 0.15,
    "delta_max": 0.35,
    "min_iv_percentile": 20
  }
}
```

These values are used by the new `filter_options` utility in `src/filters.py` and across the scoring engine.

### Scoring Weight Customization

Edit `config.json` to adjust the composite quality score calculation:

```json
{
  "composite_weights": {
    "pop": 0.18,           // Probability of Profit
    "em_realism": 0.12,    // Expected Move Realism
    "rr": 0.15,            // Risk/Reward Ratio
    "momentum": 0.10,      // Price momentum
    "liquidity": 0.15,     // Volume/OI quality
    "catalyst": 0.05,      // Earnings/events
    "theta": 0.10,         // Time decay
    "ev": 0.05,            // Expected value
    "trader_pref": 0.10    // Profile alignment
  }
}
```

**Or use the Web Dashboard sliders** to tune weights interactively without editing JSON!

---

## 🚨 Troubleshooting

### Import Errors
```bash
# ❌ Wrong: Direct script execution
python src/options_screener.py

# ✅ Correct: Module execution
python -m src.options_screener
```

### Streamlit Not Found
```bash
# Install/update streamlit
pip install --upgrade streamlit
```

### Virtual Environment Issues
```bash
# Recreate virtual environment
deactivate
rm -rf env
python -m venv env
env\Scripts\activate  # Windows
pip install -r requirements.txt
```

---

## 🛣️ Roadmap

- [x] Multi-factor quality scoring engine
- [x] Institutional flow detection
- [x] Spread builders (debit/credit/iron condor)
- [x] Advanced safety filters (macro risk, yield spike, max pain)
- [x] Portfolio protection (correlation analysis)
- [x] **Streamlit web dashboard with weight tuning**
- [x] **Portfolio manager CSV editor**
- [ ] Real-time P/L tracking integration
- [ ] Email/SMS alerts for high-quality setups
- [ ] Options chain heatmaps
- [ ] Strategy backtesting UI
- [ ] Mobile-responsive dashboard

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only and does not constitute financial advice. Options trading involves substantial risk and is not suitable for all investors. All data is provided by Yahoo Finance and may be delayed or contain inaccuracies. Always perform your own due diligence before making any trade.

---

## 📄 License

This project is for personal use. Not licensed for commercial distribution.

---

## 🤝 Contributing

This is a personal project, but feedback and bug reports are welcome via GitHub Issues.