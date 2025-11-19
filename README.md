# Options Screener - Professional Edition

A comprehensive Python-based options screening tool designed to identify high-probability trading opportunities by layering advanced analytics, institutional-level metrics, and dynamic safety filters.

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
    git clone <your_repo_url>
    cd <your_repo_directory>
    ```

2.  **(Recommended) Create and activate a virtual environment:**
    ```bash
    python -m venv env
    source env/bin/activate  # On Windows: env\Scripts\activate
    ```

3.  **Install required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

**Run the Screener:**
```bash
python3 src/options_screener.py
```
The script will guide you through interactive prompts to select a mode and configure your scan.

**Run the Backtester:**
```bash
python3 src/backtest_screener.py
```
This script runs automatically, analyzing past screener results saved in the `logs/` directory.

---

## 📊 Features Deep Dive

The screener's power comes from its layered, multi-factor analysis. Features are grouped into four categories:

### 1. The Core Engine (Base Analytics)

These are the foundational metrics calculated for every option.

-   **Multi-Factor Quality Score:** A sophisticated weighted algorithm (configurable in `config.json`) that combines dozens of metrics into a single `quality_score` from 0 to 1. This is the primary sorting mechanism.
-   **Greeks Calculation:** Provides standard Black-Scholes Greeks (`delta`, `gamma`, `vega`, `theta`) to assess risk and sensitivity.
-   **Volatility Analysis:**
    -   Calculates 30-day **Historical Volatility (HV)**.
    -   Compares **Implied Volatility (IV)** to HV to determine if options are statistically cheap or expensive.
    -   Calculates **IV Rank and Percentile** (30 & 90-day) to contextualize the current IV level.
-   **Probability Metrics:**
    -   **Probability of Profit (PoP):** A delta-based, expected-move-aware calculation of the likelihood of the trade being profitable at expiration.
    -   **Expected Move (EM):** The market-implied 1 standard deviation price move until expiration.
-   **Risk/Reward Analysis:** Calculates the break-even price, max loss, and a realistic risk/reward ratio based on a target price derived from the Expected Move.

### 2. Edge-Finding Features (The "Why")

These features are designed to find a statistical "edge" in the market.

-   **Institutional Flow ("Unusual Whales"):** Automatically flags contracts with unusually high `Volume` relative to their `Open Interest` (`Vol_OI_Ratio > 1.5`), indicating potential large, informed traders are entering a position. These trades are prioritized in the output.
-   **Vertical Spread Finder:** Identifies potential debit spreads by pairing "buy" candidates with a suitable "sell" leg 1-2 strikes further OTM, filtering for trades where `Max Profit > 1.5 * Risk`.
-   **Earnings Volatility:** Flags options as an **"Earnings Play"** if an earnings report is due before expiration. For these plays, it checks if the option is **"Underpriced"** by comparing its IV to the stock's recent historical volatility.
-   **Sentiment Analysis:** Fetches recent news headlines for the underlying stock and uses `TextBlob` to perform sentiment analysis, scoring it from -1 (Bearish) to +1 (Bullish).

### 3. Safety Filters (The "Why Not")

These filters act as a defensive layer, penalizing the `quality_score` of trades that have hidden risks.

-   **Market Context Filter:**
    -   At runtime, the screener fetches data for **SPY** and the **VIX**.
    -   It determines the broad **Market Trend** (Bull/Bear) based on SPY's position relative to its 50-day SMA.
    -   It determines the **Volatility Regime** (High/Low) based on whether the VIX is above 20.
    -   This status is printed in the report header for immediate context.
-   **Trend Alignment Filter:** Calculates the stock's 20-day SMA and rewards trades that align with the trend (`+0.15 score` bonus for Calls above the SMA or Puts below it).
-   **Theta (Time Decay) Safety Check:** Calculates the `Theta_Burn_Rate` (daily decay as a % of the option's price). If the decay is too high (`>6%`), the trade is flagged as **"HIGH DECAY RISK"** and its score is heavily penalized (`-0.20 penalty`).
-   **Support/Resistance Warning:** Calculates the 20-day high and low for the stock. If a Call is being bought near the high or a Put is being bought near the low, it's flagged as **"NEAR RESISTANCE"** or **"NEAR SUPPORT"** and its score is penalized (`-0.10 penalty`).

### 4. Probability Enhancers (The "When")

These features look for specific market structures that can increase the probability of a successful trade.

-   **Historical Seasonality Check:** For the current month, it calculates the **`Seasonal_Win_Rate`**—the percentage of times the stock has finished positive over the last 5 years. Trades aligned with strong seasonality get a score bonus (`+0.1`), while those fighting it get a penalty (`-0.1`).
-   **"OI Wall" Detection:** For each expiration, the screener identifies the strike with the highest Call Open Interest (the `Call_Wall`) and Put Open Interest (`Put_Wall`). Trades placed too close to these "walls" are flagged with **"LIMITED UPSIDE"** or **"LIMITED DOWNSIDE"** and their score is penalized (`-0.10 penalty`).
-   **Bollinger Band Squeeze:** The screener calculates Bollinger Bands and Keltner Channels to detect when a stock is in a "squeeze" (low volatility, coiling for a big move). If a squeeze is detected AND institutional whale activity is present, the trade is flagged as a **"🔥 SQUEEZE PLAY"** and receives a massive score bonus (`+0.25 bonus`).

---

## ⚙️ Modes of Operation

The screener can be run in several modes by following the initial prompt:

1.  **Single-Stock Analysis (e.g., `AAPL`):** A deep dive into a single ticker. Best for focused analysis.
2.  **Budget Scan (`ALL`):** Scans a user-defined list of tickers for trades that fall within a specific budget per contract.
3.  **Discovery Scan (`DISCOVER`):** Scans a pre-defined list of the top 100 most liquid tickers to find the absolute best opportunities across the market, without a budget constraint.
4.  **Premium Selling (`SELL`):** A specialized mode that scans for high-probability short put opportunities, using a different set of filters and scoring weights optimized for selling premium.

---

## 🔬 Backtesting Engine

The repository includes a backtesting engine to evaluate the historical performance of the screener's picks.

**Logic: Managed Trade Simulation**

Instead of a simple "hold to expiration" model, the backtester simulates a realistic managed trade strategy:
-   **Entry:** The entry price is the option's premium on the day the screener ran.
-   **Exit Rules:** For each day after entry, the backtester checks if:
    -   The daily `High` hits the profit target (`Entry Price * 1.5`). If so, it's marked as a **WIN**.
    -   The daily `Low` hits the stop-loss (`Entry Price * 0.5`). If so, it's marked as a **LOSS**.
-   **Output:** The backtester calculates the overall **Win Rate %**, total P/L, and other performance statistics based on this strategy.

---

## 📋 Example Output

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
    ↳ Analysis:  IV: 25.0% (≈ median) | PoP: 58.5% | RR: 0.8x | Quality: 0.73 | Nov Hist: 60% | 🔥 SQUEEZE PLAY

...

================================================================================
  ⭐ TOP OVERALL PICK
================================================================================

  AAPL PUT | Strike $265.00 | Exp 2025-12-12 (22d) | OTM

  Premium: $5.53
  IV: 25.0% | Delta: -0.41 | Quality: 0.73
  Volume: 604 | OI: 944 | Spread: 4.5%

  💡 Rationale: Chosen for excellent liquidity, balanced IV, tight spread. Also offers optimal delta range.

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only and does not constitute financial advice. Options trading involves substantial risk and is not suitable for all investors. All data is provided by Yahoo Finance and may be delayed or contain inaccuracies. Always perform your own due diligence before making any trade.
```