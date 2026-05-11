# Quantitative Options Strategy: Technical Reference Manual
**Document ID:** STRAT-MAN-2026-05  
**Classification:** Confidential - Proprietary Strategy  
**Author:** Chief Systems Engineer & Lead Quant Researcher  

---

## 1. Executive Summary
This manual defines the operational framework for the Quantitative Options Screener. The system is engineered to identify structural mispricings in the options market by evaluating 27 distinct mathematical dimensions. The objective is to achieve positive expectancy through asymmetric risk-to-reward profiles, leveraging volatility compression and institutional flow signals.

## 2. Core Alpha Signals and Market Mechanics

### 2.1 Volatility Risk Premium (VRP)
VRP is the spread between Implied Volatility (IV) and Realized Volatility (RV). Historically, market makers overprice options to account for "gap risk." 
*   **The Edge:** Our current calibration identifies `vrp` as the highest-conviction signal (IC = +0.30). 
*   **Application:** When IV is significantly higher than recent price movement (RV), the "VRP Score" increases. Conversely, when IV is compressed below historical averages, the system flags a potential "Value Buy" for directional long positions.

### 2.2 Relative Volume (RVOL)
RVOL measures current contract volume against its 30-day moving average.
*   **The "Whale" Threshold:** An RVOL > 1.5 indicates institutional activity. Large block trades from hedge funds and institutional desks leave footprints in the options chain that retail volume cannot replicate.
*   **Significance:** High RVOL combined with a Technical Score > 0.75 indicates high-conviction positioning for an impending directional breakout.

### 2.3 Volatility Compression (The Squeeze)
The system monitors the relationship between Bollinger Bands and Keltner Channels.
*   **The Squeeze:** When Bollinger Bands move inside Keltner Channels (`is_squeezing: True`), price energy is coiled.
*   **BB Width Percentile:** We rank current band width against a 252-day rolling history. A `bb_width_pct` < 0.10 indicates that the stock is tighter than 90% of the past year, making it a prime candidate for a "Long Gamma" explosive move.

## 3. The Machine Learning Engine: Calibration
The system is self-optimizing. It uses a recursive feedback loop to refine its scoring weights based on actual performance data stored in `paper_trades.db`.

*   **Correlation Analysis:** Upon reaching 200 closed trades, the system calculates the Pearson correlation (Information Coefficient) between specific signals and realized PnL.
*   **Weight Adaptation:** If a signal (e.g., Sentiment) loses predictive power in a new market regime, the calibration tool automatically reduces its weight in `config.json` and redistributes that "influence" to more successful signals like VRP or Momentum.

## 4. Understanding the Options Greeks

The "Greeks" are the mathematical derivatives used to manage risk and predict price movement.

| Greek | Technical Definition | Quant Strategy / "The Edge" |
| :--- | :--- | :--- |
| **Delta ($\Delta$)** | Sensitivity to underlying price. | Used as a proxy for the % probability of the option expiring ITM. |
| **Gamma ($\Gamma$)** | Rate of change of Delta. | Explodes for ATM options near expiry. We hunt Gamma in "Squeeze" plays. |
| **Theta ($\Theta$)** | Sensitivity to time decay. | The "Silent Killer." Decay is non-linear and accelerates at 21 days to expiry. |
| **Vega ($\nu$)** | Sensitivity to volatility. | High Vega means the option price will crash if VIX drops (IV Crush). |
| **Rho ($\rho$)** | Sensitivity to interest rates. | Factored into pricing via the Risk-Free Rate (RFR) for precision. |

## 5. Visualizer and Dashboard Metrics

The screener surfaces several derived metrics used for final trade validation:

*   **PoP (Probability of Profit):** Derived from 5,000+ Monte Carlo simulations of the stock's path. We target trades with PoP > 55%.
*   **EV (Expected Value):** The mathematical average outcome of the trade. `EV = (Win% * Profit) - (Loss% * Loss)`. We only execute trades with positive EV.
*   **Term Structure:** Compares near-term vs. long-term IV. Inversion often precedes major market shifts or binary events.

## 6. Trade Execution and Exit Protocols

### 6.1 Entry Gating (The 4-Point Checklist)
Professional execution requires strict adherence to these gates:
1.  **Technical Score > 0.75:** Alignment with calibrated machine learning weights.
2.  **RVOL > 1.2:** Confirmation of institutional or "smart money" flow.
3.  **Spread Pct < 15%:** Minimizing the immediate "slippage" loss upon entry.
4.  **DTE 21–45:** Avoiding the "Theta Cliff" while allowing time for the thesis to develop.

### 6.2 Small Account Risk Management (Sub-$1,000 Strategy)
For accounts focusing on single-contract entries:
*   **Hard Stop-Loss:** At -50% loss, the trade is closed mechanically. Preserving the remaining 50% for the next "Top Pick" is mathematically superior to holding a "dying" contract.
*   **House Money Protocol:** At +100% gain, sell half the position (or close and re-enter at a lower cost basis). This removes all initial risk from the trade.
*   **Delta 80 Exit:** If an option reaches 0.80 Delta, it no longer provides significant leverage. Close and take profits.

### 6.3 The 21-DTE "Hard Exit"
Time decay is your greatest enemy. When an option has fewer than 21 days to expiration, the cost of holding it increases significantly every day. Unless a stock is in a vertical parabolic move, **all long positions must be closed at or before 21 DTE.**

---
**END OF DOCUMENT**
