# Options Screener: Decision Matrix & Troubleshooting Guide
**Document ID:** STRAT-REF-2026-06  
**Classification:** Operational Guide  
**Author:** Chief Systems Engineer  

---

## 1. Tactical Command Reference

Use this matrix to choose the right command for your current goal.

| Goal | Command | Why |
| :--- | :--- | :--- |
| **Daily Hunting** | `python run.py --top 15` | Scans all high-liquidity tickers for the best 15 trades. |
| **Ticker Deep-Dive** | `python run.py -s NVDA` | Focuses all scoring logic on a single stock you already like. |
| **Portfolio Review** | `python3 -m src.check_pnl` | Essential before every trade to ensure you aren't "over-leveraged." |
| **Strategy Tuning** | `./scripts/calibration_status.sh` | Check if your machine-learning weights are ready to be updated. |
| **Visual Analysis** | `python run.py --viz` | Opens the 3D Plotly visualizer to see the IV surface. |

---

## 2. Market Regime Decision Matrix

The "best" strategy changes based on the **VIX (Volatility Index)** level.

### VIX < 15 (Low Volatility / "Bull Market")
*   **Optimal Mode:** `Discovery` or `Single-stock`.
*   **Focus:** Long Calls on momentum stocks. Look for `rvol > 1.2`.
*   **Risk:** IV is low, so "IV Crush" risk is minimal, but "Theta Burn" is high if the stock stays flat.

### VIX 15 - 25 (Normal / "Range-bound")
*   **Optimal Mode:** `Credit Spreads` or `Iron Condors`.
*   **Focus:** Selling premium. Look for `VRP Score > 0.60`.
*   **Risk:** Ensure your spreads are wide enough to survive small spikes in volatility.

### VIX > 25 (High Volatility / "Market Panic")
*   **Optimal Mode:** `Long Gamma` or `Premium Selling`.
*   **Focus:** Long Puts (buying protection) or Short Puts (if you have the cash to buy the stock at a discount).
*   **Risk:** Spreads will be wide and fills will be difficult. Cut `max_bid_ask_spread_pct` slack to 0.20.

---

## 3. Troubleshooting: "Why are there 0 results?"

If the screener returns 0 rows, check these three things in order:

1.  **Market Hours:** If it is 8:00 PM ET, the "Bid/Ask" prices are $0.00. The system filters these out to protect you from "bad fills."
    *   *Fix:* Wait for market open OR use the `--no-cache` flag to try forcing a refresh.
2.  **Ticker Liquidity:** You are scanning a stock that nobody trades options on.
    *   *Fix:* Use `--watchlist liquid_large_cap` to ensure you are looking at stocks with enough volume.
3.  **Spread Filter:** Your `max_bid_ask_spread_pct` is too tight (default 0.15).
    *   *Fix:* Check `config.json`. If the market is volatile, you may need to increase this to 0.20 temporarily.

---

## 4. Decision Fatigue: How to pick "The One"

When you have 5 "Top Picks" and only \$1,000 to spend:

1.  **Check Ticker Correlation:** Do you already have a trade in a similar stock? (e.g., If you own MSFT calls, don't buy AAPL calls).
2.  **Compare PoP vs. RR:** 
    *   If you want a "Safe" trade: Pick the one with the highest **PoP** (Probability of Profit).
    *   If you want a "Moonshot": Pick the one with the highest **RR Ratio** and **RVOL**.
3.  **Trust the Technical Score:** If one pick has 5 stars (0.90) and the other has 4 (0.80), always take the 5-star pick. The math has already accounted for the risk.

---

## 5. Maintenance Checklist

*   **Daily:** Run `scripts/enforce_exits.sh` to ensure you don't miss a Profit Target or Stop Loss.
*   **Weekly:** Run `scripts/calibrate_snapshot.sh` to see if your signals are getting stronger or weaker.
*   **Monthly:** Back up your `paper_trades.db` file. This is your most valuable asset—it contains your personal "edge."

---
**END OF REFERENCE**
