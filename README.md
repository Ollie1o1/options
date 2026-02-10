# Options Screener - Professional Edition

A comprehensive Python-based options screening tool designed to identify high-probability trading opportunities by layering advanced analytics, institutional-level metrics, and dynamic safety filters. **Now with a professional Streamlit web dashboard and automated Paper Trading!**

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

**Run the screener in terminal mode:**
```bash
python -m src.options_screener
```

> **Important:** The screener must be run as a module using `-m` because it uses relative imports.

The script will guide you through interactive prompts. On startup, it will automatically check for any Take Profit or Stop Loss hits in your Paper Portfolio.

#### 🌐 Web Dashboard

**Launch the Streamlit web interface:**
```bash
python -m src.options_screener --ui
```

Your browser will open a professional dashboard featuring:
- **Real-time market context** (SPY trend, VIX regime, macro indicators).
- **Options Scanner**: High-performance vectorized Greeks and scoring.
- **Paper Portfolio**: Track forward-tests with real-time P/L and automated exit rules.
- **Visualizer**: Payoff diagrams and quality radar charts.

---

## 📊 Features Deep Dive

### 1. High-Performance Analytics
-   **Vectorized Engine:** Completely refactored math core using **NumPy** and **SciPy** for batch processing of entire options chains at 100x speed vs legacy row-by-row logic.
-   **Intelligent Caching:** Implemented `requests-cache` with a 15-minute expiry to minimize network latency and respect data provider limits.
-   **Multi-Factor Scoring:** Dynamic algorithm combining Greeks, Probability of Profit (PoP), Expected Value (EV), and Trend Alignment.

### 2. Forward Testing & Paper Trading (NEW)
-   **One-Click Logging:** Instantly log setups to a local paper portfolio directly from the CLI or Web Dashboard.
-   **Automated Position Tracking:** The system fetches real-time quotes for open positions every time you start the app.
-   **Exit Rule Enforcement:** Automatically "closes" trades when Take Profit or Stop Loss thresholds are hit.
-   **Performance Summaries:** Tracks Win Rate, Total P/L, and Average Return over time.

---

## 🗂️ Project Structure

```
options/
├── src/
│   ├── options_screener.py    # Core engine & CLI entrypoint
│   ├── dashboard.py            # Streamlit web interface
│   ├── data_fetching.py        # Optimized API wrappers with caching
│   ├── paper_manager.py        # Forward testing & portfolio logic
│   ├── filters.py              # Configuration-driven filtering
│   ├── utils.py                # Vectorized math & formatting helpers
│   ├── simulation.py           # Monte Carlo probability engine
│   └── ...
├── config.json                 # Centralized thresholds & exit rules
└── requirements.txt            # Project dependencies
```

---

## ⚙️ Configuration

### Filtering Thresholds & Exit Rules
All "magic numbers" are externalized in `config.json`:

```json
{
  "filters": {
    "min_volume": 50,
    "max_bid_ask_spread_pct": 0.40,
    "delta_min": 0.15,
    "delta_max": 0.35
  },
  "exit_rules": {
    "take_profit": 0.50,
    "stop_loss": -0.25
  }
}
```

---

## 🛣️ Roadmap

- [x] High-performance vectorized math core
- [x] Intelligent request caching (15m expiry)
- [x] **Forward Testing & Paper Trading manager**
- [x] Streamlit dashboard with integrated Portfolio tracking
- [ ] Strategy backtesting UI improvements
- [ ] Real-time Email/SMS alerts
- [ ] Multi-leg spread support in Paper Manager

---

## ⚠️ Disclaimer

This tool is for educational and informational purposes only and does not constitute financial advice. Options trading involves substantial risk. Always perform your own due diligence.

---

## 📄 License

This project is for personal use. Not licensed for commercial distribution.
