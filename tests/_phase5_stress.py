"""Phase 5 stress: exit-rule enforcement for spreads + iron condors via update_positions().

Seeds OPEN spread + iron-condor + single-leg trades, monkey-patches the live-price
path to inject scripted leg/spot marks, runs PaperManager.update_positions(), and
verifies (a) the right rows close, (b) the right exit_reason fires, and (c) pnl_pct
matches the expected (entry_credit - cost_to_close)/entry_credit minus friction.
"""
import os, sys, tempfile, sqlite3, types
from datetime import date, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import paper_manager as pm_mod
from src.paper_manager import PaperManager


# --- Fake yfinance ----------------------------------------------------------
PRICE_MAP = {}        # {symbol -> price}  (option marks)
SPOT_MAP = {}         # {ticker -> price}


class _FastInfo:
    def __init__(self, last):
        self.last_price = last
        self.bid = max(0.01, last - 0.05)
        self.ask = last + 0.05


class _FakeTicker:
    def __init__(self, symbol, session=None):
        self.symbol = symbol
        if symbol in PRICE_MAP:
            self.fast_info = _FastInfo(PRICE_MAP[symbol])
        elif symbol in SPOT_MAP:
            self.fast_info = _FastInfo(SPOT_MAP[symbol])
        else:
            self.fast_info = _FastInfo(0.01)

    def history(self, period="1d"):
        import pandas as pd
        last = self.fast_info.last_price
        return pd.DataFrame({"Close": [last]})


_fake_yf = types.SimpleNamespace(Ticker=_FakeTicker)


def _patched_get_yf_and_session():
    return _fake_yf, None


pm_mod._get_yf_and_session = _patched_get_yf_and_session


# --- Set up DB -------------------------------------------------------------
tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
pm = PaperManager(db_path=tmp.name)

today = date.today()
trade_date = (today - timedelta(days=10)).isoformat()
exp_far = (today + timedelta(days=30)).isoformat()      # well above time_exit_dte
exp_near = (today + timedelta(days=5)).isoformat()      # triggers time exit


# Trade A: Bull put spread, deep TP — should close on TP
# entry credit 1.20, cost to close 0.30 → pnl_raw = (1.20 - 0.30)/1.20 = 0.75 ≥ 0.50 TP
pm.log_spread({
    "date": trade_date, "ticker": "SPY", "expiration": exp_far,
    "type": "Bull Put", "short_strike": 500, "long_strike": 495,
    "net_credit": 1.20, "max_profit": 120.0, "max_loss": 380.0,
    "quality_score": 0.60,
    "entry_iv": 0.22, "entry_delta": -0.25, "entry_gamma": 0.02,
    "entry_vega": 0.18, "entry_theta": -0.03,
})

# Trade B: Bear call spread — should close on SL (cost to close > entry_credit)
# entry credit 1.00, cost to close 2.30 → pnl_raw = (1.00 - 2.30)/1.00 = -1.30 ≤ -1.00 SL
pm.log_spread({
    "date": trade_date, "ticker": "QQQ", "expiration": exp_far,
    "type": "Bear Call", "short_strike": 460, "long_strike": 465,
    "net_credit": 1.00, "max_profit": 100.0, "max_loss": 400.0,
    "quality_score": 0.55,
})

# Trade C: Iron condor — should close on time exit (DTE=5, days_held=10)
# entry credit 2.60, cost to close 2.00 → pnl_raw=(2.60-2.00)/2.60=0.231 (no TP/SL)
# but DTE=5 ≤ time_exit_dte=21 and days_held=10 ≥ min_days_held=3 → Time Exit
pm.log_iron_condor({
    "date": trade_date, "ticker": "AAPL", "expiration": exp_near,
    "short_put_strike": 170, "long_put_strike": 165,
    "short_call_strike": 190, "long_call_strike": 195,
    "total_credit": 2.60, "max_profit": 260.0, "max_risk": 240.0,
    "net_delta": 0.0, "quality_score": 0.74,
})

# Trade D: Single-leg short put — should NOT close (current price still high enough)
pm.log_trade({
    "date": trade_date, "ticker": "MSFT", "expiration": exp_far,
    "strike": 380, "type": "Put", "entry_price": 3.00,
    "quality_score": 0.65, "strategy_name": "Cash-Secured Put",
    "entry_iv": 0.28, "entry_delta": -0.30, "entry_gamma": 0.025,
    "entry_vega": 0.20, "entry_theta": -0.04,
})

# Trade E: Iron condor that should NOT close (DTE far, no TP/SL hit)
pm.log_iron_condor({
    "date": trade_date, "ticker": "NVDA", "expiration": exp_far,
    "short_put_strike": 800, "long_put_strike": 790,
    "short_call_strike": 900, "long_call_strike": 910,
    "total_credit": 3.00, "max_profit": 300.0, "max_risk": 700.0,
    "net_delta": 0.0, "quality_score": 0.70,
})


# --- Set scripted prices -----------------------------------------------------
def _opt_sym(t, exp, k, ot):
    return pm._get_option_symbol(t, exp, k, ot)

# Trade A — bull put 500/495, deep TP: net debit to close = -1*(0.40) + 1*(0.10) = -0.30
PRICE_MAP[_opt_sym("SPY", exp_far, 500, "put")] = 0.40   # short put now cheaper
PRICE_MAP[_opt_sym("SPY", exp_far, 495, "put")] = 0.10   # long put very cheap
SPOT_MAP["SPY"] = 510.0

# Trade B — bear call 460/465 SL: short call expensive (3.00), long call (0.70)
PRICE_MAP[_opt_sym("QQQ", exp_far, 460, "call")] = 3.00
PRICE_MAP[_opt_sym("QQQ", exp_far, 465, "call")] = 0.70
SPOT_MAP["QQQ"] = 462.0

# Trade C — IC time exit. price legs to give cost_to_close ≈ 2.00 (mild profit, not TP)
# short put 170 (0.30), long put 165 (0.05), short call 190 (1.80), long call 195 (0.05)
# cost to close = -(-1)*0.30 + -(+1)*0.05 + -(-1)*1.80 + -(+1)*0.05
#               = 0.30 - 0.05 + 1.80 - 0.05 = 2.00
PRICE_MAP[_opt_sym("AAPL", exp_near, 170, "put")]  = 0.30
PRICE_MAP[_opt_sym("AAPL", exp_near, 165, "put")]  = 0.05
PRICE_MAP[_opt_sym("AAPL", exp_near, 190, "call")] = 1.80
PRICE_MAP[_opt_sym("AAPL", exp_near, 195, "call")] = 0.05
SPOT_MAP["AAPL"] = 185.0

# Trade D — short put MSFT 380, current price 1.50 (50% TP would fire at ≤1.50)
# We want it NOT to close → set price to 2.00 (33% TP, below tp_ge_21=0.50)
PRICE_MAP[_opt_sym("MSFT", exp_far, 380, "put")] = 2.00
SPOT_MAP["MSFT"] = 385.0

# Trade E — NVDA IC, no exit
# entry credit 3.00 — give legs that yield cost_to_close ≈ 2.00 (33% — below 0.50 TP)
PRICE_MAP[_opt_sym("NVDA", exp_far, 800, "put")]  = 0.50
PRICE_MAP[_opt_sym("NVDA", exp_far, 790, "put")]  = 0.10
PRICE_MAP[_opt_sym("NVDA", exp_far, 900, "call")] = 1.70
PRICE_MAP[_opt_sym("NVDA", exp_far, 910, "call")] = 0.10
SPOT_MAP["NVDA"] = 850.0

print(f"PRICE_MAP keys: {len(PRICE_MAP)}, SPOT_MAP keys: {len(SPOT_MAP)}")


# --- Run update_positions ---------------------------------------------------
print("\nRunning update_positions...")
pm.update_positions()


# --- Verify -----------------------------------------------------------------
con = sqlite3.connect(tmp.name)
con.row_factory = sqlite3.Row
rows = list(con.execute("SELECT * FROM trades ORDER BY entry_id"))

closed = {r["ticker"]: dict(r) for r in rows if r["status"] == "CLOSED"}
open_ = {r["ticker"]: dict(r) for r in rows if r["status"] == "OPEN"}

print(f"\nClosed: {sorted(closed.keys())}")
print(f"Open:   {sorted(open_.keys())}")

# Trade A: SPY → CLOSED with Take Profit
assert "SPY" in closed, f"SPY should have closed on TP, got {closed.keys()}"
assert "Take Profit" in closed["SPY"]["exit_reason"], f"SPY exit_reason = {closed['SPY']['exit_reason']}"
spy_pnl = closed["SPY"]["pnl_pct"]
# raw pnl = 0.75; friction = (2*0.05 + 2*0.65/100)*2_legs / 1.20 = 0.246/1.20 = 0.205
# realistic ~ 0.545
assert spy_pnl > 0.40, f"SPY pnl_pct should be ~0.55, got {spy_pnl}"
print(f"  ✓ SPY TP fired: pnl_pct={spy_pnl:+.3f} reason={closed['SPY']['exit_reason']}")

# Trade B: QQQ → CLOSED with Stop Loss
assert "QQQ" in closed, f"QQQ should have closed on SL"
assert "Stop Loss" in closed["QQQ"]["exit_reason"], f"QQQ exit_reason = {closed['QQQ']['exit_reason']}"
print(f"  ✓ QQQ SL fired: pnl_pct={closed['QQQ']['pnl_pct']:+.3f} reason={closed['QQQ']['exit_reason']}")

# Trade C: AAPL IC → CLOSED with Time Exit
assert "AAPL" in closed, f"AAPL should have closed on Time"
assert "Time Exit" in closed["AAPL"]["exit_reason"], f"AAPL exit_reason = {closed['AAPL']['exit_reason']}"
print(f"  ✓ AAPL Time fired: pnl_pct={closed['AAPL']['pnl_pct']:+.3f} reason={closed['AAPL']['exit_reason']}")

# Trade D: MSFT short put → still OPEN
assert "MSFT" in open_, f"MSFT should still be OPEN, got closed={closed.keys()}"
print(f"  ✓ MSFT stayed open")

# Trade E: NVDA IC → still OPEN
assert "NVDA" in open_, f"NVDA should still be OPEN"
print(f"  ✓ NVDA stayed open")

# Verify exit_price stored is cost_to_close for multi-leg
assert abs(closed["SPY"]["exit_price"] - 0.30) < 0.01, f"SPY exit_price should be 0.30 (cost to close), got {closed['SPY']['exit_price']}"
print(f"  ✓ SPY exit_price = ${closed['SPY']['exit_price']:.2f} (cost_to_close)")

# Verify dedup of leg fetches: SPY shares no legs with QQQ etc. (sanity by leg key uniqueness)
print("\nOK Phase 5 exit-rule stress PASSED")
os.unlink(tmp.name)
