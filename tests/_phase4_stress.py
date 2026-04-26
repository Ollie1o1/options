"""Phase 4 stress: classifier + multi-leg greeks + max-loss aggregation."""
import os, sys, tempfile, sqlite3
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager
from src.stress_test import _classify_structure, compute_position_greeks, run_stress_test
from src.check_pnl import _legs_for_row

tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
tmp.close()
pm = PaperManager(db_path=tmp.name)

# --- Seed 3 positions: single short put, bull put spread, iron condor
pm.log_trade({
    "date": "2026-04-24", "ticker": "AAPL", "expiration": "2026-05-22",
    "strike": 175, "type": "Put", "entry_price": 2.50,
    "quality_score": 0.65, "strategy_name": "Short Put",
    "entry_iv": 0.28, "entry_delta": -0.30, "entry_gamma": 0.025,
    "entry_vega": 0.20, "entry_theta": -0.04,
})

pm.log_spread({
    "date": "2026-04-24", "ticker": "SPY", "expiration": "2026-05-22",
    "type": "Bull Put", "short_strike": 500, "long_strike": 495,
    "net_credit": 1.20, "max_profit": 120.0, "max_loss": 380.0,
    "quality_score": 0.60,
    "entry_iv": 0.22, "entry_delta": -0.25, "entry_gamma": 0.02,
    "entry_vega": 0.18, "entry_theta": -0.03,
})

pm.log_iron_condor({
    "date": "2026-04-24", "ticker": "QQQ", "expiration": "2026-05-22",
    "short_put_strike": 420, "long_put_strike": 410,
    "short_call_strike": 460, "long_call_strike": 470,
    "total_credit": 2.60, "max_profit": 260.0, "max_risk": 740.0,
    "net_delta": 0.0, "quality_score": 0.74,
    "entry_iv": 0.20, "entry_delta": -0.05, "entry_gamma": 0.015,
    "entry_vega": 0.22, "entry_theta": -0.05,
})

# Read back & test classifier
con = sqlite3.connect(tmp.name)
con.row_factory = sqlite3.Row
rows = [dict(r) for r in con.execute("SELECT * FROM trades")]
print(f"seeded {len(rows)} rows")

assert _classify_structure(rows[0]) == "single", f"AAPL row -> {_classify_structure(rows[0])}"
assert _classify_structure(rows[1]) == "spread", f"SPY row -> {_classify_structure(rows[1])}"
assert _classify_structure(rows[2]) == "iron_condor", f"QQQ row -> {_classify_structure(rows[2])}"
print("OK classifier")

# Test _legs_for_row
legs0 = _legs_for_row(rows[0])
legs1 = _legs_for_row(rows[1])
legs2 = _legs_for_row(rows[2])
assert legs0 == [("put", 175.0, -1)], f"single legs: {legs0}"
assert legs1 == [("put", 500.0, -1), ("put", 495.0, +1)], f"spread legs: {legs1}"
assert legs2 == [("put", 420.0, -1), ("put", 410.0, +1),
                 ("call", 460.0, -1), ("call", 470.0, +1)], f"IC legs: {legs2}"
print("OK _legs_for_row:")
print(f"  single: {legs0}")
print(f"  spread: {legs1}")
print(f"  IC:     {legs2}")

# Test compute_position_greeks against synthetic stock prices
prices = {"AAPL": 180.0, "SPY": 510.0, "QQQ": 440.0}
greeks = compute_position_greeks(rows, prices)
print(f"\ncomputed greeks for {len(greeks)} positions:")
for g in greeks:
    print(f"  {g['ticker']:5s} structure={g['structure']:11s} delta={g['delta']:+.3f} gamma={g['gamma']:+.4f} vega={g['vega']:+.3f} sign={g['sign']:+.0f}")

# Sanity checks
g_aapl = next(g for g in greeks if g["ticker"] == "AAPL")
g_spy = next(g for g in greeks if g["ticker"] == "SPY")
g_qqq = next(g for g in greeks if g["ticker"] == "QQQ")

# AAPL short put OTM @ 175 (S=180): positive delta from short, with sign=-1 baked.
# bs_delta returns negative for puts, so for short put: sign(-1) * delta(-) = +
assert g_aapl["sign"] == -1.0, "single short = sign -1"
# Bull put spread: net delta should be small positive (short put delta partially offset by long put delta)
assert g_spy["structure"] == "spread"
assert g_spy["sign"] == 1.0, "spread sign baked into combined greeks"
# Iron condor: ATM-symmetric so net delta near 0
assert g_qqq["structure"] == "iron_condor"
assert abs(g_qqq["delta"]) < 0.10, f"IC delta should be ~0, got {g_qqq['delta']}"
print("OK greeks structure-aware")

# Run stress test scenarios
df = run_stress_test(rows, prices)
assert df is not None and not df.empty, "stress_test failed"
print(f"\nstress test produced {len(df)} scenarios")
zero_zero = df[(df["stock_move"] == 0.0) & (df["iv_shock"] == 0.0)].iloc[0]
print(f"  flat scenario: P&L = ${zero_zero['total_pnl_usd']:+.2f}  bs/total = {zero_zero['method_breakdown']}")
assert abs(zero_zero["total_pnl_usd"]) < 1.0, "flat scenario should be ~0"

# Worst-case: -20% stock with +20% IV
worst = df.loc[df["total_pnl_usd"].idxmin()]
print(f"  worst scenario: stock {worst['stock_move']:+.0%} IV {worst['iv_shock']:+.0%}  P&L = ${worst['total_pnl_usd']:+.2f}")
print(f"  bs vs fallback: {df['bs_count'].sum()} BS / {df['fallback_count'].sum()} fallback")
assert df["fallback_count"].sum() == 0, "all repricings should use BS"

print("\nOK Phase 4 stress test PASSED")
os.unlink(tmp.name)
