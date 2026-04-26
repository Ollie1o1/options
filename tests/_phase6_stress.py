"""Phase 6 stress: print_credit_spreads_report + print_iron_condor_report parity check.

Builds synthetic enriched DataFrames and verifies the new detail-card pipeline
renders without errors AND emits the parity sections (thesis, breakevens,
top components, execution guidance, comparison table).
"""
import os, sys, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.cli_display import print_credit_spreads_report, print_iron_condor_report


# Build 3 synthetic credit spreads with full enrichment columns
spread_rows = [
    dict(
        symbol="SPY", expiration="2026-05-20", type="Bull Put",
        short_strike=500.0, long_strike=495.0,
        net_credit=1.20, max_profit=120.0, max_loss=380.0,
        spread_width=5.0, credit_to_width_ratio=0.24,
        return_on_risk_ratio=120.0/380.0,
        quality_score=0.72,
        pop_score=0.78, credit_to_width_score=0.55, iv_rank_score=0.62,
        return_on_risk_score=0.40, liquidity_score=0.85, theta_score=0.65,
        spread_score=0.70, momentum_score=0.45, catalyst_score=0.30,
    ),
    dict(
        symbol="QQQ", expiration="2026-05-20", type="Bear Call",
        short_strike=460.0, long_strike=465.0,
        net_credit=1.00, max_profit=100.0, max_loss=400.0,
        spread_width=5.0, credit_to_width_ratio=0.20,
        return_on_risk_ratio=0.25,
        quality_score=0.60,
        pop_score=0.70, credit_to_width_score=0.30, iv_rank_score=0.45,
        return_on_risk_score=0.20, liquidity_score=0.70, theta_score=0.50,
        spread_score=0.40, momentum_score=0.50, catalyst_score=0.40,
    ),
    dict(
        symbol="AAPL", expiration="2026-05-20", type="Bull Put",
        short_strike=170.0, long_strike=165.0,
        net_credit=1.50, max_profit=150.0, max_loss=350.0,
        spread_width=5.0, credit_to_width_ratio=0.30,
        return_on_risk_ratio=150.0/350.0,
        quality_score=0.78,
        pop_score=0.82, credit_to_width_score=0.70, iv_rank_score=0.75,
        return_on_risk_score=0.35, liquidity_score=0.90, theta_score=0.70,
        spread_score=0.75, momentum_score=0.60, catalyst_score=0.20,
    ),
]
spreads_df = pd.DataFrame(spread_rows)

condor_rows = [
    dict(
        symbol="SPY", expiration="2026-05-20",
        short_put_strike=480.0, long_put_strike=475.0,
        short_call_strike=520.0, long_call_strike=525.0,
        total_credit=2.40, max_profit=240.0, max_risk=260.0,
        return_on_risk=240.0/260.0,
        net_delta=-0.02,
        spread_width=5.0,
        credit_to_width_ratio=0.48, credit_to_width_score=0.65,
        delta_neutral_score=0.92,
        quality_score=0.74,
        pop_score=0.80, iv_rank_score=0.55,
        liquidity_score=0.88, theta_score=0.62, spread_score=0.65,
    ),
    dict(
        symbol="QQQ", expiration="2026-05-20",
        short_put_strike=420.0, long_put_strike=410.0,
        short_call_strike=460.0, long_call_strike=470.0,
        total_credit=2.60, max_profit=260.0, max_risk=740.0,
        return_on_risk=260.0/740.0,
        net_delta=0.01,
        spread_width=10.0,
        credit_to_width_ratio=0.26, credit_to_width_score=0.40,
        delta_neutral_score=0.95,
        quality_score=0.66,
        pop_score=0.72, iv_rank_score=0.50,
        liquidity_score=0.75, theta_score=0.55, spread_score=0.45,
    ),
]
condors_df = pd.DataFrame(condor_rows)


def _capture(fn, df):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        fn(df)
    return buf.getvalue()


# Spread report
out = _capture(print_credit_spreads_report, spreads_df)
print(out)
print("=" * 60)

required_spread_markers = [
    "CREDIT SPREADS REPORT",
    "[1/3]", "[2/3]", "[3/3]",
    "Thesis:",
    "Top components:",
    "Execution:",
    "BE $",
    "COMPARISON TABLE",
]
for m in required_spread_markers:
    assert m in out, f"missing marker {m!r}"
print(f"  ✓ all {len(required_spread_markers)} spread markers present")

# IC report
out2 = _capture(print_iron_condor_report, condors_df)
print(out2)

required_ic_markers = [
    "IRON CONDOR REPORT",
    "[1/2]", "[2/2]",
    "Thesis:",
    "Top components:",
    "Execution:",
    "BE $",
    "COMPARISON TABLE",
]
for m in required_ic_markers:
    assert m in out2, f"missing IC marker {m!r}"
print(f"  ✓ all {len(required_ic_markers)} IC markers present")

print("\nOK Phase 6 scan-report parity stress PASSED")
