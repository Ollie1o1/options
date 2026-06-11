#!/usr/bin/env python3
"""Render every display surface from fixture data — no network.

Usage: FORCE_COLOR=1 PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/ui_preview.py [surface]
Surfaces: report | summary | ticket | dashboard | all (default)
"""
import sys

import pandas as pd

FIXTURE_PICK = {
    "symbol": "NVDA", "type": "call", "strike": 190.0,
    "expiration": "2026-07-17", "T_years": 36 / 365.0,
    "underlying": 182.40, "premium": 4.20, "bid": 4.15, "ask": 4.24,
    "volume": 1240, "openInterest": 8450, "spread_pct": 0.021,
    "impliedVolatility": 0.42, "iv_percentile_30": 0.41, "hv_30d": 0.38,
    "iv_confidence": "high", "delta": 0.45, "gamma": 0.012, "vega": 0.31,
    "theta": -0.08, "vega_dollar": 31.0, "prob_profit": 0.62,
    "prob_touch": 0.78, "rr_ratio": 2.1, "ev_per_contract": 31.0,
    "max_loss": 420.0, "quality_score": 0.78, "score_drivers": "+mom +iv",
    "pcr": 0.82, "pcr_signal": "BULLISH FLOW", "rsi_14": 58.0,
    "ret_5d": 0.023, "sentiment_tag": "Bullish", "seasonal_win_rate": 0.64,
    "be_dist_pct": 5.2, "breakeven": 194.20, "required_move": 11.80,
    "expected_move": 11.13, "term_structure_spread": 0.03,
    "price_bucket": "MEDIUM", "Earnings Play": "NO",
    "high_premium_turnover": True, "abs_delta": 0.45,
    "quote_freshness": "fresh", "oi_change": 320,
    "iv_surface_residual": -0.18, "strategy_name": "long_call",
}

SECOND_PICK = dict(FIXTURE_PICK, symbol="AAPL", strike=210.0, type="put",
                   quality_score=0.55, prob_profit=0.51, delta=-0.38,
                   ev_per_contract=-12.0, price_bucket="LOW",
                   iv_surface_residual=0.02, high_premium_turnover=False)


def df():
    return pd.DataFrame([FIXTURE_PICK, SECOND_PICK])


def main():
    surface = sys.argv[1] if len(sys.argv) > 1 else "all"
    from src.cli_display import (print_report, print_executive_summary,
                                 print_order_ticket)
    if surface in ("report", "all"):
        print_report(df(), 182.40, 0.043, 3, 14, 45, mode="Discovery scan",
                     market_trend="Bullish", volatility_regime="Normal",
                     config={}, compact=False)
    if surface in ("summary", "all"):
        print_executive_summary(df(), {}, mode="Discovery",
                                market_trend="Bullish",
                                volatility_regime="Normal", num_tickers=2)
    if surface in ("ticket", "all"):
        print_order_ticket(pd.Series(FIXTURE_PICK), {}, account_size=10000)
    if surface == "dashboard":
        from src.regime_dashboard import print_regime_dashboard
        print_regime_dashboard()  # network — run manually only


if __name__ == "__main__":
    main()
