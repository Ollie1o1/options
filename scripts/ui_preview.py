#!/usr/bin/env python3
"""Render every display surface from fixture data — no network.

Usage: FORCE_COLOR=1 PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/ui_preview.py [surface]
Surfaces: desk | report | summary | ticket | dashboard | all (default)
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
    "iv_skew": 0.041, "iv_skew_rank": 0.83,
}

SECOND_PICK = dict(FIXTURE_PICK, symbol="AAPL", strike=210.0, type="put",
                   quality_score=0.55, prob_profit=0.51, delta=-0.38,
                   ev_per_contract=-12.0, price_bucket="LOW",
                   iv_surface_residual=0.02, high_premium_turnover=False)

# Distinct expiry + richer IV so the `report` surface exercises the IV-term
# sparkline (it needs >=2 expiries among the picks).
THIRD_PICK = dict(FIXTURE_PICK, symbol="MSFT", strike=480.0, premium=11.30,
                  quality_score=0.84, prob_profit=0.66, ev_per_contract=88.0,
                  price_bucket="HIGH", iv_surface_residual=0.0,
                  high_premium_turnover=False,
                  expiration="2026-08-21", T_years=71 / 365.0,
                  impliedVolatility=0.47)


def df():
    return pd.DataFrame([FIXTURE_PICK, SECOND_PICK, THIRD_PICK])


def preview_desk():
    """The four desk panels + the heat scale, from fixtures (no network)."""
    from src import ui, check_pnl, cli_display, stress_test

    print("\n=== heat_cell scale ===")
    span = 85000
    for v in (-85000, -40000, -5000, 0, 5000, 40000, 85000):
        print(ui.heat_cell(f"{v:>+8,.0f}", v, span), end="  ")
    print()

    print("\n=== stress heatmap ===")
    rows = [{"stock_move": sm, "iv_shock": iv,
             "total_pnl_usd": sm * 600000 + iv * 120000,
             "pnl_pct_of_book": (sm * 600000 + iv * 120000) / 200000.0}
            for iv in (-0.10, -0.05, 0.0, 0.05, 0.10)
            for sm in (-0.10, -0.05, 0.0, 0.05, 0.10)]
    _real = stress_test.run_stress_test
    stress_test.run_stress_test = lambda *a, **k: pd.DataFrame(rows)
    try:
        stress_test.print_stress_test([{"ticker": "SPY"}], {"SPY": 500.0}, width=90)
    finally:
        stress_test.run_stress_test = _real

    print("\n=== greeks by underlying ===")
    check_pnl._print_greeks_by_ticker(
        {"QQQ": [-312.0, -4.1], "SPY": [-198.0, -2.0],
         "AAPL": [-121.0, 1.2], "TSLA": [88.0, -0.6]}, width=100)

    print("\n=== P&L attribution waterfall ===")
    attrib = [{"entry_delta": 0.45, "entry_theta": -0.08, "entry_gamma": 0.012,
               "entry_vega": 0.31, "entry_price": 5.0, "pnl_pct": 0.2,
               "ticker": "AAPL", "strategy_name": "long_call",
               "date": "2026-06-01", "exit_date": "2026-06-15"} for _ in range(5)]
    check_pnl._print_pnl_attribution(attrib, {"AAPL": 190.0}, width=100)

    print("\n=== equity curve ===")
    trades = [{"pnl_pct": (0.12 if i % 3 else -0.18), "entry_price": 5.0,
               "ticker": "AAPL", "exit_date": f"2026-06-{(i % 28) + 1:02d}",
               "date": None} for i in range(40)]
    check_pnl._print_equity_curve(trades, width=100)

    print("\n=== term-structure sparkline ===")
    curve = cli_display._iv_term_curve_from_picks(pd.DataFrame([
        {"expiration": "2026-07-17", "impliedVolatility": 0.30, "T_years": 9 / 365},
        {"expiration": "2026-07-24", "impliedVolatility": 0.33, "T_years": 16 / 365},
        {"expiration": "2026-08-21", "impliedVolatility": 0.37, "T_years": 44 / 365},
    ]))
    ivs = [v for _, v in curve]
    print("  IV term  " + ui.sparkline(ivs) + "  " +
          " · ".join(f"{d}d {v:.0%}" for d, v in curve))

    print("\n=== 25-delta skew read ===")
    for skew, rank in ((0.045, 0.87), (-0.032, 0.12), (0.003, None)):
        row = {"quality_score": 0.8, "iv_skew": skew}
        if rank is not None:
            row["iv_skew_rank"] = rank
        print(ui.kv_line("Skew 25Δ", cli_display._skew_read_from_picks(pd.DataFrame([row]))))

    print("\n=== staleness banner ===")
    from datetime import date
    from src import maintenance_health as mh
    stale = {"last_autolog": {k: "2026-06-26" for k in ("ds", "sps", "ss", "ics")}}
    print(mh.health_banner(mh.compute_health(stale, date(2026, 7, 7)), width=100))


def main():
    surface = sys.argv[1] if len(sys.argv) > 1 else "all"
    from src.cli_display import (print_report, print_executive_summary,
                                 print_order_ticket)
    if surface in ("desk", "all"):
        preview_desk()
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
