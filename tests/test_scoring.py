"""Tests for enrich_and_score in src/options_screener.py."""
import pandas as pd
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta


def _make_options_df(n=5):
    """Build a minimal synthetic options DataFrame."""
    today = datetime.today()
    exp = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    return pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "type": (["call", "put"] * (n // 2 + 1))[:n],
        "strike": [150.0 + i * 5 for i in range(n)],
        "expiration": [exp] * n,
        "impliedVolatility": [0.25 + i * 0.02 for i in range(n)],
        "volume": [100 + i * 50 for i in range(n)],
        "openInterest": [500 + i * 100 for i in range(n)],
        "bid": [2.0 + i * 0.5 for i in range(n)],
        "ask": [2.2 + i * 0.5 for i in range(n)],
        "lastPrice": [2.1 + i * 0.5 for i in range(n)],
        "underlying": [155.0] * n,
        "hv_30d": [0.22] * n,
        # Columns normally populated by fetch_options_yfinance
        "sentiment_score": [0.0] * n,
        "sma_20": [150.0] * n,
        "sma_50": [148.0] * n,
        "ret_5d": [0.01] * n,
        "rsi_14": [55.0] * n,
        "atr_trend": [1.5] * n,
        "high_20": [160.0] * n,
        "low_20": [145.0] * n,
        "rvol": [1.0] * n,
        "is_squeezing": [False] * n,
        "short_interest": [0.05] * n,
        "seasonal_win_rate": [0.5] * n,
        "vwap": [154.0] * n,
        "fib_50": [152.0] * n,
        "fib_618": [153.0] * n,
        "iv_rank_30": [0.5] * n,
        "iv_percentile_30": [0.5] * n,
        "iv_rank_90": [0.5] * n,
        "iv_percentile_90": [0.5] * n,
        "iv_confidence": ["Normal"] * n,
    })


def _make_config():
    """Minimal config for enrich_and_score."""
    return {
        "filters": {
            "min_volume": 10,
            "min_open_interest": 10,
            "delta_min": 0.05,
            "delta_max": 0.95,
            "max_bid_ask_spread_pct": 0.50,
            "min_iv_percentile": 0,
        },
        "composite_weights": {
            "pop_weight": 0.30,
            "ev_weight": 0.20,
            "iv_rank_weight": 0.15,
            "spread_weight": 0.10,
            "trend_weight": 0.10,
            "hv_iv_weight": 0.15,
        },
        "min_pop": 0.40,
        "max_delta": 0.50,
        "iv_outlier_threshold": 0.50,
        "iv_outlier_min_volume": 5,
        "moneyness_band": 0.30,
    }


def _call_enrich_and_score(df, config):
    """Helper that calls enrich_and_score with sensible defaults."""
    from src.scanner import enrich_and_score
    vix_regime_weights = config.get("composite_weights", {})
    return enrich_and_score(
        df=df,
        min_dte=1,
        max_dte=90,
        risk_free_rate=0.05,
        config=config,
        vix_regime_weights=vix_regime_weights,
        trader_profile="swing",
        mode="Single-stock",
        iv_rank=0.5,
        iv_percentile=0.5,
        earnings_date=None,
        sentiment_score=0.0,
        seasonal_win_rate=None,
        term_structure_spread=None,
        macro_risk_active=False,
        sector_perf={},
        tnx_change_pct=0.0,
        short_interest=None,
        next_ex_div=None,
        earnings_move_data=None,
        hv_ewma=None,
        news_data=None,
    )


def test_enrich_and_score_returns_nonempty():
    """enrich_and_score should return a non-empty DataFrame with quality_score column."""
    df = _make_options_df(n=10)
    config = _make_config()
    with patch("src.scanner.monte_carlo_pop", return_value=(0.6, 0.4)):
        result = _call_enrich_and_score(df, config)
    assert not result.empty, "enrich_and_score returned an empty DataFrame"
    assert "quality_score" in result.columns, "quality_score column missing from output"


def test_quality_score_range():
    """All quality_score values should be in [0, 1] after clipping."""
    df = _make_options_df(n=10)
    config = _make_config()
    with patch("src.scanner.monte_carlo_pop", return_value=(0.6, 0.4)):
        result = _call_enrich_and_score(df, config)
    if result.empty:
        pytest.skip("enrich_and_score returned empty (all rows filtered out)")
    scores = result["quality_score"].dropna()
    assert (scores >= 0.0).all(), f"quality_score below 0: {scores[scores < 0]}"
    assert (scores <= 1.0).all(), f"quality_score above 1: {scores[scores > 1]}"


def test_prob_profit_range():
    """All prob_profit values should be in [0, 1]."""
    df = _make_options_df(n=10)
    config = _make_config()
    with patch("src.scanner.monte_carlo_pop", return_value=(0.6, 0.4)):
        result = _call_enrich_and_score(df, config)
    if result.empty:
        pytest.skip("enrich_and_score returned empty (all rows filtered out)")
    assert "prob_profit" in result.columns, "prob_profit column missing from output"
    pp = result["prob_profit"].dropna()
    assert (pp >= 0.0).all(), f"prob_profit below 0: {pp[pp < 0]}"
    assert (pp <= 1.0).all(), f"prob_profit above 1: {pp[pp > 1]}"
