"""Tests for src/filters.py."""
import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.filters import filter_options, categorize_by_premium, pick_top_per_bucket


def _make_option_row(**overrides):
    row = {
        "symbol": "AAPL",
        "type": "call",
        "strike": 150.0,
        "expiration": "2026-06-20",
        "premium": 3.50,
        "quality_score": 0.60,
        "volume": 500.0,
        "openInterest": 1000.0,
        "impliedVolatility": 0.30,
        "spread_pct": 0.05,
        "abs_delta": 0.30,
        "iv_percentile": 0.40,
    }
    row.update(overrides)
    return row


def _config(min_volume=50, min_oi=10, max_spread=0.40, delta_min=0.15, delta_max=0.45, min_iv_pct=0):
    return {
        "filters": {
            "min_volume": min_volume,
            "min_open_interest": min_oi,
            "max_bid_ask_spread_pct": max_spread,
            "delta_min": delta_min,
            "delta_max": delta_max,
            "min_iv_percentile": min_iv_pct,
        }
    }


def test_filter_removes_low_volume():
    """volume=10 < min_volume=50 -> filtered out."""
    df = pd.DataFrame([_make_option_row(volume=10)])
    result = filter_options(df, _config(min_volume=50))
    assert result.empty


def test_filter_keeps_high_oi_low_vol():
    """volume=10 but OI=200. filter_options uses AND, so low-vol row IS filtered out."""
    df = pd.DataFrame([_make_option_row(volume=10, openInterest=200)])
    result = filter_options(df, _config(min_volume=50, min_oi=10))
    # volume=10 < min_volume=50 → filtered regardless of OI
    assert result.empty


def test_categorize_by_premium_budget_mode():
    """budget=500: $100->LOW, $250->MEDIUM, $400->HIGH."""
    rows = [
        _make_option_row(premium=1.00),   # cost=100 -> 20% of 500 -> LOW
        _make_option_row(premium=2.50),   # cost=250 -> 50% of 500 -> MEDIUM
        _make_option_row(premium=4.00),   # cost=400 -> 80% of 500 -> HIGH
    ]
    df = pd.DataFrame(rows)
    result = categorize_by_premium(df, budget=500)
    buckets = list(result["price_bucket"])
    assert buckets[0] == "LOW"
    assert buckets[1] == "MEDIUM"
    assert buckets[2] == "HIGH"


def test_categorize_by_premium_quantile_mode():
    """No budget, 3 rows at 1.0/5.0/10.0 -> LOW/MEDIUM/HIGH."""
    rows = [
        _make_option_row(premium=1.0),
        _make_option_row(premium=5.0),
        _make_option_row(premium=10.0),
    ]
    df = pd.DataFrame(rows)
    result = categorize_by_premium(df, budget=None)
    assert "price_bucket" in result.columns
    # Lowest premium should be LOW, highest should be HIGH
    buckets = list(result["price_bucket"])
    assert buckets[0] == "LOW"
    assert buckets[-1] == "HIGH"


def test_pick_top_per_bucket_respects_per_bucket():
    """10 LOW rows, per_bucket=3 -> exactly 3 returned."""
    rows = [_make_option_row(premium=1.0, quality_score=float(i)/10) for i in range(10)]
    df = pd.DataFrame(rows)
    df["price_bucket"] = "LOW"
    result = pick_top_per_bucket(df, per_bucket=3)
    assert len(result) == 3


def test_pick_top_per_bucket_diversifies_tickers():
    """5 AAPL + 1 TSLA in LOW, diversify=True, per_bucket=3 -> TSLA in results."""
    rows_aapl = [_make_option_row(symbol="AAPL", quality_score=0.9, premium=1.0) for _ in range(5)]
    rows_tsla = [_make_option_row(symbol="TSLA", quality_score=0.5, premium=1.0)]
    df = pd.DataFrame(rows_aapl + rows_tsla)
    df["price_bucket"] = "LOW"
    result = pick_top_per_bucket(df, per_bucket=3, diversify_tickers=True)
    assert "TSLA" in result["symbol"].values
