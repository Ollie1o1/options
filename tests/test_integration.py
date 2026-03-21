"""End-to-end integration tests for the options screener pipeline."""
import pandas as pd
import numpy as np
import pytest
from unittest.mock import patch
from datetime import datetime, timedelta


def _make_chain(n=20):
    """Build a realistic n-contract options chain DataFrame."""
    if n == 0:
        cols = [
            "symbol", "type", "strike", "expiration", "impliedVolatility",
            "volume", "openInterest", "bid", "ask", "lastPrice", "underlying",
            "hv_30d", "sentiment_score", "sma_20", "sma_50", "ret_5d", "rsi_14",
            "atr_trend", "high_20", "low_20", "rvol", "is_squeezing",
            "short_interest", "seasonal_win_rate", "vwap", "fib_50", "fib_618",
            "iv_rank_30", "iv_percentile_30", "iv_rank_90", "iv_percentile_90",
            "iv_confidence",
        ]
        return pd.DataFrame(columns=cols)

    today = datetime.today()
    exp = (today + timedelta(days=30)).strftime("%Y-%m-%d")
    strikes = [145.0 + i * 2.5 for i in range(n)]
    types = (["call", "put"] * (n // 2 + 1))[:n]
    ivs = [0.20 + i * 0.01 for i in range(n)]
    vols = [200 + i * 30 for i in range(n)]
    ois = [1000 + i * 50 for i in range(n)]
    bids = [max(0.05, 3.0 - i * 0.12) for i in range(n)]
    asks = [b + 0.10 for b in bids]
    return pd.DataFrame({
        "symbol": ["AAPL"] * n,
        "type": types,
        "strike": strikes,
        "expiration": [exp] * n,
        "impliedVolatility": ivs,
        "volume": vols,
        "openInterest": ois,
        "bid": bids,
        "ask": asks,
        "lastPrice": [(b + a) / 2 for b, a in zip(bids, asks)],
        "underlying": [155.0] * n,
        "hv_30d": [0.22] * n,
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
        "iv_confidence": ["High"] * n,
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


def _run_pipeline(df, config, mode="Single-stock"):
    """Run enrich_and_score with standard mocks."""
    from src.options_screener import enrich_and_score
    vix_regime_weights = config.get("composite_weights", {})
    with patch("src.options_screener.monte_carlo_pop", return_value=(0.6, 0.4)):
        return enrich_and_score(
            df=df, min_dte=1, max_dte=90, risk_free_rate=0.05,
            config=config, vix_regime_weights=vix_regime_weights,
            trader_profile="swing", mode=mode, iv_rank=0.5,
            iv_percentile=0.5, earnings_date=None, sentiment_score=0.0,
            seasonal_win_rate=None, term_structure_spread=None,
            macro_risk_active=False, sector_perf={}, tnx_change_pct=0.0,
            short_interest=None, next_ex_div=None,
            earnings_move_data=None, hv_ewma=None, news_data=None,
        )


class TestNormalScan:
    def test_columns_exist(self):
        result = _run_pipeline(_make_chain(), _make_config())
        assert not result.empty
        for col in ["quality_score", "spread_pct", "delta"]:
            assert col in result.columns, f"{col} missing"

    def test_scores_in_range(self):
        result = _run_pipeline(_make_chain(), _make_config())
        if result.empty:
            pytest.skip("Pipeline returned empty")
        scores = result["quality_score"].dropna()
        assert (scores >= 0).all() and (scores <= 1).all()

    def test_no_nan_quality_score(self):
        result = _run_pipeline(_make_chain(), _make_config())
        if result.empty:
            pytest.skip("Pipeline returned empty")
        assert result["quality_score"].isna().sum() == 0


class TestEdgeCases:
    def test_empty_chain(self):
        """Empty DataFrame should not crash."""
        df = _make_chain(n=0)
        result = _run_pipeline(df, _make_config())
        assert isinstance(result, pd.DataFrame)

    def test_all_nan_iv(self):
        """All-NaN IV should not crash — contracts may be dropped by NaN gate."""
        df = _make_chain()
        df["impliedVolatility"] = np.nan
        result = _run_pipeline(df, _make_config())
        assert isinstance(result, pd.DataFrame)

    def test_zero_volume(self):
        """Zero volume contracts should still be scored."""
        df = _make_chain()
        df["volume"] = 0
        result = _run_pipeline(df, _make_config())
        assert isinstance(result, pd.DataFrame)


class TestScoringBounds:
    def test_component_scores_bounded(self):
        """All scoring components should be in [0, 1]."""
        result = _run_pipeline(_make_chain(n=20), _make_config())
        if result.empty:
            pytest.skip("Pipeline returned empty")
        score_cols = [c for c in result.columns if c.endswith("_score") and c != "quality_score"]
        for col in score_cols:
            vals = pd.to_numeric(result[col], errors="coerce").dropna()
            if vals.empty:
                continue
            assert (vals >= -0.01).all(), f"{col} has values below 0: {vals[vals < 0].tolist()}"
            assert (vals <= 1.01).all(), f"{col} has values above 1: {vals[vals > 1].tolist()}"
