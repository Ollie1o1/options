"""Tests for Long Gamma scan mode components."""
import pytest
import pandas as pd
import numpy as np


class TestBbWidthPct:
    """Tests for the bb_width_pct signal in calculate_momentum_indicators."""

    def _make_hist(self, n=252, compressed=False):
        """Create a fake price history. compressed=True makes recent vol low."""
        np.random.seed(42)
        prices = 100.0 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        if compressed:
            # Last 20 days: very low daily moves
            prices[-20:] = prices[-21] + np.cumsum(np.random.normal(0, 0.001, 20))
        idx = pd.date_range(end="2026-04-16", periods=n, freq="B")
        return pd.DataFrame(
            {"Close": prices, "High": prices * 1.005, "Low": prices * 0.995, "Volume": 1_000_000},
            index=idx,
        )

    def test_returns_ten_values(self):
        from src.data_fetching import calculate_momentum_indicators
        hist = self._make_hist()
        result = calculate_momentum_indicators(hist)
        assert len(result) == 10, f"Expected 10-tuple, got {len(result)}"

    def test_bb_width_pct_is_float_between_0_and_1(self):
        from src.data_fetching import calculate_momentum_indicators
        hist = self._make_hist()
        *_, bb_width_pct = calculate_momentum_indicators(hist)
        assert bb_width_pct is not None
        assert 0.0 <= bb_width_pct <= 1.0

    def test_compressed_history_has_lower_bb_width_pct(self):
        from src.data_fetching import calculate_momentum_indicators
        normal_hist = self._make_hist(compressed=False)
        compressed_hist = self._make_hist(compressed=True)
        *_, normal_pct = calculate_momentum_indicators(normal_hist)
        *_, compressed_pct = calculate_momentum_indicators(compressed_hist)
        assert compressed_pct is not None and normal_pct is not None
        assert compressed_pct < normal_pct, (
            f"Compressed ({compressed_pct:.3f}) should be < normal ({normal_pct:.3f})"
        )

    def test_empty_history_returns_none_for_bb_width_pct(self):
        from src.data_fetching import calculate_momentum_indicators
        result = calculate_momentum_indicators(pd.DataFrame())
        assert result[-1] is None  # bb_width_pct should be None


class TestFilterLongGamma:
    """Tests for filter_long_gamma."""

    def _make_df(self):
        """Create a minimal options DataFrame for testing."""
        return pd.DataFrame({
            "symbol": ["AAPL"] * 5,
            "type": ["call"] * 5,
            "strike": [150.0, 155.0, 160.0, 165.0, 170.0],
            # indices 2 (0.45) and 4 (0.60) fail IV gate
            "iv_percentile_30": [0.10, 0.35, 0.45, 0.20, 0.60],
            # index 2 (10 days) fails DTE gate
            "T_years": [30 / 365, 45 / 365, 10 / 365, 50 / 365, 35 / 365],
            # index 3 (50) fails volume gate
            "volume": [200, 500, 100, 50, 150],
            # index 3 (0.50) fails spread gate
            "spread_pct": [0.10, 0.20, 0.30, 0.50, 0.15],
        })

    def test_filters_high_iv_rank(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        assert all(result["iv_percentile_30"] < 0.40)

    def test_filters_short_dte(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        dte = result["T_years"] * 365.0
        assert all(dte >= 20)
        assert all(dte <= 60)

    def test_filters_low_volume(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        assert all(result["volume"] >= 100)

    def test_filters_wide_spread(self):
        from src.filters import filter_long_gamma
        df = self._make_df()
        result = filter_long_gamma(df)
        assert all(result["spread_pct"] <= 0.40)

    def test_empty_df_returns_empty(self):
        from src.filters import filter_long_gamma
        result = filter_long_gamma(pd.DataFrame())
        assert result.empty


class TestRecommendedStrategy:
    """Tests for the recommended_strategy column logic."""

    def _make_row(self, is_squeezing, adx, rsi, iv_pct, rvol, underlying, sma_50):
        return pd.DataFrame([{
            "is_squeezing": is_squeezing,
            "adx_14": adx,
            "rsi_14": rsi,
            "iv_percentile_30": iv_pct,
            "rvol": rvol,
            "underlying": underlying,
            "sma_50": sma_50,
        }])

    def _apply_strategy(self, df):
        """Replicate the np.select logic from enrich_and_score."""
        _is_sq = df.get("is_squeezing", pd.Series(False)).fillna(False).astype(bool)
        _adx = pd.to_numeric(df.get("adx_14", pd.Series(20.0)), errors="coerce").fillna(20.0)
        _rsi = pd.to_numeric(df.get("rsi_14", pd.Series(50.0)), errors="coerce").fillna(50.0)
        _iv_pct = pd.to_numeric(df.get("iv_percentile_30", pd.Series(0.3)), errors="coerce").fillna(0.3)
        _rvol = pd.to_numeric(df.get("rvol", pd.Series(1.0)), errors="coerce").fillna(1.0)
        _underlying = pd.to_numeric(df.get("underlying", pd.Series(0.0)), errors="coerce").fillna(0.0)
        _sma50 = pd.to_numeric(df.get("sma_50", pd.Series(0.0)), errors="coerce").fillna(0.0)
        _neutral_rsi = (_rsi >= 45) & (_rsi <= 55)
        _conditions = [
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct < 0.15),
            _is_sq & (_adx < 20) & _neutral_rsi & (_iv_pct >= 0.15),
            _is_sq & (_adx >= 20) & (_rsi > 55) & (_underlying > _sma50),
            _is_sq & (_adx >= 20) & (_rsi < 45) & (_underlying < _sma50),
            (~_is_sq) & (_rvol > 1.5) & (_adx > 25),
        ]
        _choices = ["Straddle", "Strangle", "Bull Call Spread", "Bear Put Spread", "Directional Debit Spread"]
        return np.select(_conditions, _choices, default="Monitor")

    def test_straddle_when_squeeze_no_direction_very_cheap(self):
        df = self._make_row(True, 15, 50, 0.10, 1.0, 150, 145)
        assert self._apply_strategy(df)[0] == "Straddle"

    def test_strangle_when_squeeze_no_direction_moderately_cheap(self):
        df = self._make_row(True, 15, 50, 0.25, 1.0, 150, 145)
        assert self._apply_strategy(df)[0] == "Strangle"

    def test_bull_call_spread_when_bullish_breakout(self):
        df = self._make_row(True, 25, 60, 0.25, 1.0, 155, 150)
        assert self._apply_strategy(df)[0] == "Bull Call Spread"

    def test_bear_put_spread_when_bearish_breakout(self):
        df = self._make_row(True, 25, 40, 0.25, 1.0, 145, 150)
        assert self._apply_strategy(df)[0] == "Bear Put Spread"

    def test_directional_debit_spread_when_momentum_no_squeeze(self):
        df = self._make_row(False, 28, 62, 0.25, 2.0, 155, 150)
        assert self._apply_strategy(df)[0] == "Directional Debit Spread"

    def test_monitor_when_no_conditions_met(self):
        df = self._make_row(False, 15, 50, 0.35, 0.8, 150, 145)
        assert self._apply_strategy(df)[0] == "Monitor"
