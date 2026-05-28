"""Regression tests for the variance-zero scorer bug.

The 2026-05-11 calibration snapshot reported six scorers (gamma_pin, max_pain,
oi_change, option_rvol, pcr, sentiment) had 181 rows but zero variance in the
paper-trade ledger — i.e. each scorer was pinned to the same fallback constant
across every logged trade. Root cause:

  Each scorer was gated by `if w.get("<name>", 0) > 0:`. Once calibration set
  the weight to 0, the scorer short-circuited and stored the fallback constant
  (0.5 in most cases, 0.3 for max_pain), so the next calibration could never
  rediscover signal in the column. Self-reinforcing.

These tests pin the post-fix behavior: the score is *always* computed from
whatever data is available; the weight only governs how much that score
contributes to the composite. With weight=0 *and* meaningful input data, the
score column must NOT collapse to the fallback constant.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_scorer_signal_recovery -v
"""
from __future__ import annotations

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import numpy as np
import pandas as pd


def _make_chain(n: int = 20, expirations: int = 1, with_rvol: bool = False,
                with_sentiment_var: bool = False, max_pain_value: float | None = None):
    """Synthetic options chain. n rows per expiration."""
    today = datetime.today()
    rows = []
    rng = np.random.default_rng(seed=42)
    for ei in range(expirations):
        exp = (today + timedelta(days=10 + ei * 14)).strftime("%Y-%m-%d")
        # Volume tilt per expiration so pcr varies across expirations.
        call_vol_base = 200 + ei * 300
        put_vol_base = 500 - ei * 250
        for i in range(n):
            is_call = (i % 2 == 0)
            base_vol = call_vol_base if is_call else put_vol_base
            row = {
                "symbol": "TEST",
                "type": "call" if is_call else "put",
                "strike": 150.0 + i * 5,
                "expiration": exp,
                "impliedVolatility": 0.25 + (i % 5) * 0.02,
                "volume": base_vol + i * 20,
                "openInterest": 500 + i * 100,
                "bid": 2.0 + i * 0.5,
                "ask": 2.2 + i * 0.5,
                "lastPrice": 2.1 + i * 0.5,
                "underlying": 155.0,
                "hv_30d": 0.22,
                "sentiment_score": (float(rng.normal(0, 0.4)) if with_sentiment_var else 0.0),
                "sma_20": 150.0, "sma_50": 148.0,
                "ret_5d": 0.01, "rsi_14": 55.0, "atr_trend": 1.5,
                "high_20": 160.0, "low_20": 145.0,
                "rvol": 1.0, "is_squeezing": False,
                "short_interest": 0.05, "seasonal_win_rate": 0.5,
                "vwap": 154.0, "fib_50": 152.0, "fib_618": 153.0,
                "iv_rank_30": 0.5, "iv_percentile_30": 0.5,
                "iv_rank_90": 0.5, "iv_percentile_90": 0.5,
                "iv_confidence": "Normal",
            }
            if with_rvol:
                row["option_rvol"] = 0.5 + (i / n) * 4.0  # 0.5 .. 4.5 spread
            if max_pain_value is not None:
                # Older data-fetch path stores `max_pain`, not `max_pain_strike`.
                row["max_pain"] = max_pain_value
            rows.append(row)
    return pd.DataFrame(rows)


def _config(**overrides) -> dict:
    base = {
        "filters": {
            "min_volume": 1, "min_open_interest": 1,
            "delta_min": 0.0, "delta_max": 1.0,
            "max_bid_ask_spread_pct": 0.95, "min_iv_percentile": 0,
        },
        # All broken-scorer weights zero to reproduce the calibration end-state
        # that triggered the variance-zero report.
        "composite_weights": {
            "pop": 0.13, "iv_mispricing": 0.05, "rr": 0.10, "momentum": 0.10,
            "iv_rank": 0.08, "liquidity": 0.08, "catalyst": 0.00, "theta": 0.06,
            "ev": 0.07, "trader_pref": 0.00, "iv_edge": 0.08, "skew_align": 0.02,
            "gamma_theta": 0.00, "pcr": 0.0, "gex": 0.01, "oi_change": 0.0,
            "sentiment": 0.0, "option_rvol": 0.0, "vrp": 0.05, "gamma_pin": 0.0,
            "max_pain": 0.0, "iv_velocity": 0.05, "em_realism": 0.00,
            "gamma_magnitude": 0.03, "vega_risk": 0.03, "term_structure": 0.04,
            "spread": 0.01,
        },
        "min_pop": 0.0, "max_delta": 1.0,
        "iv_outlier_threshold": 0.50, "iv_outlier_min_volume": 5,
        "moneyness_band": 0.50,
    }
    base.update(overrides)
    return base


def _run(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    from src.options_screener import enrich_and_score, _invalidate_ic_weights_cache
    _invalidate_ic_weights_cache()  # tests run in same process
    with patch("src.options_screener.monte_carlo_pop", return_value=(0.6, 0.4)):
        out = enrich_and_score(
            df=df, min_dte=1, max_dte=120, risk_free_rate=0.05, config=config,
            vix_regime_weights=config.get("composite_weights", {}),
            trader_profile="swing", mode="Single-stock",
            iv_rank=0.5, iv_percentile=0.5,
            earnings_date=None, sentiment_score=0.0,
            seasonal_win_rate=None, term_structure_spread=None,
            macro_risk_active=False, sector_perf={},
            tnx_change_pct=0.0, short_interest=None,
            next_ex_div=None, earnings_move_data=None,
            hv_ewma=None, news_data=None,
        )
    return out


class VarianceZeroRecovery(unittest.TestCase):
    """Each broken scorer must produce non-fallback output when weight=0."""

    def test_option_rvol_score_not_constant_when_weight_zero(self):
        df = _make_chain(n=12, with_rvol=True)
        out = _run(df, _config())
        if out.empty:
            self.skipTest("enrich_and_score filtered everything")
        scores = out["option_rvol_score"].dropna()
        distinct = scores.unique()
        self.assertGreater(
            len(distinct), 1,
            f"option_rvol_score collapsed to {distinct} despite varied option_rvol input "
            "— weight gate is still short-circuiting",
        )
        self.assertFalse(
            np.allclose(scores.values, 0.5),
            "option_rvol_score pinned to fallback 0.5",
        )

    def test_pcr_score_not_constant_when_weight_zero(self):
        # Multiple expirations so pcr (computed per-expiration) varies cross-row.
        df = _make_chain(n=10, expirations=3)
        out = _run(df, _config())
        if out.empty:
            self.skipTest("enrich_and_score filtered everything")
        scores = out["pcr_score"].dropna()
        self.assertGreater(
            len(scores.unique()), 1,
            f"pcr_score collapsed to {scores.unique()} despite varied per-expiration PCR",
        )

    def test_max_pain_score_uses_max_pain_when_strike_missing(self):
        # Older data-fetch path stores `max_pain` (no _strike suffix). The scorer
        # should fall back to that column rather than emit constant 0.3.
        df = _make_chain(n=10, max_pain_value=158.0)  # ~2% away from 155
        out = _run(df, _config())
        if out.empty:
            self.skipTest("enrich_and_score filtered everything")
        scores = out["max_pain_score"].dropna()
        self.assertFalse(
            np.allclose(scores.values, 0.3),
            f"max_pain_score pinned to 0.3 — data fallback (df['max_pain'] = 158) "
            f"was not honored; got {scores.unique()}",
        )

    def test_sentiment_score_norm_responds_to_per_row_sentiment(self):
        df = _make_chain(n=12, with_sentiment_var=True)
        out = _run(df, _config())
        if out.empty:
            self.skipTest("enrich_and_score filtered everything")
        scores = out["sentiment_score_norm"].dropna()
        # With varied per-row sentiment_score, normalized score should vary.
        self.assertGreater(
            len(scores.unique()), 1,
            f"sentiment_score_norm collapsed to {scores.unique()} despite varied input",
        )


if __name__ == "__main__":
    unittest.main()
