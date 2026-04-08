import pytest
import numpy as np
import pandas as pd
import datetime
from src.utils import bs_call, bs_put, bs_delta, bs_gamma
from src.scoring import enrich_and_score

@pytest.fixture
def mock_options_df():
    # A realistic options dataframe with 1000 rows
    np.random.seed(42)
    n = 1000
    df = pd.DataFrame({
        "strike": np.random.uniform(50, 150, n),
        "lastPrice": np.random.uniform(1, 10, n),
        "bid": np.random.uniform(0.9, 9.9, n),
        "ask": np.random.uniform(1.1, 10.1, n),
        "impliedVolatility": np.random.uniform(0.1, 1.0, n),
        "volume": np.random.randint(10, 10000, n),
        "openInterest": np.random.randint(100, 50000, n),
        "delta": np.random.uniform(0, 1, n),
        "gamma": np.random.uniform(0, 0.1, n),
        "theta": np.random.uniform(-0.1, 0, n),
        "vega": np.random.uniform(0, 0.5, n),
        "type": np.random.choice(["call", "put"], n),
        "symbol": ["AAPL"] * n,
        "expiration": [datetime.datetime.now() + datetime.timedelta(days=30)] * n,
        "underlying": [100.0] * n,
    })
    return df

def test_benchmark_bs_call_scalar(benchmark):
    benchmark(bs_call, 100.0, 100.0, 1.0, 0.05, 0.20)

def test_benchmark_bs_call_vectorized_10k(benchmark):
    # Benchmarking vectorization over 10k rows
    S = np.linspace(50, 150, 10000)
    K = np.linspace(50, 150, 10000)
    benchmark(bs_call, S, K, 1.0, 0.05, 0.20)

def test_benchmark_enrich_and_score(benchmark, mock_options_df):
    config = {
        "score_weights": {"liquidity": 0.3, "volatility": 0.4, "greeks": 0.3},
        "moneyness_band": 0.5,
        "min_volume": 10,
        "min_oi": 100
    }
    kwargs = {
        "iv_rank": 0.5,
        "iv_percentile": 0.5,
        "sentiment_score": 0.0,
        "macro_risk_active": False,
        "sector_perf": {},
        "tnx_change_pct": 0.0
    }
    benchmark(enrich_and_score, mock_options_df, 1, 60, 0.05, config, "Single-stock", "Single-stock", **kwargs)
