"""Tests for in-session chain cache in src/data_fetching.py."""


def test_chain_cache_hit_returns_cached():
    """fetch_options_yfinance should return the cached value without re-fetching."""
    from src.data_fetching import fetch_options_yfinance, clear_chain_cache, _CHAIN_CACHE
    clear_chain_cache()
    sentinel = {"df": "cached_value", "context": {"test": True}}
    _CHAIN_CACHE["TEST_SYMBOL"] = sentinel
    result = fetch_options_yfinance("TEST_SYMBOL", 2)
    assert result is sentinel


def test_clear_chain_cache_empties_cache():
    """clear_chain_cache() should remove all entries from _CHAIN_CACHE."""
    from src.data_fetching import clear_chain_cache, _CHAIN_CACHE
    _CHAIN_CACHE["AAPL"] = {"df": None, "context": {}}
    _CHAIN_CACHE["MSFT"] = {"df": None, "context": {}}
    assert len(_CHAIN_CACHE) >= 2
    clear_chain_cache()
    assert len(_CHAIN_CACHE) == 0
