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


def test_news_cache_deduplicates():
    """_get_news_cached should return the same list object on repeated calls."""
    from src.data_fetching import _get_news_cached, _NEWS_CACHE
    _NEWS_CACHE.clear()
    _NEWS_CACHE["TEST"] = [{"title": "Test headline"}]
    result1 = _get_news_cached("TEST", ticker=None)
    result2 = _get_news_cached("TEST", ticker=None)
    assert result1 is result2
    assert result1 == [{"title": "Test headline"}]


def test_info_cache_deduplicates():
    """_get_info_cached should return the same dict on repeated calls."""
    from src.data_fetching import _get_info_cached, _INFO_CACHE
    _INFO_CACHE.clear()
    _INFO_CACHE["TEST"] = {"shortPercentOfFloat": 0.05, "dividendYield": 1.2}
    result1 = _get_info_cached("TEST", ticker=None)
    result2 = _get_info_cached("TEST", ticker=None)
    assert result1 is result2
    assert result1["shortPercentOfFloat"] == 0.05


import os

def test_seasonality_sqlite_cache_hit():
    """check_seasonality should return cached value from SQLite when fresh."""
    from src.data_fetching import _read_seasonality_cache, _write_seasonality_cache
    db = "/tmp/test_seasonality_cache.db"
    if os.path.exists(db):
        os.remove(db)
    _write_seasonality_cache("AAPL", 4, 0.65, db)
    result = _read_seasonality_cache("AAPL", 4, db)
    assert result == 0.65
    if os.path.exists(db):
        os.remove(db)


def test_seasonality_sqlite_cache_miss():
    """_read_seasonality_cache returns None when no entry exists."""
    from src.data_fetching import _read_seasonality_cache
    db = "/tmp/test_seasonality_miss.db"
    if os.path.exists(db):
        os.remove(db)
    result = _read_seasonality_cache("AAPL", 4, db)
    assert result is None
    if os.path.exists(db):
        os.remove(db)
