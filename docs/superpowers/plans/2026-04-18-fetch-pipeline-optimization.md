# Fetch Pipeline Optimization Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut per-ticker fetch time by ~40% by parallelizing sequential network calls, deduplicating redundant API hits, caching stable data, and raising the multi-ticker concurrency cap.

**Architecture:** Four changes to `src/data_fetching.py` and one to `src/options_screener.py`. All changes are in the I/O layer — scoring logic is untouched. The per-ticker fetch function (`fetch_options_yfinance`) currently makes 7+ sequential network calls for auxiliary data (earnings, sentiment, news, seasonality, sector perf, short interest, news aggregation). These are all independent and can run concurrently via a thread pool. Additionally, `get_sentiment` and `get_news_headlines` both call `ticker.news` separately (duplicate API hit), `get_short_interest` and dividend-yield both call `ticker.info` separately (duplicate API hit), seasonality fetches 5 years of monthly data every session despite it barely changing, and the multi-ticker scan caps at 8 workers for 100+ tickers.

**Tech Stack:** Python `concurrent.futures.ThreadPoolExecutor`, SQLite (existing `iv_cache.db`), yfinance

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `src/data_fetching.py` | Modify | All 4 optimizations live here |
| `src/options_screener.py` | Modify | Raise multi-ticker worker cap |
| `tests/test_data_fetching.py` | Modify | Add tests for new helpers |

---

### Task 1: Deduplicate `ticker.news` calls

`get_sentiment()` (line 918) and `get_news_headlines()` (line 935) both call `ticker.news`. Extract a shared `_get_news_cached()` helper that fetches once per ticker per session and returns the cached list.

**Files:**
- Modify: `src/data_fetching.py:907-946`
- Test: `tests/test_data_fetching.py`

- [ ] **Step 1: Write the failing test**

```python
def test_news_cache_deduplicates():
    """_get_news_cached should return the same list object on repeated calls."""
    from src.data_fetching import _get_news_cached, _NEWS_CACHE
    _NEWS_CACHE.clear()
    # Stuff a fake entry
    _NEWS_CACHE["TEST"] = [{"title": "Test headline"}]
    result1 = _get_news_cached("TEST", ticker=None)
    result2 = _get_news_cached("TEST", ticker=None)
    assert result1 is result2
    assert result1 == [{"title": "Test headline"}]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_data_fetching.py::test_news_cache_deduplicates -v`
Expected: FAIL — `_get_news_cached` not defined

- [ ] **Step 3: Implement `_get_news_cached` and `_NEWS_CACHE`**

In `src/data_fetching.py`, near the other cache dicts (line ~113), add:

```python
_NEWS_CACHE: Dict[str, list] = {}
```

Then add the helper function before `get_sentiment`:

```python
def _get_news_cached(symbol: str, ticker: "yf.Ticker") -> list:
    """Fetch ticker.news once per session, cache result."""
    if symbol in _NEWS_CACHE:
        return _NEWS_CACHE[symbol]
    try:
        news = ticker.news if ticker else []
    except Exception:
        news = []
    _NEWS_CACHE[symbol] = news or []
    return _NEWS_CACHE[symbol]
```

Then update `get_sentiment` to use it:

```python
def get_sentiment(ticker: yf.Ticker) -> Optional[float]:
    try:
        from textblob import TextBlob
    except ImportError:
        return None

    key = f"{ticker.ticker}:sentiment"
    if key in _SENTIMENT_CACHE:
        return _SENTIMENT_CACHE[key]

    try:
        news = _get_news_cached(ticker.ticker, ticker)
        if not news:
            return None
        headlines = ". ".join([item.get("title", "") for item in news])
        if not headlines.strip():
            return None
        blob = TextBlob(headlines)
        score = blob.sentiment.polarity
        _SENTIMENT_CACHE[key] = score
        return score
    except Exception as exc:
        logger.debug("Sentiment analysis failed: %s", exc)
        return None
```

And update `get_news_headlines`:

```python
def get_news_headlines(ticker: yf.Ticker, max_headlines: int = 3) -> list:
    """Return up to max_headlines recent news titles for a ticker."""
    try:
        news = _get_news_cached(ticker.ticker, ticker)
        if not news:
            return []
        titles = []
        for item in news[:max_headlines]:
            title = item.get("title") or item.get("content", {}).get("title", "")
            if title:
                titles.append(title.strip())
        return titles[:max_headlines]
    except Exception as exc:
        logger.debug("News headlines fetch failed: %s", exc)
        return []
```

Also add `_NEWS_CACHE` to `clear_chain_cache`:

```python
def clear_chain_cache() -> None:
    _CHAIN_CACHE.clear()
    _NEWS_CACHE.clear()
    _INFO_CACHE.clear()  # added in Task 2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_data_fetching.py::test_news_cache_deduplicates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_fetching.py tests/test_data_fetching.py
git commit -m "perf: deduplicate ticker.news calls with _NEWS_CACHE"
```

---

### Task 2: Deduplicate `ticker.info` calls

`get_short_interest()` (line 1387) and the dividend-yield block (line 1862) both call `ticker.info`. Extract `_get_info_cached()` so `ticker.info` is fetched once per ticker.

**Files:**
- Modify: `src/data_fetching.py:1384-1398, 1859-1872`
- Test: `tests/test_data_fetching.py`

- [ ] **Step 1: Write the failing test**

```python
def test_info_cache_deduplicates():
    """_get_info_cached should return the same dict on repeated calls."""
    from src.data_fetching import _get_info_cached, _INFO_CACHE
    _INFO_CACHE.clear()
    _INFO_CACHE["TEST"] = {"shortPercentOfFloat": 0.05, "dividendYield": 1.2}
    result1 = _get_info_cached("TEST", ticker=None)
    result2 = _get_info_cached("TEST", ticker=None)
    assert result1 is result2
    assert result1["shortPercentOfFloat"] == 0.05
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/python -m pytest tests/test_data_fetching.py::test_info_cache_deduplicates -v`
Expected: FAIL — `_get_info_cached` not defined

- [ ] **Step 3: Implement `_get_info_cached` and `_INFO_CACHE`**

Near the other cache dicts:

```python
_INFO_CACHE: Dict[str, dict] = {}
```

Helper:

```python
def _get_info_cached(symbol: str, ticker: "yf.Ticker") -> dict:
    """Fetch ticker.info once per session, cache result."""
    if symbol in _INFO_CACHE:
        return _INFO_CACHE[symbol]
    try:
        info = (ticker.info if ticker else None) or {}
    except Exception:
        info = {}
    _INFO_CACHE[symbol] = info
    return _INFO_CACHE[symbol]
```

Update `get_short_interest`:

```python
def get_short_interest(ticker: yf.Ticker) -> Optional[float]:
    """Fetch short interest (shortPercentOfFloat) from ticker info."""
    try:
        info = _get_info_cached(ticker.ticker, ticker)
        si = info.get("shortPercentOfFloat", None)
        if si is None:
            return None
        si = float(si)
        if not (0 <= si <= 200):
            return None
        if si > 1:
            si = si / 100.0
        return si
    except Exception:
        return None
```

Update the dividend-yield block in `fetch_options_yfinance` (line ~1859):

```python
    # Dividend yield from ticker.info
    dividend_yield = 0.0
    try:
        _info = _get_info_cached(symbol, tkr)
        _dy = _info.get("dividendYield", 0)
        dividend_yield = float(_dy) if _dy else 0.0
        if dividend_yield > 0.20:
            dividend_yield /= 100.0
        dividend_yield = min(dividend_yield, 0.15)
    except Exception:
        dividend_yield = 0.0
    df["dividend_yield"] = dividend_yield
```

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/python -m pytest tests/test_data_fetching.py::test_info_cache_deduplicates -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_fetching.py tests/test_data_fetching.py
git commit -m "perf: deduplicate ticker.info calls with _INFO_CACHE"
```

---

### Task 3: Cache seasonality in SQLite

`check_seasonality()` fetches 5 years of monthly data every session. This data changes at most once per month. Cache results in `iv_cache.db` with a 7-day TTL.

**Files:**
- Modify: `src/data_fetching.py:948-968`
- Test: `tests/test_data_fetching.py`

- [ ] **Step 1: Write the failing test**

```python
import sqlite3
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_data_fetching.py::test_seasonality_sqlite_cache_hit tests/test_data_fetching.py::test_seasonality_sqlite_cache_miss -v`
Expected: FAIL — functions not defined

- [ ] **Step 3: Implement SQLite seasonality cache**

Add near the other cache helpers in `src/data_fetching.py`:

```python
def _read_seasonality_cache(symbol: str, month: int, db_path: str = "iv_cache.db") -> Optional[float]:
    """Read cached seasonality win rate. Returns None if stale (>7 days) or missing."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS seasonality_cache (
            symbol TEXT, month INTEGER, win_rate REAL, updated TEXT,
            PRIMARY KEY (symbol, month))""")
        row = conn.execute(
            "SELECT win_rate, updated FROM seasonality_cache WHERE symbol=? AND month=?",
            (symbol, month)
        ).fetchone()
        conn.close()
        if row is None:
            return None
        from datetime import datetime, timedelta
        updated = datetime.fromisoformat(row[1])
        if datetime.now() - updated > timedelta(days=7):
            return None
        return row[0]
    except Exception:
        return None


def _write_seasonality_cache(symbol: str, month: int, win_rate: float, db_path: str = "iv_cache.db") -> None:
    """Write seasonality win rate to SQLite cache."""
    try:
        conn = sqlite3.connect(db_path)
        conn.execute("""CREATE TABLE IF NOT EXISTS seasonality_cache (
            symbol TEXT, month INTEGER, win_rate REAL, updated TEXT,
            PRIMARY KEY (symbol, month))""")
        conn.execute(
            "INSERT OR REPLACE INTO seasonality_cache (symbol, month, win_rate, updated) VALUES (?, ?, ?, ?)",
            (symbol, month, win_rate, datetime.now().isoformat())
        )
        conn.commit()
        conn.close()
    except Exception:
        pass
```

Then update `check_seasonality`:

```python
@retry_with_backoff(retries=2, backoff_in_seconds=1)
def check_seasonality(ticker: yf.Ticker) -> Optional[float]:
    key = f"{ticker.ticker}:seasonality"
    if key in _SEASONALITY_CACHE:
        return _SEASONALITY_CACHE[key]
    current_month = datetime.now().month
    # Check SQLite cache first (7-day TTL)
    cached = _read_seasonality_cache(ticker.ticker, current_month)
    if cached is not None:
        _SEASONALITY_CACHE[key] = cached
        return cached
    try:
        hist = ticker.history(period="5y", interval="1mo")
        if hist.empty:
            return None
        monthly_data = hist[hist.index.month == current_month]
        if monthly_data.empty:
            return None
        wins = (monthly_data['Close'] > monthly_data['Open']).sum()
        total = len(monthly_data)
        win_rate = wins / total if total > 0 else 0.0
        _SEASONALITY_CACHE[key] = win_rate
        _write_seasonality_cache(ticker.ticker, current_month, win_rate)
        return win_rate
    except Exception as exc:
        logger.debug("Seasonality check failed for %s: %s", ticker.ticker, exc)
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_data_fetching.py -k seasonality -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/data_fetching.py tests/test_data_fetching.py
git commit -m "perf: cache seasonality in SQLite with 7-day TTL"
```

---

### Task 4: Parallelize per-ticker auxiliary fetches

The main optimization. In `fetch_options_yfinance`, lines 1724-1741 make 7 sequential calls. Wrap them in a `ThreadPoolExecutor` so they run concurrently. The history fetch (step 2) must complete first since some auxiliary functions use `tkr` but NOT the history. The option chain fetch (step 4) is already parallel. So we parallelize step 3 (the auxiliary data calls).

**Files:**
- Modify: `src/data_fetching.py:1724-1741`

- [ ] **Step 1: Refactor the sequential block into parallel execution**

Replace the sequential block (lines 1724-1741) in `fetch_options_yfinance`:

```python
    # 3. Fetch Other Data (Earnings, Sentiment, Seasonality) — PARALLEL
    from concurrent.futures import ThreadPoolExecutor, as_completed

    _aux_results: Dict[str, Any] = {}
    def _aux(name, fn, *args, **kwargs):
        try:
            return name, fn(*args, **kwargs)
        except Exception:
            return name, None

    with ThreadPoolExecutor(max_workers=6) as aux_pool:
        aux_futures = [
            aux_pool.submit(_aux, "earnings", get_next_earnings_date, tkr),
            aux_pool.submit(_aux, "sentiment", get_sentiment, tkr),
            aux_pool.submit(_aux, "news", get_news_headlines, tkr),
            aux_pool.submit(_aux, "seasonality", check_seasonality, tkr),
            aux_pool.submit(_aux, "sector", get_sector_performance, symbol),
            aux_pool.submit(_aux, "short_interest", get_short_interest, tkr),
        ]
        for fut in as_completed(aux_futures):
            name, val = fut.result()
            _aux_results[name] = val

    earnings_date = _aux_results.get("earnings")
    sentiment_score = _aux_results.get("sentiment")
    news_headlines = _aux_results.get("news") or []
    seasonal_win_rate = _aux_results.get("seasonality")
    sector_perf = _aux_results.get("sector") or {}
    short_interest = _aux_results.get("short_interest")

    # Rich news + analyst data (uses cached news from _get_news_cached)
    news_data = None
    if _HAS_NEWS_FETCHER:
        try:
            news_data = fetch_news_and_events(symbol, ticker_obj=tkr, max_age_hours=72, max_headlines=5)
            if news_data and news_data.top_headlines:
                news_headlines = news_data.top_headlines
        except Exception:
            pass
```

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `./venv/bin/python -m pytest tests/test_data_fetching.py -v`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/data_fetching.py
git commit -m "perf: parallelize per-ticker auxiliary fetches (6 concurrent)"
```

---

### Task 5: Raise multi-ticker concurrency cap

For DISCOVER scans with 100+ tickers, 8 workers is conservative. Bump to 12 since the work is I/O-bound.

**Files:**
- Modify: `src/options_screener.py:2741`

- [ ] **Step 1: Update the worker cap**

Change line 2741 in `src/options_screener.py` from:

```python
        with ThreadPoolExecutor(max_workers=min(len(tickers), 8)) as executor:
```

to:

```python
        with ThreadPoolExecutor(max_workers=min(len(tickers), 12)) as executor:
```

- [ ] **Step 2: Run existing tests to verify no regressions**

Run: `./venv/bin/python -m pytest tests/ -v --timeout=30`
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add src/options_screener.py
git commit -m "perf: raise multi-ticker scan concurrency from 8 to 12 workers"
```

---

### Task 6: Verify end-to-end

- [ ] **Step 1: Run full test suite**

Run: `./venv/bin/python -m pytest tests/ -v`
Expected: All PASS

- [ ] **Step 2: Smoke test a single-ticker scan**

Run: `./venv/bin/python -c "from src.data_fetching import fetch_options_yfinance; import time; t=time.time(); r=fetch_options_yfinance('AAPL', 2); print(f'AAPL fetch: {time.time()-t:.1f}s, rows: {len(r[\"df\"])}')"` 
Expected: completes in under 10s, returns rows > 0

- [ ] **Step 3: Commit all and push**

```bash
git push
```
