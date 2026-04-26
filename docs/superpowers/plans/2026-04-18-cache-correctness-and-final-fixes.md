# Cache Correctness & Final Fixes Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix two remaining inefficiencies and ensure all caching layers return correct, consistent data — no stale values leak between scans.

**Architecture:** Three small changes to `src/data_fetching.py`. No scoring logic changes. The audit found: (1) `_SENTIMENT_CACHE` and `_SEASONALITY_CACHE` are not cleared when `clear_chain_cache()` runs between scans, so stale sentiment/seasonality could persist across scan modes within a session; (2) `get_sector_performance` redundantly re-fetches the ticker's 5d history despite `fetch_options_yfinance` already having 1y of daily history; (3) `get_underlying_price` bypasses `_get_info_cached` in its fallback path.

**Tech Stack:** Python, yfinance, SQLite

---

## Cache Audit Results

| Cache | Scope | TTL | Cleared between scans? | Issue? |
|-------|-------|-----|----------------------|--------|
| `_CHAIN_CACHE` | in-memory | session | YES | OK |
| `_NEWS_CACHE` | in-memory | session | YES | OK |
| `_INFO_CACHE` | in-memory | session | YES | OK |
| `_FETCH_TIMESTAMPS` | in-memory | session | YES | OK |
| `_SENTIMENT_CACHE` | in-memory | session | **NO** | **Stale across scans** |
| `_SEASONALITY_CACHE` | in-memory | session | **NO** | **Stale across scans** |
| `_rfr_cache` | in-memory | 15 min | self-expiring | OK |
| `_market_context_cache` | in-memory | 15 min | self-expiring | OK |
| `requests_cache` (SQLite) | disk | 15 min | self-expiring | OK |
| `seasonality_cache` (SQLite) | disk | 7 days | self-expiring | OK |
| `iv_cache.db` | disk | append-only | N/A | OK |
| `.ai_score_cache.db` | disk | 1 trading day | self-expiring | OK |

---

### Task 1: Clear all in-memory caches in `clear_chain_cache`

**Files:**
- Modify: `src/data_fetching.py:215-220`

Add `_SENTIMENT_CACHE.clear()` and `_SEASONALITY_CACHE.clear()` to `clear_chain_cache()` so stale sentiment/seasonality don't leak between scans within a session.

---

### Task 2: Eliminate redundant history fetch in `get_sector_performance`

**Files:**
- Modify: `src/data_fetching.py:1472-1494` (function definition)
- Modify: `src/data_fetching.py:1810` (call site in parallel block)

Add an optional `hist` parameter to `get_sector_performance`. When provided, derive the ticker's 5d return from the tail of the existing 1y history instead of fetching again. The ETF history still needs a fetch. Pass `hist` from the parallel block in `fetch_options_yfinance`.

---

### Task 3: Use `_get_info_cached` in `get_underlying_price`

**Files:**
- Modify: `src/data_fetching.py:895-896`

Replace `info = ticker.info or {}` with `info = _get_info_cached(ticker.ticker, ticker)` so the fallback path doesn't make a redundant uncached API call.

---
