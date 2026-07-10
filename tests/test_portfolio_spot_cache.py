"""Spot-price fetching in the portfolio GEX path must not refetch per position.

`get_open_positions_with_greeks` used to call `_fetch_spot(ticker)` once for
EVERY open trade — 60 positions but only 29 unique tickers, so ~31 fetches were
pure duplicates — and it did so BEFORE the free local expiry check, fetching for
positions it then discarded. It also runs twice per session (startup
`update_positions` + the in-scan RISK-OFF gate), doubling the whole cost. On
2026-07-10 this was ~3.6s per call and the dominant post-input stall on a
single-ticker scan.

These tests pin: dedup within a call, a short-TTL cache across calls, and no
spot fetch for an expired position.
"""
from __future__ import annotations

import unittest

from src import portfolio_risk
from src.portfolio_risk import RiskAggregator


class _Recorder(RiskAggregator):
    """A RiskAggregator with the network stubbed and every spot fetch counted."""

    def __init__(self, trades, spot=100.0):
        super().__init__(db_path=":memory:")
        self._trades = trades
        self._spot = spot
        self.fetch_calls = []      # spot network hops
        self.iv_calls = []         # option-chain IV fetches (the ~7s hotspot)

    def _load_open_trades(self):
        return list(self._trades)

    def _fetch_spot_uncached(self, ticker):  # the real network hop, stubbed
        self.fetch_calls.append(ticker)
        return self._spot

    def _get_current_iv(self, ticker, expiration, strike, opt_type):
        self.iv_calls.append((ticker, strike, opt_type))
        return 0.30, "market"


def _trade(ticker, expiration="2030-01-18", strike=100.0, opt_type="call"):
    return {"ticker": ticker, "expiration": expiration, "strike": strike,
            "type": opt_type, "quantity": 1}


class TestSpotDedup(unittest.TestCase):
    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def test_duplicate_tickers_are_fetched_once_per_call(self):
        trades = [_trade("NVDA"), _trade("NVDA", strike=110.0),
                  _trade("AAPL"), _trade("NVDA", opt_type="put")]
        agg = _Recorder(trades)
        agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(sorted(agg.fetch_calls), ["AAPL", "NVDA"])

    def test_all_positions_still_priced_despite_dedup(self):
        trades = [_trade("NVDA"), _trade("NVDA", strike=110.0), _trade("AAPL")]
        df = _Recorder(trades).get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(len(df), 3)

    def test_a_failing_ticker_is_fetched_once_per_call_not_per_position(self):
        # None is deliberately not cached across calls, but a dead ticker held in
        # several positions must still not be retried once per position.
        agg = _Recorder([_trade("DEAD"), _trade("DEAD", strike=110.0),
                         _trade("DEAD", opt_type="put")], spot=None)
        agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(agg.fetch_calls, ["DEAD"])

    def test_expired_position_is_never_fetched(self):
        # The expiry check is free and local; it must gate the network fetch.
        trades = [_trade("NVDA", expiration="2020-01-17"), _trade("AAPL")]
        agg = _Recorder(trades)
        agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertNotIn("NVDA", agg.fetch_calls)
        self.assertIn("AAPL", agg.fetch_calls)


class TestSpotCacheAcrossCalls(unittest.TestCase):
    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def test_second_call_within_ttl_reuses_the_first_calls_spots(self):
        # The startup update_positions and the in-scan gate are two calls in one
        # process; the second must not refetch.
        trades = [_trade("NVDA"), _trade("AAPL")]
        agg = _Recorder(trades)
        agg.get_open_positions_with_greeks(rfr=0.04)
        agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(sorted(agg.fetch_calls), ["AAPL", "NVDA"])

    def test_spot_cache_expires_after_ttl(self):
        # Unit-level: exercise _fetch_spot directly so the result memo (which
        # would otherwise short-circuit a second get_open_positions call) is not
        # in the way of testing the spot TTL itself.
        agg = _Recorder([], spot=100.0)
        agg._fetch_spot("NVDA")
        portfolio_risk._SPOT_CACHE["NVDA"] = (
            100.0, portfolio_risk._time.monotonic() - portfolio_risk._SPOT_CACHE_TTL - 1)
        agg._fetch_spot("NVDA")
        self.assertEqual(agg.fetch_calls, ["NVDA", "NVDA"])

    def test_cached_spot_is_used_by_a_fresh_aggregator(self):
        # Cache is process-wide, not per-instance.
        _Recorder([_trade("NVDA")]).get_open_positions_with_greeks(rfr=0.04)
        second = _Recorder([_trade("NVDA")])
        second.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(second.fetch_calls, [])

    def test_reset_clears_the_cache(self):
        _Recorder([_trade("NVDA")]).get_open_positions_with_greeks(rfr=0.04)
        portfolio_risk.reset_spot_cache()
        agg = _Recorder([_trade("NVDA")])
        agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(agg.fetch_calls, ["NVDA"])


class TestResultMemo(unittest.TestCase):
    """The whole priced-positions frame is memoized: run_scan asks for it TWICE
    per single-ticker scan (the RISK-OFF GEX gate AND the executive-summary VaR),
    and each pass re-fetches an option chain per position (~104 fetches / 7s).
    One computation must serve both."""

    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def test_second_identical_call_does_no_fetching_at_all(self):
        agg = _Recorder([_trade("NVDA"), _trade("AAPL")])
        agg.get_open_positions_with_greeks(rfr=0.04)
        agg.get_open_positions_with_greeks(rfr=0.04)
        # The IV/option-chain fetch is the dominant cost; it must run once total.
        self.assertEqual(len(agg.iv_calls), 2)   # NVDA + AAPL, once each

    def test_second_call_returns_the_same_rows(self):
        agg = _Recorder([_trade("NVDA"), _trade("AAPL")])
        a = agg.get_open_positions_with_greeks(rfr=0.04)
        b = agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(len(a), len(b))
        self.assertEqual(list(a["ticker"]), list(b["ticker"]))

    def test_memo_invalidates_when_the_open_book_changes(self):
        # A position closing between calls must not serve a stale frame.
        agg = _Recorder([_trade("NVDA")])
        agg.get_open_positions_with_greeks(rfr=0.04)
        agg._trades = [_trade("NVDA"), _trade("AAPL")]
        agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertIn("AAPL", agg.fetch_calls)

    def test_memo_respects_ttl(self):
        agg = _Recorder([_trade("NVDA")])
        agg.get_open_positions_with_greeks(rfr=0.04)
        for k in list(portfolio_risk._POSITIONS_CACHE):
            df, _ts = portfolio_risk._POSITIONS_CACHE[k]
            portfolio_risk._POSITIONS_CACHE[k] = (
                df, portfolio_risk._time.monotonic() - portfolio_risk._POSITIONS_CACHE_TTL - 1)
        agg.get_open_positions_with_greeks(rfr=0.04)
        # The body re-ran, so IV was fetched a second time (the spot may still be
        # served from its own longer-lived cache — that is fine).
        self.assertEqual(len(agg.iv_calls), 2)

    def test_mutating_the_returned_frame_does_not_corrupt_the_memo(self):
        agg = _Recorder([_trade("NVDA")])
        a = agg.get_open_positions_with_greeks(rfr=0.04)
        a.drop(a.index, inplace=True)          # empty the caller's copy
        b = agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(len(b), 1)

    def test_reset_clears_the_result_memo_too(self):
        agg = _Recorder([_trade("NVDA")])
        agg.get_open_positions_with_greeks(rfr=0.04)
        portfolio_risk.reset_spot_cache()
        agg.get_open_positions_with_greeks(rfr=0.04)
        self.assertEqual(agg.fetch_calls, ["NVDA", "NVDA"])


class TestFetchSpotStillWorks(unittest.TestCase):
    def setUp(self):
        portfolio_risk.reset_spot_cache()

    def test_cached_fetch_spot_returns_the_value(self):
        agg = _Recorder([], spot=123.0)
        self.assertEqual(agg._fetch_spot("NVDA"), 123.0)

    def test_a_none_spot_is_not_cached(self):
        # A failed fetch must not poison the cache with None for the whole TTL.
        agg = _Recorder([])
        agg._spot = None
        self.assertIsNone(agg._fetch_spot("NVDA"))
        self.assertNotIn("NVDA", portfolio_risk._SPOT_CACHE)


if __name__ == "__main__":
    unittest.main()
