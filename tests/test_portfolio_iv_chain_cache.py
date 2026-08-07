"""The portfolio GEX path must fetch each option chain once, not once per strike.

Profiling a squeeze scan on 2026-08-07: the whole run was 34.4s and 33.4s of it
was `is_risk_off_required` -> `get_portfolio_greeks` ->
`get_open_positions_with_greeks`. The squeeze scan's own fetch and scoring was
4s. The cost is `_get_current_iv`, which hits yfinance once per POSITION.

`_IV_CACHE` is keyed per contract (ticker:expiry:strike:type), but the network
work is per chain: `tkr.options` then `tkr.option_chain(exp)`. Twenty positions
on the same expiry at different strikes miss the contract cache twenty times
and refetch one chain twenty times. The live book that day: 94 open positions,
38 distinct (ticker, expiry) chains, 23 distinct tickers — 2.5x duplication on
the chain call and 4x on the expiry list.

These tests pin the chain-level dedup and, more importantly, that it changes no
returned value.
"""
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from src import portfolio_risk
from src.portfolio_risk import RiskAggregator


class _FakeTicker:
    def __init__(self, symbol, counters, ivs):
        self.symbol = symbol
        self._counters = counters
        self._ivs = ivs

    @property
    def options(self):
        self._counters["options"].append(self.symbol)
        return ("2030-01-18", "2030-02-15")

    def option_chain(self, exp):
        self._counters["chain"].append((self.symbol, exp))
        strikes = [90.0, 100.0, 110.0]
        frame = pd.DataFrame({
            "strike": strikes,
            # .get, not []: the prefetch tests use synthetic tickers outside
            # the IV table, and a KeyError here would look like a cache miss.
            "impliedVolatility": [
                self._ivs.get((self.symbol, exp, s), 0.25) for s in strikes],
        })
        return mock.Mock(calls=frame, puts=frame)


class _FakeYF:
    def __init__(self, counters, ivs):
        self._counters = counters
        self._ivs = ivs

    def Ticker(self, symbol):
        return _FakeTicker(symbol, self._counters, self._ivs)


def _ivs():
    """Distinct IV per (symbol, expiry, strike) so a mixed-up cache shows up."""
    out = {}
    for i, sym in enumerate(("NVDA", "AAPL")):
        for j, exp in enumerate(("2030-01-18", "2030-02-15")):
            for k, strike in enumerate((90.0, 100.0, 110.0)):
                out[(sym, exp, strike)] = 0.20 + i * 0.10 + j * 0.03 + k * 0.01
    return out


class _Harness(unittest.TestCase):
    def setUp(self):
        portfolio_risk.reset_spot_cache()
        portfolio_risk.reset_iv_cache()
        self.counters = {"options": [], "chain": []}
        self.ivs = _ivs()
        patcher = mock.patch.object(portfolio_risk, "_get_yf",
                                    return_value=_FakeYF(self.counters, self.ivs))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.agg = RiskAggregator(db_path=":memory:")


class TestChainFetchDedup(_Harness):
    def test_many_strikes_on_one_expiry_fetch_the_chain_once(self):
        for strike in (90.0, 100.0, 110.0):
            self.agg._get_current_iv("NVDA", "2030-01-18", strike, "call")
        self.assertEqual(len(self.counters["chain"]), 1,
                         f"chain fetched {len(self.counters['chain'])}x for one expiry")

    def test_calls_and_puts_share_one_chain_fetch(self):
        self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "put")
        self.assertEqual(len(self.counters["chain"]), 1)

    def test_expiry_list_is_fetched_once_per_ticker(self):
        self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        self.agg._get_current_iv("NVDA", "2030-02-15", 100.0, "call")
        self.assertEqual(self.counters["options"], ["NVDA"])

    def test_separate_expiries_still_fetch_separate_chains(self):
        self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        self.agg._get_current_iv("NVDA", "2030-02-15", 100.0, "call")
        self.assertEqual(len(self.counters["chain"]), 2)

    def test_separate_tickers_still_fetch_separate_chains(self):
        self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        self.agg._get_current_iv("AAPL", "2030-01-18", 100.0, "call")
        self.assertEqual(len(self.counters["chain"]), 2)


class TestValuesAreUnchanged(_Harness):
    """Caching that returns a wrong number is worse than the slow version."""

    def test_every_strike_gets_its_own_iv(self):
        got = {s: self.agg._get_current_iv("NVDA", "2030-01-18", s, "call")[0]
               for s in (90.0, 100.0, 110.0)}
        self.assertEqual(got, {90.0: self.ivs[("NVDA", "2030-01-18", 90.0)],
                               100.0: self.ivs[("NVDA", "2030-01-18", 100.0)],
                               110.0: self.ivs[("NVDA", "2030-01-18", 110.0)]})

    def test_tickers_do_not_bleed_into_each_other(self):
        nvda = self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")[0]
        aapl = self.agg._get_current_iv("AAPL", "2030-01-18", 100.0, "call")[0]
        self.assertNotAlmostEqual(nvda, aapl)
        self.assertAlmostEqual(aapl, self.ivs[("AAPL", "2030-01-18", 100.0)])

    def test_expiries_do_not_bleed_into_each_other(self):
        jan = self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")[0]
        feb = self.agg._get_current_iv("NVDA", "2030-02-15", 100.0, "call")[0]
        self.assertNotAlmostEqual(jan, feb)

    def test_source_still_reports_cache_on_the_second_contract_lookup(self):
        _, first = self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        _, second = self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        self.assertEqual(first, "market")
        self.assertEqual(second, "cache")

    def test_an_empty_chain_still_falls_back(self):
        with mock.patch.object(_FakeTicker, "option_chain",
                               return_value=mock.Mock(calls=pd.DataFrame(),
                                                      puts=pd.DataFrame())):
            iv, source = self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        self.assertEqual(source, "fallback")
        self.assertAlmostEqual(iv, RiskAggregator._FALLBACK_IV)


class TestConcurrentPrefetch(_Harness):
    """The per-ticker expiry list is the remaining serial cost.

    After chain dedup the book still makes one `tkr.options` round trip per
    distinct ticker — 23 of them at ~1s each on 2026-08-07, which was most of
    what was left. They are independent, so they are warmed concurrently before
    the sequential pricing loop, which then runs entirely out of cache.
    """

    def _trades(self, n_tickers=6):
        return [{"ticker": f"T{i}", "expiration": "2030-01-18", "strike": 100.0,
                 "type": "call", "quantity": 1} for i in range(n_tickers)]

    def test_prefetch_populates_the_expiry_cache_for_every_ticker(self):
        self.agg.prefetch_chains(self._trades())
        self.assertEqual(sorted(self.counters["options"]),
                         sorted(f"T{i}" for i in range(6)))

    def test_prefetch_is_idempotent(self):
        trades = self._trades(3)
        self.agg.prefetch_chains(trades)
        self.agg.prefetch_chains(trades)
        self.assertEqual(len(self.counters["options"]), 3)

    def test_pricing_after_prefetch_makes_no_further_expiry_calls(self):
        self.agg.prefetch_chains(self._trades(3))
        before = len(self.counters["options"])
        for i in range(3):
            self.agg._get_current_iv(f"T{i}", "2030-01-18", 100.0, "call")
        self.assertEqual(len(self.counters["options"]), before)

    def test_a_failing_ticker_does_not_sink_the_others(self):
        boom = {"T1"}
        real_ticker = _FakeYF.Ticker

        def flaky(self_yf, symbol):
            if symbol in boom:
                raise RuntimeError("network")
            return real_ticker(self_yf, symbol)

        with mock.patch.object(_FakeYF, "Ticker", flaky):
            self.agg.prefetch_chains(self._trades(3))
        self.assertIn("T0", self.counters["options"])
        self.assertIn("T2", self.counters["options"])

    def test_prefetch_of_an_empty_book_is_a_no_op(self):
        self.agg.prefetch_chains([])
        self.assertEqual(self.counters["options"], [])

    def test_prefetch_warms_the_chains_too_not_just_the_expiry_lists(self):
        # The expiry list is the cheap half. The 38 chain downloads were the
        # rest of the serial cost, so warming only `.options` moves ~2s of 13.
        self.agg.prefetch_chains(self._trades(3))
        self.assertEqual(len(self.counters["chain"]), 3,
                         "chains were not warmed by the prefetch")

    def test_pricing_after_prefetch_makes_no_network_calls_at_all(self):
        trades = self._trades(3)
        self.agg.prefetch_chains(trades)
        before = (len(self.counters["options"]), len(self.counters["chain"]))
        for t in trades:
            self.agg._get_current_iv(t["ticker"], t["expiration"], 100.0, "call")
        after = (len(self.counters["options"]), len(self.counters["chain"]))
        self.assertEqual(before, after)

    def test_prefetch_warms_spots_concurrently_too(self):
        # 23 serial spot fetches were most of what remained after the chains.
        seen = []
        with mock.patch.object(RiskAggregator, "_fetch_spot_uncached",
                               lambda self, tk: seen.append(tk) or 100.0):
            self.agg.prefetch_chains(self._trades(4))
        self.assertEqual(sorted(seen), ["T0", "T1", "T2", "T3"])

    def test_a_failing_spot_does_not_sink_the_prefetch(self):
        def boom(self, tk):
            raise RuntimeError("network")
        with mock.patch.object(RiskAggregator, "_fetch_spot_uncached", boom):
            self.agg.prefetch_chains(self._trades(3))
        self.assertEqual(len(self.counters["chain"]), 3)

    def test_warmed_chain_key_matches_the_expiry_pricing_will_ask_for(self):
        # The book's expiration need not be a listed one; _get_current_iv snaps
        # to the closest. Warming the raw string would cache a key nothing reads.
        trades = [{"ticker": "NVDA", "expiration": "2030-01-20", "strike": 100.0,
                   "type": "call", "quantity": 1}]
        self.agg.prefetch_chains(trades)
        self.assertEqual(self.counters["chain"], [("NVDA", "2030-01-18")])
        self.agg._get_current_iv("NVDA", "2030-01-20", 100.0, "call")
        self.assertEqual(len(self.counters["chain"]), 1)


class TestCacheReset(_Harness):
    def test_reset_forces_a_refetch(self):
        self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        portfolio_risk.reset_iv_cache()
        self.agg._get_current_iv("NVDA", "2030-01-18", 100.0, "call")
        self.assertEqual(len(self.counters["chain"]), 2)


if __name__ == "__main__":
    unittest.main()
