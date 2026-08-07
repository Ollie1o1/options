"""Marking a book must not refetch the same option chain per call site.

`_fetch_chain_quotes` already fetches one chain per (ticker, expiration)
*within* a call — but it is invoked from several places while marking a book
(position marking, shadow marks, the risk gate), so the same pair is refetched
once per call site. Profiling a squeeze scan on 2026-08-07: 113 calls against
38 distinct (ticker, expiration) pairs in the live book, 17.0s cumulative.

The TTL is deliberately short. These are live bid/ask used to mark open
positions, and a stale mark is a worse failure than a slow one — see
docs/MARK_TRUSTWORTHINESS_SPEC.md. 60s dedupes within a single run without
carrying quotes across runs, the same reasoning as portfolio_risk._SPOT_CACHE.
"""
from __future__ import annotations

import unittest
from unittest import mock

import pandas as pd

from src import paper_manager
from src.paper_manager import PaperManager


def _chain(bid=1.00, ask=1.10):
    frame = pd.DataFrame({"strike": [100.0, 105.0], "bid": [bid, bid],
                          "ask": [ask, ask]})
    return mock.Mock(calls=frame, puts=frame)


class _Counter:
    def __init__(self):
        self.calls = []

    def Ticker(self, ticker, session=None):
        outer = self

        class _T:
            def option_chain(self, exp):
                outer.calls.append((ticker, exp))
                return _chain()
        return _T()


class TestChainQuoteMemo(unittest.TestCase):
    def setUp(self):
        paper_manager.reset_chain_quote_cache()
        self.counter = _Counter()
        patcher = mock.patch.object(paper_manager, "_get_yf_and_session",
                                    return_value=(self.counter, None))
        patcher.start()
        self.addCleanup(patcher.stop)
        self.mgr = PaperManager(db_path=":memory:")

    def test_repeat_calls_for_one_pair_fetch_once(self):
        for _ in range(4):
            self.mgr._fetch_chain_quotes("NVDA", "2030-01-18")
        self.assertEqual(len(self.counter.calls), 1)

    def test_distinct_pairs_still_fetch_separately(self):
        self.mgr._fetch_chain_quotes("NVDA", "2030-01-18")
        self.mgr._fetch_chain_quotes("NVDA", "2030-02-15")
        self.mgr._fetch_chain_quotes("AAPL", "2030-01-18")
        self.assertEqual(len(self.counter.calls), 3)

    def test_the_quotes_returned_are_unchanged(self):
        first = self.mgr._fetch_chain_quotes("NVDA", "2030-01-18")
        second = self.mgr._fetch_chain_quotes("NVDA", "2030-01-18")
        self.assertEqual(first, second)
        self.assertEqual(first[(100.0, "call")], (1.00, 1.10))

    def test_a_failed_fetch_is_not_cached(self):
        # A transient outage must not pin the pair to "no quotes" for the TTL —
        # that would silently degrade marks for a whole run.
        with mock.patch.object(paper_manager, "_get_yf_and_session",
                               side_effect=RuntimeError("network")):
            self.assertEqual(self.mgr._fetch_chain_quotes("NVDA", "2030-01-18"), {})
        got = self.mgr._fetch_chain_quotes("NVDA", "2030-01-18")
        self.assertEqual(got[(100.0, "call")], (1.00, 1.10))

    def test_reset_forces_a_refetch(self):
        self.mgr._fetch_chain_quotes("NVDA", "2030-01-18")
        paper_manager.reset_chain_quote_cache()
        self.mgr._fetch_chain_quotes("NVDA", "2030-01-18")
        self.assertEqual(len(self.counter.calls), 2)

    def test_ttl_is_short_enough_not_to_stale_a_mark(self):
        # Live bid/ask. Minutes-long caching would hand the exit logic a quote
        # from a different market.
        self.assertLessEqual(paper_manager._CHAIN_QUOTE_TTL, 120)


if __name__ == "__main__":
    unittest.main()
