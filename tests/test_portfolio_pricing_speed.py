"""Marking the book costs one request per CHAIN, not one per leg.

`view_portfolio` priced every open leg with its own `yf.Ticker(occ)` lookup —
one network round trip per leg, eight at a time. On the live book that is 119
open positions, 87 of them four-legged iron condors, so ~381 lookups. It ran
for over eleven minutes with 10.9 seconds of CPU: almost entirely network wait,
and it had to be killed.

Those 381 legs sit on only **41 distinct (ticker, expiration) pairs** — QQQ
2026-09-18 alone carries 15 positions. `PaperManager._fetch_chain_quotes`
already serves every leg on a pair from ONE `option_chain` request and memoises
it for 60s, which is the same fix the scan path got when the GEX path was
taking 90% of every scan.

So the round trips drop ~9x. These tests pin the property that matters — the
number of REQUESTS, not the wall clock, because a timing assertion on a network
path is a flake generator.
"""
from __future__ import annotations

import unittest


def _condor(ticker, exp, base):
    return {"ticker": ticker, "expiration": exp, "strategy_name": "Iron Condor",
            "strike": base, "short_put_strike": base, "long_put_strike": base - 5,
            "short_call_strike": base + 20, "long_call_strike": base + 25,
            "type": "put", "entry_price": 1.0}


def _call(ticker, exp, strike):
    return {"ticker": ticker, "expiration": exp, "strategy_name": "Long Call",
            "strike": strike, "type": "call", "entry_price": 2.0}


class _SpyPM:
    """Counts chain requests and answers every strike on the pair."""

    def __init__(self):
        self.calls = []

    def _fetch_chain_quotes(self, ticker, expiration):
        self.calls.append((ticker, str(expiration)[:10]))
        # Every strike this test asks for, both types.
        return {(float(s), t): (1.00, 1.10)
                for s in range(0, 1000, 5) for t in ("call", "put")}


class TestOneRequestPerChain(unittest.TestCase):

    def _trades(self):
        # 6 positions, 21 legs, on 2 distinct (ticker, expiration) pairs.
        # Bases stepped by 5 so every derived leg (base, base-5, base+20,
        # base+25) lands on the spy's strike grid — otherwise the fixture,
        # not the code, is what fails to price a leg.
        return ([_condor("QQQ", "2026-09-18", 700 + i * 5) for i in range(4)]
                + [_call("QQQ", "2026-09-18", 740),
                   _call("SPY", "2026-10-16", 650)])

    def test_it_fetches_one_chain_per_ticker_expiration(self):
        from src.check_pnl import _price_open_legs
        pm = _SpyPM()
        _price_open_legs(self._trades(), pm=pm)
        self.assertEqual(sorted(set(pm.calls)),
                         [("QQQ", "2026-09-18"), ("SPY", "2026-10-16")])

    def test_it_does_not_fetch_once_per_leg(self):
        """The defect: 21 legs used to mean 21 round trips."""
        from src.check_pnl import _price_open_legs
        pm = _SpyPM()
        _price_open_legs(self._trades(), pm=pm)
        self.assertEqual(len(pm.calls), 2,
                         f"{len(pm.calls)} requests for 2 distinct chains")

    def test_it_does_not_refetch_the_same_pair(self):
        from src.check_pnl import _price_open_legs
        pm = _SpyPM()
        _price_open_legs(self._trades(), pm=pm)
        self.assertEqual(len(pm.calls), len(set(pm.calls)))

    def test_every_leg_still_gets_a_mark(self):
        from src.check_pnl import _price_open_legs, _legs_for_row
        pm = _SpyPM()
        trades = self._trades()
        prices = _price_open_legs(trades, pm=pm)
        for r in trades:
            for opt_type, strike, _q in _legs_for_row(r):
                key = (r["ticker"], r["expiration"][:10], strike, opt_type)
                self.assertIn(key, prices, f"{key} was not priced")
                self.assertAlmostEqual(prices[key], 1.05, places=6)  # mid


class TestItDegradesRatherThanBreaks(unittest.TestCase):

    def test_an_empty_book_makes_no_requests(self):
        from src.check_pnl import _price_open_legs
        pm = _SpyPM()
        self.assertEqual(_price_open_legs([], pm=pm), {})
        self.assertEqual(pm.calls, [])

    def test_a_chain_that_returns_nothing_leaves_legs_unpriced(self):
        """A missing mark must stay missing — the caller has traded-price
        rungs to fall back on, and a fabricated mark could fire an exit."""
        from src.check_pnl import _price_open_legs

        class _Empty:
            def _fetch_chain_quotes(self, t, e):
                return {}

        prices = _price_open_legs([_call("SPY", "2026-10-16", 650)], pm=_Empty())
        self.assertTrue(all(v is None for v in prices.values()) or prices == {})

    def test_a_raising_chain_does_not_abort_the_whole_book(self):
        from src.check_pnl import _price_open_legs

        class _Boom:
            def __init__(self): self.n = 0
            def _fetch_chain_quotes(self, t, e):
                self.n += 1
                if t == "QQQ":
                    raise RuntimeError("feed down")
                return {(650.0, "call"): (2.0, 2.2)}

        pm = _Boom()
        prices = _price_open_legs(
            [_call("QQQ", "2026-09-18", 740), _call("SPY", "2026-10-16", 650)], pm=pm)
        self.assertAlmostEqual(prices[("SPY", "2026-10-16", 650.0, "call")], 2.1,
                               places=6)


if __name__ == "__main__":
    unittest.main()


class TestAMalformedChainDoesNotAbortMarking(unittest.TestCase):
    """`view_portfolio` is rendered in tests with `PaperManager` mocked, so the
    chain helper can return something that is not a {(strike, type): (bid, ask)}
    mapping at all. That must leave legs unpriced, not raise."""

    def test_a_non_mapping_quote_leaves_the_leg_unpriced(self):
        from unittest import mock
        from src.check_pnl import _price_open_legs

        class _Mocky:
            def _fetch_chain_quotes(self, t, e):
                return mock.MagicMock()

        prices = _price_open_legs([_call("SPY", "2026-10-16", 650)], pm=_Mocky())
        self.assertIsNone(prices[("SPY", "2026-10-16", 650.0, "call")])

    def test_a_wrong_shaped_tuple_leaves_the_leg_unpriced(self):
        from src.check_pnl import _price_open_legs

        class _Odd:
            def _fetch_chain_quotes(self, t, e):
                return {(650.0, "call"): (1.0, 2.0, 3.0)}

        prices = _price_open_legs([_call("SPY", "2026-10-16", 650)], pm=_Odd())
        self.assertIsNone(prices[("SPY", "2026-10-16", 650.0, "call")])
