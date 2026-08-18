"""A missing spot price must fail loudly at the source, not 170 lines later.

Observed live 2026-08-18 00:25-00:27, on 13 large caps in three minutes
(AVGO, CSCO, QCOM, MS, EOG, PSX, SCHW, TXN, SLB, BLK, WFC, C, XYZ):

    File "src/data_fetching.py", line 2489, in fetch_options_yfinance
        earnings_move_data = calculate_implied_earnings_move(...)
    File "src/data_fetching.py", line 2146, in calculate_implied_earnings_move
        if earnings_date is None or df_chain.empty or underlying <= 0:
    TypeError: '<=' not supported between instances of 'NoneType' and 'int'

`hist` was non-empty but its last Close was NaN — which yfinance returns
instead of an empty frame when it is rate-limited or serving a partial
response. `safe_float(NaN)` is None by design, so `underlying` became None and
travelled ~170 lines before a comparison finally raised. The retry decorator
then burned four attempts per symbol on an error that could never succeed, and
the whole ticker dropped out of the scan.

The yahooquery path in the same module never had this bug: it checks the
resolved price and falls back to the live quote (`data_fetching.py:498`). The
yfinance path did neither, and annotated the parameter `underlying: float`
while reaching it with None.
"""
from __future__ import annotations

import unittest
from datetime import datetime

import numpy as np
import pandas as pd


class TestTheCrashSiteToleratesAMissingPrice(unittest.TestCase):
    """Defense in depth. This is where it blew up, so this is where a None
    must stop being a crash — but it is NOT the root cause, and fixing only
    this would hand every downstream consumer a chain priced against nothing.
    """

    def _chain(self):
        return pd.DataFrame({
            "expiration": ["2026-09-18"],
            "strike": [100.0],
            "type": ["call"],
            "lastPrice": [1.0],
        })

    def test_a_none_underlying_returns_none_rather_than_raising(self):
        from src.data_fetching import calculate_implied_earnings_move
        out = calculate_implied_earnings_move(
            None, datetime(2026, 9, 1), self._chain(), None)
        self.assertIsNone(out)

    def test_a_zero_underlying_still_returns_none(self):
        """The pre-existing guard must not regress while widening it."""
        from src.data_fetching import calculate_implied_earnings_move
        out = calculate_implied_earnings_move(
            None, datetime(2026, 9, 1), self._chain(), 0.0)
        self.assertIsNone(out)


class TestANaNCloseIsResolvedOrRefusedAtTheSource(unittest.TestCase):
    """The root cause. A frame whose last Close is NaN must either resolve a
    price from the live quote — the yahooquery path's behaviour — or refuse the
    symbol with a clear error. It must never return a chain built around a
    None spot, and must never raise TypeError from a distant comparison.
    """

    def _nan_close_history(self):
        idx = pd.date_range("2026-08-01", periods=5, freq="D", tz="UTC")
        return pd.DataFrame(
            {"Open": [10.0] * 5, "High": [11.0] * 5, "Low": [9.0] * 5,
             "Close": [10.0, 10.0, 10.0, 10.0, np.nan],
             "Volume": [1000] * 5},
            index=idx)

    def _run_fetch(self, monkey_price):
        """Drive fetch_options_yfinance far enough to resolve the spot."""
        from src import data_fetching as dfetch
        orig = {
            "_init_yfinance": dfetch._init_yfinance,
            "_init_yf_session": dfetch._init_yf_session,
            "_init_request_cache": dfetch._init_request_cache,
            "_yf_cached": dfetch._yf_cached,
            "get_underlying_price": dfetch.get_underlying_price,
            "yf": dfetch.yf,
        }
        hist = self._nan_close_history()

        class _FakeTicker:
            def __init__(self, *a, **k):
                self.ticker = "TESTSYM"

            def history(self, *a, **k):
                return hist

        class _FakeYF:
            Ticker = _FakeTicker

        dfetch._init_yfinance = lambda: None
        dfetch._init_yf_session = lambda: None
        dfetch._init_request_cache = lambda: None
        dfetch._yf_cached = lambda sym, key, fn, ttl_s=None: fn()
        dfetch.get_underlying_price = lambda tkr: monkey_price
        dfetch.yf = _FakeYF()
        dfetch.clear_chain_cache()
        try:
            return dfetch.fetch_options_yfinance.__wrapped__("TESTSYM", 2)
        finally:
            for k, v in orig.items():
                setattr(dfetch, k, v)
            dfetch.clear_chain_cache()

    def test_a_nan_close_does_not_raise_a_typeerror(self):
        """The symptom. Whatever happens, it is not an unhandled TypeError
        about NoneType and int."""
        try:
            self._run_fetch(monkey_price=None)
        except TypeError as e:
            self.fail(f"NaN close still produces a TypeError: {e}")
        except Exception:
            pass  # a clear refusal is the acceptable outcome — asserted below

    def test_an_unresolvable_price_refuses_the_symbol_clearly(self):
        from src import data_fetching as dfetch
        with self.assertRaises(RuntimeError) as ctx:
            self._run_fetch(monkey_price=None)
        msg = str(ctx.exception).lower()
        self.assertIn("price", msg)
        self.assertIn("testsym", msg.upper().lower())

    def test_the_live_quote_rescues_a_nan_close(self):
        """The yahooquery path's behaviour: a NaN last close is recoverable
        from the quote, and a recoverable symbol must not be dropped."""
        from src import data_fetching as dfetch
        try:
            self._run_fetch(monkey_price=123.45)
        except RuntimeError as e:
            if "price" in str(e).lower():
                self.fail("a resolvable price was refused: " + str(e))
        except Exception:
            pass  # failing later (no options chain) is fine; refusing is not


if __name__ == "__main__":
    unittest.main()
