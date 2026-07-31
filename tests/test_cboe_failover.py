"""CBOE failover: serve a scan from the second source when Yahoo's chain fails.

Two properties carry the weight. First, the failover must be OFF for broad
sweeps — CBOE is a free unauthenticated endpoint and a top-100 scan would hammer
it. Second, a chain served from CBOE must be visibly labelled: it is delayed and
mid-priced rather than last-traded, and a reader who mistakes it for Yahoo's
book is reading prices that were never printed.
"""
import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import data_fetching as dfetch  # noqa: E402
from src.cli_display import format_data_quality_summary  # noqa: E402


def _cboe_row(strike=100.0, opt_type="call", exp="2026-08-21", bid=1.0, ask=1.2):
    return {"symbol": "AAPL", "contract": "AAPL260821C00100000", "type": opt_type,
            "strike": strike, "expiration": exp, "bid": bid, "ask": ask,
            "iv": 0.32, "delta": 0.5, "gamma": 0.01, "theta": -0.05, "vega": 0.1,
            "open_interest": 400, "volume": 25, "last_trade_time": "2026-07-31T15:45:00",
            "spot": 101.0, "source": "cboe"}


class MidPricingTest(unittest.TestCase):
    def test_mid_of_a_two_sided_book(self):
        self.assertAlmostEqual(dfetch._cboe_mid(1.0, 1.4), 1.2)

    def test_one_sided_book_uses_the_side_that_exists(self):
        self.assertEqual(dfetch._cboe_mid(None, 1.4), 1.4)
        self.assertEqual(dfetch._cboe_mid(1.0, None), 1.0)

    def test_crossed_or_empty_book_yields_none(self):
        # ask < bid is a crossed quote — not a price to trade off.
        self.assertIsNone(dfetch._cboe_mid(2.0, 1.0))
        self.assertIsNone(dfetch._cboe_mid(0, 0))
        self.assertIsNone(dfetch._cboe_mid(None, None))


class FrameMappingTest(unittest.TestCase):
    def setUp(self):
        self._orig = getattr(dfetch, "_cboe_frames").__globals__
        import src.cboe_client as cc
        self._saved_fetch = cc.fetch_chain

    def tearDown(self):
        import src.cboe_client as cc
        cc.fetch_chain = self._saved_fetch

    def _patch(self, rows):
        import src.cboe_client as cc
        cc.fetch_chain = lambda symbol, *a, **k: rows

    def test_maps_into_the_yfinance_column_shape(self):
        self._patch([_cboe_row(), _cboe_row(opt_type="put")])
        frames = dfetch._cboe_frames("AAPL", ["2026-08-21"])
        self.assertEqual(len(frames), 2)
        df = pd.concat(frames, ignore_index=True)
        for col in ("strike", "bid", "ask", "lastPrice", "volume",
                    "openInterest", "impliedVolatility", "type",
                    "expiration", "symbol", "quote_source"):
            self.assertIn(col, df.columns, f"missing {col}")
        self.assertTrue((df["quote_source"] == "cboe").all())

    def test_iv_is_passed_through_undivided(self):
        # CBOE and yfinance both express IV as a decimal — cross_check compares
        # them with no scaling. Dividing by 100 here would silently make every
        # CBOE-served chain look like a 0.3-vol world.
        self._patch([_cboe_row()])
        df = pd.concat(dfetch._cboe_frames("AAPL", []), ignore_index=True)
        self.assertAlmostEqual(float(df["impliedVolatility"].iloc[0]), 0.32)

    def test_uncomputed_iv_becomes_missing_not_zero(self):
        # CBOE returns iv == 0.0 for strikes it does not solve (17% of a live
        # AAPL chain). Passing 0.0 through would tell the scorer the contract
        # cannot move, rather than that the vol is unknown.
        self._patch([_cboe_row()])
        rows = dfetch._cboe_frames("AAPL", [])
        df = pd.concat(rows, ignore_index=True)
        self.assertAlmostEqual(float(df["impliedVolatility"].iloc[0]), 0.32)

        self.assertIsNone(dfetch._cboe_iv(0.0))
        self.assertIsNone(dfetch._cboe_iv(None))
        self.assertIsNone(dfetch._cboe_iv(-0.1))
        self.assertAlmostEqual(dfetch._cboe_iv(0.32), 0.32)

    def test_expirations_outside_the_requested_set_are_dropped(self):
        self._patch([_cboe_row(exp="2026-08-21"), _cboe_row(exp="2027-01-15")])
        df = pd.concat(dfetch._cboe_frames("AAPL", ["2026-08-21"]), ignore_index=True)
        self.assertEqual(set(df["expiration"]), {"2026-08-21"})

    def test_an_empty_or_failing_source_returns_no_frames(self):
        self._patch([])
        self.assertEqual(dfetch._cboe_frames("AAPL", []), [])

        def _boom(symbol, *a, **k):
            raise RuntimeError("cboe down")
        import src.cboe_client as cc
        cc.fetch_chain = _boom
        # The failover must never turn a Yahoo outage into a second exception.
        self.assertEqual(dfetch._cboe_frames("AAPL", []), [])


class GatingTest(unittest.TestCase):
    def tearDown(self):
        dfetch.set_cboe_fallback(False)

    def test_default_is_off(self):
        # Import-time default: a broad sweep that never sets the flag must not
        # inherit an enabled failover from a previous scan.
        self.assertFalse(dfetch.cboe_fallback_enabled())

    def test_toggle_round_trips(self):
        dfetch.set_cboe_fallback(True)
        self.assertTrue(dfetch.cboe_fallback_enabled())
        dfetch.set_cboe_fallback(False)
        self.assertFalse(dfetch.cboe_fallback_enabled())


class DataQualityLineTest(unittest.TestCase):
    def test_cboe_rows_are_called_out_beside_the_freshness_counts(self):
        df = pd.DataFrame({"quote_freshness": ["delayed", "delayed"],
                           "quote_source": ["cboe", "cboe"]})
        line = format_data_quality_summary(df)
        self.assertIn("2 delayed", line)
        self.assertIn("CBOE fallback", line)
        self.assertIn("not last trade", line)

    def test_a_normal_yahoo_scan_says_nothing_about_cboe(self):
        df = pd.DataFrame({"quote_freshness": ["fresh", "delayed"],
                           "quote_source": ["yfinance", "yfinance"]})
        line = format_data_quality_summary(df)
        self.assertNotIn("CBOE", line)


if __name__ == "__main__":
    unittest.main()
