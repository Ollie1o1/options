"""Quote validation: inverted (bid > ask) quotes must never be trusted.

2026-07-13 audit finding: `enrich_and_score` validated only bid > 0 and
ask > 0. A crossed quote (bid > ask) produced a NEGATIVE spread_pct that
sailed through the `spread_pct <= max` filter looking like the tightest
spread in the chain, and its corrupted mid fed premium, EV, and the IV
cross-validation solver. The dolt path already guarded this in SQL
(`ask >= bid`); the live path did not.

The mid/premium logic is extracted into `_compute_quote_columns` so this
is testable without running the whole scoring pipeline.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_quote_validation -v
"""
from __future__ import annotations

import math
import unittest

import pandas as pd

from src.options_screener import _compute_quote_columns


def _chain(rows):
    return pd.DataFrame(rows)


def _row(bid=1.0, ask=1.2, lastPrice=1.1):
    return {"bid": bid, "ask": ask, "lastPrice": lastPrice}


class TestComputeQuoteColumns(unittest.TestCase):
    def test_normal_quote_gets_mid_and_premium(self):
        df = _compute_quote_columns(_chain([_row(bid=1.0, ask=1.2)]))
        self.assertAlmostEqual(df["mid"].iloc[0], 1.1)
        self.assertAlmostEqual(df["premium"].iloc[0], 1.1)

    def test_crossed_quote_is_not_trusted(self):
        # bid 2.00 / ask 1.00 is a broken quote: mid must be NaN and premium
        # must fall back to lastPrice, NOT (bid+ask)/2 = 1.50.
        df = _compute_quote_columns(_chain([_row(bid=2.0, ask=1.0, lastPrice=1.4)]))
        self.assertTrue(math.isnan(df["mid"].iloc[0]))
        self.assertAlmostEqual(df["premium"].iloc[0], 1.4)

    def test_crossed_quote_never_produces_negative_spread(self):
        # Downstream: spread_pct = (ask - bid) / mid. With mid NaN the spread
        # is NaN -> set to inf -> dropped by the max-spread filter. It must
        # never come out negative (negative would rank as the TIGHTEST quote).
        df = _compute_quote_columns(_chain([_row(bid=2.0, ask=1.0, lastPrice=1.4)]))
        spread = (df["ask"] - df["bid"]) / df["mid"]
        self.assertFalse((spread < 0).any())

    def test_touching_quote_bid_equals_ask_is_valid(self):
        # A locked market (bid == ask) is a real, tradeable quote.
        df = _compute_quote_columns(_chain([_row(bid=1.0, ask=1.0)]))
        self.assertAlmostEqual(df["mid"].iloc[0], 1.0)

    def test_zero_bid_falls_back_to_last_price(self):
        df = _compute_quote_columns(_chain([_row(bid=0.0, ask=0.5, lastPrice=0.3)]))
        self.assertTrue(math.isnan(df["mid"].iloc[0]))
        self.assertAlmostEqual(df["premium"].iloc[0], 0.3)
        self.assertTrue(math.isnan(df["bid"].iloc[0]))

    def test_zero_ask_falls_back_to_last_price(self):
        df = _compute_quote_columns(_chain([_row(bid=0.2, ask=0.0, lastPrice=0.3)]))
        self.assertTrue(math.isnan(df["mid"].iloc[0]))
        self.assertAlmostEqual(df["premium"].iloc[0], 0.3)
        self.assertTrue(math.isnan(df["ask"].iloc[0]))

    def test_crossed_quote_does_not_poison_valid_rows(self):
        df = _compute_quote_columns(_chain([
            _row(bid=1.0, ask=1.2),
            _row(bid=2.0, ask=1.0, lastPrice=1.4),
        ]))
        self.assertAlmostEqual(df["mid"].iloc[0], 1.1)
        self.assertTrue(math.isnan(df["mid"].iloc[1]))


if __name__ == "__main__":
    unittest.main()
