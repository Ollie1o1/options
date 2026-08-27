"""Forward returns. Prices are always supplied; no network in tests."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst.backtest import outcomes

# Dates land exactly on the horizon boundaries (+91, +182, +365 days), so the
# bounded snap is not being exercised accidentally.
PRICES = {"2025-01-01": 100.0, "2025-04-02": 110.0,
          "2025-07-02": 130.0, "2026-01-01": 90.0}
BENCH = {"2025-01-01": 50.0, "2025-04-02": 52.5,
         "2025-07-02": 55.0, "2026-01-01": 60.0}


class TestElapsed(unittest.TestCase):
    def test_a_fully_elapsed_window(self):
        self.assertTrue(outcomes.elapsed("2025-01-01", 6, "2026-08-25"))

    def test_an_unelapsed_window(self):
        self.assertFalse(outcomes.elapsed("2026-07-01", 6, "2026-08-25"))

    def test_twelve_months_from_the_last_vintage_is_not_elapsed(self):
        # Measured: 12mo is 11/12 vintages, losing 2025-10-01.
        self.assertFalse(outcomes.elapsed("2025-10-01", 12, "2026-08-25"))

    def test_six_months_from_the_last_vintage_is_elapsed(self):
        self.assertTrue(outcomes.elapsed("2025-10-01", 6, "2026-08-25"))


class TestForwardReturn(unittest.TestCase):
    def test_simple_gain(self):
        r = outcomes.forward_return(PRICES, "2025-01-01", 90)
        self.assertAlmostEqual(r, 0.10, places=4)

    def test_loss(self):
        r = outcomes.forward_return(PRICES, "2025-01-01", 365)
        self.assertAlmostEqual(r, -0.10, places=4)

    def test_missing_start_price_is_none(self):
        self.assertIsNone(outcomes.forward_return(PRICES, "2024-01-01", 90))

    def test_missing_end_price_is_none_not_zero(self):
        self.assertIsNone(outcomes.forward_return(PRICES, "2025-07-02", 3650))

    def test_zero_start_price_is_none_not_infinity(self):
        self.assertIsNone(outcomes.forward_return(
            {"2025-01-01": 0.0, "2025-04-02": 5.0}, "2025-01-01", 91))

    def test_a_weekend_gap_still_resolves(self):
        # 2025-01-04 is a Saturday; the next close is Monday the 6th.
        prices = {"2025-01-01": 100.0, "2025-01-06": 110.0}
        self.assertAlmostEqual(
            outcomes.forward_return(prices, "2025-01-01", 3), 0.10, places=4)

    def test_a_sparse_series_does_not_stretch_the_window(self):
        # REGRESSION: an unbounded snap turned a 91-day horizon into a
        # six-month return and reported it as three-month.
        sparse = {"2025-01-01": 100.0, "2025-07-02": 130.0}
        self.assertIsNone(outcomes.forward_return(sparse, "2025-01-01", 91))

    def test_a_vintage_before_the_series_begins_is_none_not_zero(self):
        # REGRESSION: both ends snapped onto the same first price and returned
        # a confident 0.0%.
        self.assertIsNone(outcomes.forward_return(PRICES, "2020-01-01", 91))

    def test_empty_price_series_is_none(self):
        self.assertIsNone(outcomes.forward_return({}, "2025-01-01", 91))


class TestOutcomesFor(unittest.TestCase):
    def test_relative_return_subtracts_the_benchmark(self):
        out = outcomes.outcomes_for("ANNX", "2025-01-01", "2026-08-25",
                                    PRICES, BENCH)
        three = [o for o in out if o.months == 3][0]
        self.assertAlmostEqual(three.absolute, 0.10, places=4)
        self.assertAlmostEqual(three.relative, 0.10 - 0.05, places=4)

    def test_unelapsed_horizons_are_absent_not_zero(self):
        out = outcomes.outcomes_for("ANNX", "2026-07-01", "2026-08-25",
                                    PRICES, BENCH)
        self.assertEqual([o.months for o in out], [])

    def test_missing_benchmark_leaves_relative_none_but_keeps_absolute(self):
        out = outcomes.outcomes_for("ANNX", "2025-01-01", "2026-08-25",
                                    PRICES, {})
        three = [o for o in out if o.months == 3][0]
        self.assertIsNotNone(three.absolute)
        self.assertIsNone(three.relative)


if __name__ == "__main__":
    unittest.main()
