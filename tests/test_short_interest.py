"""Tests for src.short_interest — enriched SI detail from ticker info."""
import unittest

from src import short_interest as si


class ShortInterestDetailTest(unittest.TestCase):
    def test_extracts_pct_float_and_days_to_cover(self):
        info = {"shortPercentOfFloat": 0.18, "shortRatio": 3.4}
        d = si.short_interest_detail(info)
        self.assertAlmostEqual(d.pct_float, 0.18)
        self.assertAlmostEqual(d.days_to_cover, 3.4)

    def test_percent_given_as_whole_number_normalized(self):
        info = {"shortPercentOfFloat": 22.0}  # some feeds give 22 not 0.22
        d = si.short_interest_detail(info)
        self.assertAlmostEqual(d.pct_float, 0.22)

    def test_trend_rising_when_shares_short_grew(self):
        info = {"shortPercentOfFloat": 0.20, "sharesShort": 11_000_000,
                "sharesShortPriorMonth": 10_000_000}
        d = si.short_interest_detail(info)
        self.assertEqual(d.trend, "rising")

    def test_trend_falling_when_shares_short_shrank(self):
        info = {"sharesShort": 8_000_000, "sharesShortPriorMonth": 10_000_000}
        d = si.short_interest_detail(info)
        self.assertEqual(d.trend, "falling")

    def test_trend_flat_within_tolerance(self):
        info = {"sharesShort": 10_100_000, "sharesShortPriorMonth": 10_000_000}
        d = si.short_interest_detail(info)
        self.assertEqual(d.trend, "flat")

    def test_trend_none_without_prior(self):
        info = {"shortPercentOfFloat": 0.05}
        d = si.short_interest_detail(info)
        self.assertIsNone(d.trend)

    def test_garbage_values_do_not_crash(self):
        info = {"shortPercentOfFloat": "n/a", "shortRatio": None}
        d = si.short_interest_detail(info)
        self.assertIsNone(d.pct_float)
        self.assertIsNone(d.days_to_cover)

    def test_empty_info(self):
        d = si.short_interest_detail({})
        self.assertIsNone(d.pct_float)
        self.assertIsNone(d.days_to_cover)
        self.assertIsNone(d.trend)


class FormatShortInterestTest(unittest.TestCase):
    def test_factual_line_has_pct_and_dtc(self):
        d = si.ShortInterest(pct_float=0.184, days_to_cover=3.4,
                             pct_float_prior=0.16, shares_short=11_000_000,
                             trend="rising")
        line = si.format_short_interest(d)
        self.assertIn("18", line)        # 18.4% of float
        self.assertIn("3.4", line)       # days to cover
        self.assertIn("rising", line.lower())

    def test_none_detail_returns_empty(self):
        d = si.ShortInterest(None, None, None, None, None)
        self.assertEqual(si.format_short_interest(d), "")

    def test_pct_only_still_renders(self):
        d = si.ShortInterest(pct_float=0.05, days_to_cover=None,
                             pct_float_prior=None, shares_short=None, trend=None)
        line = si.format_short_interest(d)
        self.assertIn("5", line)


if __name__ == "__main__":
    unittest.main()
