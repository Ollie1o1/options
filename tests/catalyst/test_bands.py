"""Banding and budget allocation. Pure arithmetic — no rendering, no network."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import bands

TODAY = "2026-08-26"


class TestDaysUntil(unittest.TestCase):
    def test_day_precision(self):
        self.assertEqual(bands.days_until("2026-08-31", TODAY), 5)

    def test_month_precision_resolves_to_mid_month(self):
        # "2026-09" means sometime in September. The 15th is the central
        # estimate; first-of-month would pull it a band nearer than the
        # source supports, end-of-month a band further.
        self.assertEqual(bands.days_until("2026-09", TODAY), 20)

    def test_a_past_date_is_negative(self):
        self.assertEqual(bands.days_until("2026-08-20", TODAY), -6)


class TestBandFor(unittest.TestCase):
    def test_day_30_is_near(self):
        self.assertEqual(bands.band_for("2026-09-25", TODAY), bands.NEXT_30)

    def test_day_31_is_mid(self):
        self.assertEqual(bands.band_for("2026-09-26", TODAY), bands.D31_90)

    def test_day_90_is_mid(self):
        self.assertEqual(bands.band_for("2026-11-24", TODAY), bands.D31_90)

    def test_day_91_is_far(self):
        self.assertEqual(bands.band_for("2026-11-25", TODAY), bands.BEYOND_90)

    def test_an_elapsed_date_still_bands_as_near(self):
        # The sweep can return a date that slipped past today between fetch
        # and render. It must not crash or land in BEYOND_90.
        self.assertEqual(bands.band_for("2026-08-20", TODAY), bands.NEXT_30)


class TestAllocate(unittest.TestCase):
    def test_everything_fits_under_budget(self):
        counts = {bands.NEXT_30: 6, bands.D31_90: 10, bands.BEYOND_90: 8}
        out = bands.allocate(counts, budget=40)
        self.assertEqual(out, counts)

    def test_near_band_is_satisfied_first(self):
        counts = {bands.NEXT_30: 17, bands.D31_90: 21, bands.BEYOND_90: 59}
        out = bands.allocate(counts, budget=40)
        self.assertEqual(out[bands.NEXT_30], 17)

    def test_far_band_is_represented_rather_than_starved(self):
        # The defect this fixes: `collapsed[:40]` front-loaded by date and
        # returned zero names beyond 2026-10-31 for a 6-month window.
        counts = {bands.NEXT_30: 17, bands.D31_90: 21, bands.BEYOND_90: 59}
        out = bands.allocate(counts, budget=40)
        self.assertGreaterEqual(out[bands.BEYOND_90], 5)

    def test_never_exceeds_the_budget(self):
        counts = {bands.NEXT_30: 50, bands.D31_90: 50, bands.BEYOND_90: 50}
        out = bands.allocate(counts, budget=40)
        self.assertLessEqual(sum(out.values()), 40)

    def test_never_allocates_more_than_a_band_has(self):
        counts = {bands.NEXT_30: 2, bands.D31_90: 3, bands.BEYOND_90: 1}
        out = bands.allocate(counts, budget=40)
        for band, n in out.items():
            self.assertLessEqual(n, counts[band])

    def test_an_empty_band_gets_nothing(self):
        counts = {bands.NEXT_30: 0, bands.D31_90: 30, bands.BEYOND_90: 0}
        out = bands.allocate(counts, budget=10)
        self.assertEqual(out[bands.NEXT_30], 0)
        self.assertEqual(out[bands.BEYOND_90], 0)
        self.assertEqual(out[bands.D31_90], 10)

    def test_near_band_alone_can_consume_the_whole_budget(self):
        counts = {bands.NEXT_30: 50, bands.D31_90: 10, bands.BEYOND_90: 10}
        out = bands.allocate(counts, budget=10)
        self.assertEqual(out[bands.NEXT_30], 10)
        self.assertEqual(sum(out.values()), 10)

    def test_zero_budget_allocates_nothing(self):
        counts = {bands.NEXT_30: 5, bands.D31_90: 5, bands.BEYOND_90: 5}
        self.assertEqual(sum(bands.allocate(counts, budget=0).values()), 0)

    def test_spends_the_whole_budget_when_supply_allows(self):
        counts = {bands.NEXT_30: 5, bands.D31_90: 40, bands.BEYOND_90: 40}
        self.assertEqual(sum(bands.allocate(counts, budget=40).values()), 40)


if __name__ == "__main__":
    unittest.main()
