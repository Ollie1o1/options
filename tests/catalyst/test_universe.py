"""Market-cap band. Network is always mocked."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import universe


class TestInBand(unittest.TestCase):
    def test_real_measured_values_are_in_band(self):
        # Measured 2026-08-25.
        self.assertTrue(universe.in_band(976_332_558.0))    # ANNX
        self.assertTrue(universe.in_band(1_998_463_245.0))  # SRPT

    def test_mega_cap_is_out(self):
        self.assertFalse(universe.in_band(150_000_000_000.0))  # PFE

    def test_nano_cap_is_out(self):
        self.assertFalse(universe.in_band(12_000_000.0))

    def test_boundaries_are_inclusive(self):
        self.assertTrue(universe.in_band(universe.MCAP_LO))
        self.assertTrue(universe.in_band(universe.MCAP_HI))

    def test_unknown_mcap_is_excluded_not_defaulted_in(self):
        # A missing cap must not fall through into the band.
        self.assertFalse(universe.in_band(None))


class TestMarketCaps(unittest.TestCase):
    def test_maps_each_ticker(self):
        fake = mock.Mock()
        fake.fast_info = {"marketCap": 976_332_558.0}
        with mock.patch.object(universe, "_ticker", return_value=fake):
            self.assertEqual(universe.market_caps(["ANNX"]),
                             {"ANNX": 976_332_558.0})

    def test_failure_for_one_ticker_yields_none_not_crash(self):
        with mock.patch.object(universe, "_ticker", side_effect=OSError("boom")):
            self.assertEqual(universe.market_caps(["ANNX"]), {"ANNX": None})

    def test_delisted_ticker_with_no_mcap_is_none(self):
        fake = mock.Mock()
        fake.fast_info = {}
        with mock.patch.object(universe, "_ticker", return_value=fake):
            self.assertEqual(universe.market_caps(["APLS"]), {"APLS": None})


if __name__ == "__main__":
    unittest.main()
