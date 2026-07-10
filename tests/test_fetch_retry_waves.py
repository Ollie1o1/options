"""Fetch-failure handling for interactive scans.

A single-ticker interactive scan that hits a Yahoo rate-limit used to sit through
TWO serial retry waves — a 20s cool-down then a 45s cool-down — on top of the
inner decorator's 4 attempts, i.e. 65s+ of "cooling down" only to still fail.
The retry waves exist to let a big parallel *storm* subside; a single ticker is
not a storm. Waves are now scaled to batch size.

Separately, when every expiration's chain fetch failed, the reason was swallowed
twice and the user saw only "No options data frames fetched" — no 429, no
timeout, nothing actionable. The reason now survives.
"""
from __future__ import annotations

import unittest
import uuid


class RetryWaveScalingTest(unittest.TestCase):
    def setUp(self):
        from src.options_screener import _retry_waves_for
        self.waves = _retry_waves_for

    def test_single_ticker_gets_one_short_wave(self):
        w = self.waves(1)
        self.assertEqual(len(w), 1)
        self.assertLessEqual(w[0], 10)

    def test_a_pair_is_also_treated_as_interactive(self):
        self.assertEqual(len(self.waves(2)), 1)

    def test_large_bulk_scan_keeps_the_long_waves(self):
        # The storm case the cool-downs were designed for is preserved.
        self.assertEqual(self.waves(100), [20, 45])

    def test_single_ticker_total_wait_is_a_fraction_of_the_bulk_wait(self):
        self.assertLess(sum(self.waves(1)), sum(self.waves(100)) / 3)

    def test_waves_are_non_negative_and_ordered_for_every_size(self):
        for n in (1, 2, 3, 10, 29, 30, 80, 200):
            w = self.waves(n)
            self.assertTrue(all(x >= 0 for x in w))
            self.assertEqual(w, sorted(w))


class ChainErrorSurfacingTest(unittest.TestCase):
    def test_process_option_chain_propagates_the_reason_not_an_empty_list(self):
        # A rate-limit on the chain endpoint must not be silently turned into
        # "this expiry has no options"; that reason is the whole diagnosis.
        from src import data_fetching as D

        class _Boom:
            def option_chain(self, exp):
                raise RuntimeError("Too Many Requests. Rate limited. Try after a while.")

        sym = "ZZ" + uuid.uuid4().hex[:6].upper()   # unique → no disk-cache hit
        with self.assertRaises(Exception) as ctx:
            D._process_option_chain(_Boom(), sym, "2030-01-18")
        self.assertIn("Too Many Requests", str(ctx.exception))

    def test_no_frames_error_carries_the_underlying_cause(self):
        # The message the retry layer and the user see should name the cause.
        from src import data_fetching as D
        msg = D._no_frames_message("NVDA", RuntimeError("Too Many Requests"))
        self.assertIn("NVDA", msg)
        self.assertIn("Too Many Requests", msg)

    def test_no_frames_message_without_a_cause_is_still_valid(self):
        from src import data_fetching as D
        msg = D._no_frames_message("NVDA", None)
        self.assertIn("NVDA", msg)


if __name__ == "__main__":
    unittest.main()
