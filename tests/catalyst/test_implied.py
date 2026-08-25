"""Straddle-implied move. Network mocked; no chain is ever fetched in tests."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import implied

EXPIRIES = ["2026-09-18", "2026-10-16", "2026-11-20", "2027-01-15"]


class Row(dict):
    """Minimal stand-in for a chain row; the code reads it like a mapping."""


class TestPickExpiry(unittest.TestCase):
    def test_picks_the_first_expiry_at_or_after_the_event(self):
        self.assertEqual(implied.pick_expiry(EXPIRIES, "2026-10-31"), "2026-11-20")

    def test_an_expiry_exactly_on_the_event_date_qualifies(self):
        self.assertEqual(implied.pick_expiry(EXPIRIES, "2026-11-20"), "2026-11-20")

    def test_month_precision_event_anchors_to_the_first(self):
        self.assertEqual(implied.pick_expiry(EXPIRIES, "2026-11"), "2026-11-20")

    def test_none_when_no_expiry_reaches_the_event(self):
        self.assertIsNone(implied.pick_expiry(EXPIRIES, "2027-06-30"))

    def test_none_with_no_expiries(self):
        self.assertIsNone(implied.pick_expiry([], "2026-10-31"))


class TestStraddleMove(unittest.TestCase):
    def test_uses_the_strike_nearest_spot(self):
        calls = [Row(strike=20.0, lastPrice=3.0, bid=2.9, ask=3.1),
                 Row(strike=22.0, lastPrice=2.0, bid=1.9, ask=2.1)]
        puts = [Row(strike=20.0, lastPrice=1.0, bid=0.9, ask=1.1),
                Row(strike=22.0, lastPrice=2.2, bid=2.1, ask=2.3)]
        # spot 22.10 -> nearest strike is 22 -> straddle 2.0 + 2.2 = 4.2
        move = implied.straddle_move(calls, puts, 22.10)
        self.assertAlmostEqual(move, 4.2 / 22.10, places=4)

    def test_prefers_mid_over_last_when_both_sides_quoted(self):
        calls = [Row(strike=22.0, lastPrice=99.0, bid=1.9, ask=2.1)]
        puts = [Row(strike=22.0, lastPrice=99.0, bid=2.1, ask=2.3)]
        move = implied.straddle_move(calls, puts, 22.0)
        self.assertAlmostEqual(move, (2.0 + 2.2) / 22.0, places=4)

    def test_none_when_a_side_is_missing(self):
        calls = [Row(strike=22.0, lastPrice=2.0, bid=1.9, ask=2.1)]
        self.assertIsNone(implied.straddle_move(calls, [], 22.0))

    def test_none_when_spot_is_zero(self):
        calls = [Row(strike=22.0, lastPrice=2.0, bid=1.9, ask=2.1)]
        puts = [Row(strike=22.0, lastPrice=2.2, bid=2.1, ask=2.3)]
        self.assertIsNone(implied.straddle_move(calls, puts, 0.0))


class TestImpliedMove(unittest.TestCase):
    def test_no_options_yields_empty_not_raise(self):
        with mock.patch.object(implied, "_expiries", return_value=[]):
            m = implied.implied_move("ANNX", "2026-10-31")
        self.assertIsNone(m.move_pct)
        self.assertIsNone(m.expiry)

    def test_network_failure_yields_empty_not_raise(self):
        with mock.patch.object(implied, "_expiries", side_effect=OSError("boom")):
            self.assertIsNone(implied.implied_move("ANNX", "2026-10-31").move_pct)


if __name__ == "__main__":
    unittest.main()
