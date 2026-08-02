"""The frozen squeeze exit ladder: TP1 half, TP2 rest, hard time exit, no stop."""
import unittest

from src.squeeze.sleeve import ladder


class LadderTest(unittest.TestCase):
    # sigma_h = 0.40 -> TP1 at +20% (120.0), TP2 at +50% (150.0)
    SPOT = 100.0
    SIG = 0.40

    def test_neither_level_touched_exits_on_time_at_full_size(self):
        path = [100.0] * 42
        fills = ladder.simulate(path, self.SPOT, self.SIG)
        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0].reason, "time")
        self.assertEqual(fills[0].bar, 42)
        self.assertAlmostEqual(fills[0].fraction, 1.0)

    def test_tp1_only_leaves_the_remainder_to_the_time_exit(self):
        path = [100.0] * 10 + [125.0] + [110.0] * 31
        fills = ladder.simulate(path, self.SPOT, self.SIG)
        self.assertEqual([f.reason for f in fills], ["tp1", "time"])
        self.assertEqual(fills[0].bar, 11)
        self.assertAlmostEqual(fills[0].fraction, 0.5)
        self.assertAlmostEqual(fills[1].fraction, 0.5)
        self.assertEqual(fills[1].bar, 42)

    def test_both_levels_touched_closes_the_position_early(self):
        path = [100.0] * 5 + [125.0] + [160.0] + [90.0] * 35
        fills = ladder.simulate(path, self.SPOT, self.SIG)
        self.assertEqual([f.reason for f in fills], ["tp1", "tp2"])
        self.assertEqual(fills[0].bar, 6)
        self.assertEqual(fills[1].bar, 7)
        self.assertAlmostEqual(sum(f.fraction for f in fills), 1.0)

    def test_a_gap_through_both_levels_fills_both_on_the_same_bar(self):
        path = [100.0] * 3 + [200.0] + [100.0] * 38
        fills = ladder.simulate(path, self.SPOT, self.SIG)
        self.assertEqual([f.reason for f in fills], ["tp1", "tp2"])
        self.assertEqual(fills[0].bar, 4)
        self.assertEqual(fills[1].bar, 4)
        self.assertAlmostEqual(fills[0].price, 200.0)

    def test_no_stop_loss_a_collapse_still_rides_to_the_time_exit(self):
        path = [10.0] * 42
        fills = ladder.simulate(path, self.SPOT, self.SIG)
        self.assertEqual([f.reason for f in fills], ["time"])
        self.assertAlmostEqual(fills[0].fraction, 1.0)

    def test_a_short_path_exits_on_its_final_bar(self):
        fills = ladder.simulate([100.0] * 20, self.SPOT, self.SIG)
        self.assertEqual(fills[0].bar, 20)
        self.assertEqual(fills[0].reason, "time")
