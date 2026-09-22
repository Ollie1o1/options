"""Replay is pure: given a path, a policy and a cost model it returns what that
policy would have earned. The no-look-ahead test is the one that matters — a
policy that can see its own future will find edge on any data at all.
"""
import math
import unittest

from src.policy_lab.costs import CostModel
from src.policy_lab.policies import ExitPolicy
from src.policy_lab.replay import replay
from src.policy_lab.types import PathPoint, PricePath


def _pt(date, mid, dte, spread=0.10):
    return PathPoint(date=date, bid=mid - spread / 2, ask=mid + spread / 2,
                     mid=mid, spot=100.0, dte=dte)


def _credit_path(points, entry_price=1.00, car=400.0):
    """A short spread opened for 1.00 credit with 400 at risk."""
    return PricePath(
        position_id="p1", symbol="AMD", strategy="Bull Put",
        entry_date="2026-08-18", entry_price=entry_price,
        capital_at_risk=car, is_credit=True, points=tuple(points),
        actual_exit_date=points[-1].date, actual_pnl_frac=0.0, corpus="A",
    )


class TestReplay(unittest.TestCase):
    def test_take_profit_fires_when_credit_decays_far_enough(self):
        # Opened at 1.00; buying back at 0.50 captures 50% of the credit.
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 0.45, 29),
            _pt("2026-08-25", 0.20, 24),
        ])
        pol = ExitPolicy("tp50", 0.50, None, None, None)
        out = replay(path, pol, CostModel.mid())
        self.assertEqual(out.exit_reason, "take_profit")
        self.assertEqual(out.exit_date, "2026-08-20")
        self.assertAlmostEqual(out.pnl_frac_car, (1.00 - 0.45) * 100 / 400.0)

    def test_stop_fires_when_the_credit_doubles_against_you(self):
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 2.10, 29),
            _pt("2026-08-25", 3.00, 24),
        ])
        pol = ExitPolicy("sl2", None, 2.0, None, None)
        out = replay(path, pol, CostModel.mid())
        self.assertEqual(out.exit_reason, "stop_loss")
        self.assertEqual(out.exit_date, "2026-08-20")
        self.assertAlmostEqual(out.pnl_frac_car, (1.00 - 2.10) * 100 / 400.0)

    def test_time_exit_fires_at_the_dte_threshold(self):
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 0.95, 21),
            _pt("2026-08-25", 0.90, 14),
        ])
        pol = ExitPolicy("dte21", None, None, 21, None)
        out = replay(path, pol, CostModel.mid())
        self.assertEqual(out.exit_reason, "time_exit")
        self.assertEqual(out.exit_date, "2026-08-20")

    def test_max_hold_fires_on_calendar_days(self):
        # 2026-08-25 is exactly 7 days after entry, so the threshold fires
        # on the boundary rather than strictly past it.
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-25", 0.95, 27),
            _pt("2026-08-30", 0.90, 19),
        ])
        pol = ExitPolicy("hold7", None, None, None, 7)
        out = replay(path, pol, CostModel.mid())
        self.assertEqual(out.exit_reason, "max_hold")
        self.assertEqual(out.exit_date, "2026-08-25")

    def test_max_hold_fires_at_the_first_mark_past_the_threshold(self):
        """With sparse marks, a hold limit cannot fire on the day it expires.

        Entry 08-18 with a 7-day limit: the next quote after 08-22 (4 days) is
        08-30 (12 days), so the rule acts at 08-30. Exiting on 08-22 would mean
        knowing, while standing there, that no quote arrives for another 8 days.
        The real corpus averages 5.75 marks per position, so this is the normal
        case, not an edge case — a policy's effective hold is always at least
        its nominal one.
        """
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-22", 0.95, 27),
            _pt("2026-08-30", 0.90, 19),
        ])
        pol = ExitPolicy("hold7", None, None, None, 7)
        out = replay(path, pol, CostModel.mid())
        self.assertEqual(out.exit_reason, "max_hold")
        self.assertEqual(out.exit_date, "2026-08-30")

    def test_unarmed_policy_rides_to_the_last_point(self):
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 0.45, 29),
            _pt("2026-08-25", 0.20, 24),
        ])
        pol = ExitPolicy("null", None, None, None, None)
        out = replay(path, pol, CostModel.mid())
        self.assertEqual(out.exit_reason, "hold_to_end")
        self.assertEqual(out.exit_date, "2026-08-25")

    def test_earliest_trigger_wins_when_two_fire_the_same_day(self):
        # Stop and take-profit cannot both be right; the stop is checked first
        # because it is the risk rule.
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 2.50, 29),
        ])
        pol = ExitPolicy("both", 0.50, 2.0, None, None)
        out = replay(path, pol, CostModel.mid())
        self.assertEqual(out.exit_reason, "stop_loss")

    def test_no_look_ahead(self):
        """A policy must decide at step i using path[:i] only.

        Replacing every point after the decision day with NaN must not change
        the decision. If it does, something is reading its own future.
        """
        good = [_pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 0.45, 29),
                _pt("2026-08-25", 0.20, 24)]
        poisoned = good[:2] + [PathPoint("2026-08-25", math.nan, math.nan,
                                         math.nan, math.nan, 24)]
        pol = ExitPolicy("tp50", 0.50, None, None, None)
        a = replay(_credit_path(good), pol, CostModel.mid())
        b = replay(_credit_path(poisoned), pol, CostModel.mid())
        self.assertEqual(a.exit_date, b.exit_date)
        self.assertEqual(a.exit_reason, b.exit_reason)
        self.assertAlmostEqual(a.pnl_frac_car, b.pnl_frac_car)

    def test_cross_costs_strictly_less_than_mid_for_a_credit_close(self):
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 0.45, 29, spread=0.20),
        ])
        pol = ExitPolicy("null", None, None, None, None)
        at_mid = replay(path, pol, CostModel.mid()).pnl_frac_car
        at_cross = replay(path, pol, CostModel.cross()).pnl_frac_car
        self.assertLess(at_cross, at_mid)

    def test_decision_points_counts_interior_days_only(self):
        path = _credit_path([
            _pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 0.90, 29),
            _pt("2026-08-22", 0.80, 27), _pt("2026-08-25", 0.20, 24),
        ])
        pol = ExitPolicy("null", None, None, None, None)
        self.assertEqual(replay(path, pol, CostModel.mid()).decision_points, 2)

    def test_single_point_path_is_refused(self):
        path = _credit_path([_pt("2026-08-18", 1.00, 31)])
        with self.assertRaises(ValueError):
            replay(path, ExitPolicy("n", None, None, None, None),
                   CostModel.mid())

    def test_zero_entry_price_is_refused_not_treated_as_missing(self):
        path = _credit_path([_pt("2026-08-18", 0.0, 31), _pt("2026-08-20", 0.0, 29)],
                            entry_price=0.0)
        with self.assertRaises(ValueError):
            replay(path, ExitPolicy("tp", 0.50, None, None, None),
                   CostModel.mid())


if __name__ == "__main__":
    unittest.main()
