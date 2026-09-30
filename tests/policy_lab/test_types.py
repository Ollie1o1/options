"""Types carry the price path and its replay result. They hold no logic beyond
invariants, so the tests here pin immutability and the interior-point rule that
every exit policy depends on.
"""
import unittest

from src.policy_lab.types import Outcome, PathPoint, PricePath


def _pt(date, mid, dte):
    return PathPoint(date=date, bid=mid - 0.05, ask=mid + 0.05, mid=mid,
                     spot=100.0, dte=dte)


def _path(points, entry="2026-08-18", exit_="2026-08-25"):
    return PricePath(
        position_id="p1", symbol="AMD", strategy="Bull Put",
        entry_date=entry, entry_price=1.00, capital_at_risk=400.0,
        is_credit=True, points=tuple(points),
        actual_exit_date=exit_, actual_pnl_frac=0.10, corpus="A",
    )


class TestTypes(unittest.TestCase):
    def test_pathpoint_is_frozen(self):
        p = _pt("2026-08-19", 0.90, 30)
        with self.assertRaises(Exception):
            p.mid = 1.5

    def test_interior_points_excludes_entry_and_exit_days(self):
        pts = [_pt("2026-08-18", 1.00, 31), _pt("2026-08-20", 0.80, 29),
               _pt("2026-08-25", 0.50, 24)]
        self.assertEqual([p.date for p in _path(pts).interior_points],
                         ["2026-08-20"])

    def test_interior_points_empty_when_only_endpoints(self):
        pts = [_pt("2026-08-18", 1.00, 31), _pt("2026-08-25", 0.50, 24)]
        self.assertEqual(_path(pts).interior_points, ())

    def test_zero_mid_is_a_value_not_a_missing_point(self):
        # `if not mid` shipped as a bug three times. A $0.00 mid is priceable.
        pts = [_pt("2026-08-18", 1.00, 31), PathPoint("2026-08-20", 0.0, 0.05,
                                                      0.0, 100.0, 29),
               _pt("2026-08-25", 0.50, 24)]
        self.assertEqual(len(_path(pts).interior_points), 1)

    def test_outcome_is_frozen(self):
        o = Outcome("p1", "2026-08-25", "take_profit", 0.5, 3)
        with self.assertRaises(Exception):
            o.pnl_frac_car = 1.0


if __name__ == "__main__":
    unittest.main()
