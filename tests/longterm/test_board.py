"""Tests for banner + board rendering (plain-mode, color pinned off)."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.formatting as fmt
from src.longterm import board as B
from src.longterm import plan as P
from src.longterm import zones as Z

fmt._COLOR_ENABLED = False  # pin: never env vars (supports_color memoizes)


def mu_plan():
    return P.Plan(5000.0, [P.PlanName("MU", [P.Tranche(750, 0.4), P.Tranche(650, 0.6)])])


def read(state, spot=780.0, level=750.0):
    return Z.ZoneRead("MU", state, spot, level, (spot - level) / level * 100,
                      1.0, -32.4, True)


class TestBanner(unittest.TestCase):
    def test_silent_when_nothing_triggered(self):
        self.assertEqual(B.banner([read(Z.WATCHING)], mu_plan(), 5000.0), "")
        self.assertEqual(B.banner([], mu_plan(), 5000.0), "")

    def test_near_line_contents(self):
        out = B.banner([read(Z.NEAR)], mu_plan(), 5000.0)
        self.assertIn("MU", out)
        self.assertIn("NEAR", out)
        self.assertIn("750", out)
        self.assertIn("$2,000", out)          # 5000 × 1.0 alloc × 0.4 weight
        self.assertIn("-32.4%", out)          # drawdown context

    def test_size_capped_at_remaining(self):
        out = B.banner([read(Z.IN_ZONE, spot=749.0)], mu_plan(), 500.0)
        self.assertIn("$500", out)
        self.assertNotIn("$2,000", out)

    def test_earnings_flag(self):
        out = B.banner([read(Z.NEAR)], mu_plan(), 5000.0, earnings={"MU": "07-28"})
        self.assertIn("earnings 07-28", out)


class TestBoard(unittest.TestCase):
    def test_board_shows_ladder_states_and_book(self):
        book = {"MU": {"shares": 2.0, "cost": 1500.0, "avg_price": 750.0}}
        out = B.render_board(mu_plan(), [read(Z.NEAR)], book, 3500.0)
        self.assertIn("MU", out)
        self.assertIn("750", out)
        self.assertIn("650", out)
        self.assertIn("NEAR", out)
        self.assertIn("2.0", out)            # held shares visible
        self.assertIn("3,500", out)          # remaining cash

    def test_board_empty_plan(self):
        out = B.render_board(P.Plan(), [], {}, 0.0)
        self.assertIn("empty", out.lower())  # points user at ADD

    def test_suggested_size(self):
        plan = mu_plan()
        n, t = plan.names[0], plan.names[0].tranches[0]
        self.assertEqual(B.suggested_size(plan, n, t, 10000.0), 2000.0)
        self.assertEqual(B.suggested_size(plan, n, t, 500.0), 500.0)


if __name__ == "__main__":
    unittest.main()
