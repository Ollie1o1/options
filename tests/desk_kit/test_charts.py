import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src.desk_kit import charts as C
from src.desk_kit import theme


class TestPayoff(unittest.TestCase):
    def _svg(self, **over):
        kw = dict(spot=100.0, strike=105.0, opt_type="call", premium=2.5,
                  breakeven=107.5)
        kw.update(over)
        return C.payoff_chart(**kw)

    def test_renders_svg_with_markers(self):
        svg = self._svg()
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("spot 100.00", svg)
        self.assertIn("BE 107.50", svg)
        self.assertIn("var(--good)", svg)
        self.assertIn("var(--bad)", svg)

    def test_expiry_math_call_at_strike_loses_premium(self):
        # at the strike a long call expires worthless: P&L == -premium * 100
        ladder = [90.0, 105.0, 120.0]
        pnl = [(max(0.0, p - 105.0) - 2.5) * 100 for p in ladder]
        self.assertEqual(pnl[1], -250.0)
        svg = self._svg(today_prices=ladder)
        self.assertIn("-250", svg)

    def test_put_payoff_direction(self):
        ladder = [80.0, 100.0, 120.0]
        svg = C.payoff_chart(100.0, 100.0, "put", 3.0, breakeven=97.0,
                             today_prices=ladder)
        # deep ITM put at 80: (20 - 3) * 100 = +1700
        self.assertIn("+1,700", svg)

    def test_today_curve_is_dashed_projection(self):
        ladder = [90.0, 100.0, 110.0]
        svg = self._svg(today_prices=ladder, today_pnl=[-100.0, 20.0, 200.0])
        self.assertIn("today", svg)
        self.assertIn("stroke-dasharray", svg)

    def test_mismatched_today_curve_is_dropped(self):
        svg = self._svg(today_prices=[90.0, 100.0, 110.0],
                        today_pnl=[1.0, 2.0])
        self.assertNotIn(">today<", svg)

    def test_garbage_inputs_return_empty(self):
        self.assertEqual(C.payoff_chart(None, 100, "call", 2.0), "")
        self.assertEqual(C.payoff_chart(100, 100, "call", -1.0), "")
        self.assertEqual(C.payoff_chart("x", 100, "call", 2.0), "")

    def test_deterministic(self):
        self.assertEqual(self._svg(), self._svg())


class TestWaterfall(unittest.TestCase):
    def test_net_column_and_signed_labels(self):
        svg = C.waterfall([["Gross edge", 60.0], ["Entry spread", -90.0],
                           ["Exit spread", -90.0]])
        self.assertTrue(svg.startswith("<svg"))
        self.assertIn("Net", svg)
        self.assertIn("-120", svg)   # net total
        self.assertIn("+60", svg)
        self.assertIn("var(--bad)", svg)
        self.assertIn("var(--good)", svg)

    def test_empty_returns_empty(self):
        self.assertEqual(C.waterfall([]), "")
        self.assertEqual(C.waterfall(None), "")

    def test_bar_cap_respected(self):
        svg = C.waterfall([["A", 5.0], ["B", -2.0]])
        import re
        widths = [float(m) for m in
                  re.findall(r'<rect[^>]*width="([\d.]+)"', svg)]
        self.assertTrue(all(bw <= 24.0 for bw in widths))


class TestHeatGrid(unittest.TestCase):
    def test_grid_carries_both_inks_and_signed_values(self):
        stress = {"moves": [-0.05, 0.0, 0.05],
                  "rows": [{"iv": 0.10, "pnls": [-288.0, -12.0, 180.0]}]}
        html = C.heat_grid(stress, theme.heat_inks)
        self.assertIn("--hl:#", html)
        self.assertIn("--hd:#", html)
        self.assertIn("+180", html)
        self.assertIn("-288", html)
        self.assertIn("IV +10pp", html)

    def test_empty_returns_empty(self):
        self.assertEqual(C.heat_grid({}, theme.heat_inks), "")


class TestMeterSparkline(unittest.TestCase):
    def test_meter_states(self):
        short = C.meter(27, 50)
        done = C.meter(50, 50)
        self.assertIn("54%", short)
        self.assertIn("var(--accent)", short)
        self.assertIn("var(--good)", done)
        self.assertEqual(C.meter(1, 0), "")
        self.assertEqual(C.meter("x", 50), "")

    def test_sparkline(self):
        self.assertTrue(C.sparkline([1, 2, 3]).startswith("<svg"))
        self.assertEqual(C.sparkline([1]), "")


class TestPortedPrimitives(unittest.TestCase):
    """The primitives ported from research/charts keep their contracts."""

    def test_area_chart_flat_series_no_zero_division(self):
        self.assertTrue(C.area_chart([2, 2, 2], ["a", "b", "c"], "u"))

    def test_hbar_diverging_signed_labels(self):
        svg = C.hbar_diverging([("SPY", 1.5), ("QQQ", -2.0)], unit="vp")
        self.assertIn("+1.5vp", svg)
        self.assertIn("-2.0vp", svg)

    def test_term_chart_needs_two_points(self):
        self.assertEqual(C.term_chart([[9, 0.3]]), "")
        self.assertTrue(C.term_chart([[9, 0.3], [44, 0.37]]))

    def test_cone_chart_marks_current(self):
        rows = [{"window": 10, "p25": .2, "median": .3, "p75": .4, "current": .35},
                {"window": 30, "p25": .22, "median": .31, "p75": .42}]
        self.assertIn("var(--accent)", C.cone_chart(rows))

    def test_all_ink_is_css_vars(self):
        svg = C.area_chart([1, 2, 3], [], "u") + C.hbar_diverging([("A", 1)])
        self.assertNotIn("#0", svg)
        self.assertNotIn("#f", svg)


if __name__ == "__main__":
    unittest.main()
