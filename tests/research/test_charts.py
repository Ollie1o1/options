import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest

from src.research import charts as C


class TestAreaChart(unittest.TestCase):
    def test_empty_and_single_point_return_empty(self):
        self.assertEqual(C.area_chart([], [], "a"), "")
        self.assertEqual(C.area_chart([1.0], ["x"], "a"), "")
        self.assertEqual(C.area_chart([None, float("nan")], ["x", "y"], "a"), "")

    def test_basic_geometry(self):
        svg = C.area_chart([1.0, 2.0, 3.0], ["Jan 1", "Jan 2", "Jan 3"], "spy")
        self.assertIn("<svg", svg)
        self.assertIn("polyline", svg)
        self.assertIn("polygon", svg)          # gradient area fill
        self.assertIn("gspy", svg)             # per-chart gradient id
        self.assertIn('class="xh"', svg)       # crosshair opt-in
        self.assertIn("data-labels", svg)
        self.assertIn("data-ys", svg)
        self.assertIn("ch-line", svg)
        self.assertIn("Jan 1", svg)            # x-axis end labels
        self.assertNotIn("None", svg)
        self.assertNotIn("nan", svg)

    def test_flat_series_does_not_divide_by_zero(self):
        svg = C.area_chart([5.0, 5.0, 5.0], ["a", "b", "c"], "f")
        self.assertIn("<svg", svg)


class TestPriceChart(unittest.TestCase):
    def test_renders_mas_and_levels(self):
        closes = [100.0 + i for i in range(30)]
        ma50 = [99.0 + i for i in range(30)]
        svg = C.price_chart(closes, ma50, [], {"level": 105.0, "label": "50d MA"},
                            {"level": 140.0, "label": "recent high"},
                            ["d%d" % i for i in range(30)], "px")
        self.assertIn("<svg", svg)
        self.assertIn("50d MA", svg)
        self.assertIn("recent high", svg)
        self.assertIn("stroke-dasharray", svg)  # MA line style
        self.assertIn('class="xh"', svg)

    def test_level_outside_close_range_stays_on_canvas(self):
        # y-scale must span closes AND levels (tearsheet lesson).
        closes = [100.0, 101.0, 102.0]
        svg = C.price_chart(closes, [], [], None, {"level": 200.0, "label": "R"},
                            ["a", "b", "c"], "p2")
        self.assertIn("200.00", svg)

    def test_no_history_returns_empty(self):
        self.assertEqual(C.price_chart([], [], [], None, None, [], "p3"), "")


class TestRsiStrip(unittest.TestCase):
    def test_bands_and_line(self):
        svg = C.rsi_strip([45.0, 55.0, 72.0, 28.0])
        self.assertIn("<svg", svg)
        self.assertIn("70", svg)   # overbought guide
        self.assertIn("30", svg)   # oversold guide
        self.assertEqual(C.rsi_strip([]), "")


class TestHbarDiverging(unittest.TestCase):
    def test_signed_bars_and_labels(self):
        svg = C.hbar_diverging([("SPY", 1.5), ("QQQ", -2.0)], unit="%")
        self.assertIn("SPY", svg)
        self.assertIn("+1.5%", svg)   # signed label = secondary encoding
        self.assertIn("-2.0%", svg)
        self.assertIn("var(--good)", svg)
        self.assertIn("var(--bad)", svg)
        self.assertEqual(C.hbar_diverging([]), "")

    def test_none_values_skipped(self):
        svg = C.hbar_diverging([("A", None), ("B", 1.0)])
        self.assertNotIn(">A<", svg)
        self.assertIn("B", svg)


class TestConeChart(unittest.TestCase):
    def test_envelope_median_current(self):
        rows = [{"window": 10, "p25": 0.20, "median": 0.25, "p75": 0.32, "current": 0.28},
                {"window": 30, "p25": 0.22, "median": 0.27, "p75": 0.35, "current": 0.30}]
        svg = C.cone_chart(rows)
        self.assertIn("polygon", svg)   # p25-p75 envelope
        self.assertIn("10d", svg)
        self.assertIn("30d", svg)
        self.assertEqual(C.cone_chart([]), "")


class TestTermChart(unittest.TestCase):
    def test_curve_with_point_labels(self):
        svg = C.term_chart([[7, 0.55], [30, 0.48], [60, 0.45]])
        self.assertIn("7d", svg)
        self.assertIn("55%", svg)
        self.assertEqual(C.term_chart([]), "")
        self.assertEqual(C.term_chart([[7, 0.5]]), "")  # needs >= 2 points


if __name__ == "__main__":
    unittest.main()
