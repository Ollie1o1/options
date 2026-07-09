"""Desk visual layer — primitives and display-only panel smoke tests."""
import io
import os
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force color ON for deterministic prefix tests. Must precede the src imports:
# formatting.supports_color() memoizes its answer on first call.
os.environ["FORCE_COLOR"] = "1"
os.environ["COLORTERM"] = "truecolor"

from src import ui  # noqa: E402
from src import formatting as fmt  # noqa: E402


class ColorOffMixin:
    """Turn color off for one test, restoring the memoized flag afterwards.

    `fmt.supports_color()` caches into `fmt._COLOR_ENABLED`, so flipping the
    env var alone is a no-op once anything has rendered.
    """

    def force_color_off(self):
        prev_flag = fmt._COLOR_ENABLED
        prev_env = os.environ.get("NO_COLOR")
        fmt._COLOR_ENABLED = None
        os.environ["NO_COLOR"] = "1"

        def _restore():
            fmt._COLOR_ENABLED = prev_flag
            if prev_env is None:
                os.environ.pop("NO_COLOR", None)
            else:
                os.environ["NO_COLOR"] = prev_env

        self.addCleanup(_restore)


class TestHeatCell(ColorOffMixin, unittest.TestCase):
    def test_positive_value_is_green_ish(self):
        out = ui.heat_cell("+9k", 9000, 9000, glyph=False)
        self.assertIn("+9k", out)
        # full-positive maps to the 'good' anchor (94,201,141)
        self.assertIn("38;2;94;201;141", out)

    def test_negative_value_is_red_ish(self):
        out = ui.heat_cell("-9k", -9000, 9000, glyph=False)
        self.assertIn("38;2;224;108;117", out)

    def test_zero_is_neutral(self):
        out = ui.heat_cell("0", 0, 9000, glyph=False)
        self.assertIn("38;2;98;104;112", out)

    def test_glyph_density_tracks_magnitude(self):
        full = ui.heat_cell("x", 9000, 9000, glyph=True)
        none = ui.heat_cell("x", 0, 9000, glyph=True)
        self.assertIn("█", full)
        self.assertIn(" x", none)
        self.assertNotIn("█", none)

    def test_zero_span_does_not_crash(self):
        self.assertIn("x", ui.heat_cell("x", 5, 0, glyph=False))

    def test_plain_when_color_off(self):
        self.force_color_off()
        out = ui.heat_cell("+9k", 9000, 9000, glyph=False)
        self.assertEqual(out, "+9k")


class TestStressHeatmap(unittest.TestCase):
    def test_worst_cell_more_saturated_than_mild(self):
        # The larger loss must carry the denser shade glyph on a shared span.
        span = 85000
        worst = ui.heat_cell(" -85,909", -85909, span, glyph=True)
        mild = ui.heat_cell(" -10,866", -10866, span, glyph=True)
        self.assertIn("█", worst)
        self.assertNotIn("█", mild)


if __name__ == "__main__":
    unittest.main()
