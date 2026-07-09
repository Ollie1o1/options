"""Desk visual layer — primitives and display-only panel smoke tests."""
import io
import os
import sys
import unittest
import unittest.mock
from contextlib import redirect_stdout

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import ui  # noqa: E402
from src import formatting as fmt  # noqa: E402


class ColorMixin:
    """Pin color state per-test rather than trusting import order.

    `fmt.supports_color()` memoizes into `fmt._COLOR_ENABLED` on first call, so
    a module-level env var only wins if this module imports first. Under the
    full suite (and under pytest's non-TTY run) it does not. Both helpers
    restore the previous flag on cleanup so neighbouring modules are unaffected.
    """

    def force_color_on(self):
        prev = fmt._COLOR_ENABLED
        prev_ct = os.environ.get("COLORTERM")
        fmt._COLOR_ENABLED = True
        os.environ["COLORTERM"] = "truecolor"

        def _restore():
            fmt._COLOR_ENABLED = prev
            if prev_ct is None:
                os.environ.pop("COLORTERM", None)
            else:
                os.environ["COLORTERM"] = prev_ct

        self.addCleanup(_restore)

    def force_color_off(self):
        prev = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False
        self.addCleanup(lambda: setattr(fmt, "_COLOR_ENABLED", prev))


class TestHeatCell(ColorMixin, unittest.TestCase):
    def setUp(self):
        self.force_color_on()
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


class TestGreeksByName(unittest.TestCase):
    def test_breakdown_renders_sorted_bars(self):
        from src import check_pnl
        rows = {"QQQ": [-312.0, -4.1], "SPY": [-198.0, -2.0], "AAPL": [-121.0, 1.2]}
        buf = io.StringIO()
        with redirect_stdout(buf):
            check_pnl._print_greeks_by_ticker(rows, width=100)
        out = buf.getvalue()
        # Largest |vega| name appears first, all names present, delta shown.
        self.assertLess(out.index("QQQ"), out.index("SPY"))
        self.assertIn("AAPL", out)
        self.assertIn("-312", out)
        self.assertIn("VEGA BY UNDERLYING", out.upper())

    def test_single_name_prints_nothing(self):
        from src import check_pnl
        buf = io.StringIO()
        with redirect_stdout(buf):
            check_pnl._print_greeks_by_ticker({"SPY": [-198.0, -2.0]}, width=100)
        self.assertEqual(buf.getvalue(), "")

    def test_overflow_names_roll_into_others(self):
        from src import check_pnl
        rows = {f"T{i}": [float(-100 - i), 0.5] for i in range(10)}
        buf = io.StringIO()
        with redirect_stdout(buf):
            check_pnl._print_greeks_by_ticker(rows, width=100, top=8)
        out = buf.getvalue()
        self.assertIn("others", out)


class TestPreStyledTitle(ColorMixin, unittest.TestCase):
    """A caller-styled title must survive rule()/card() unchanged.

    The staleness banner needs a severity color the kit does not own; wrapping
    it again in 'heading' would emit a dead escape whose only effect is to be
    immediately overridden.
    """

    def setUp(self):
        self.force_color_on()

    def test_rule_does_not_restyle_ansi_title(self):
        styled = fmt.style("DANGER", "bad", bold=True)
        out = ui.rule(60, title=styled)
        self.assertIn(styled, out)
        self.assertNotIn(fmt.style(styled, "heading"), out)

    def test_card_does_not_restyle_ansi_title(self):
        styled = fmt.style("DANGER", "bad", bold=True)
        out = ui.card(styled, ["body"], 60, boxed=True)
        self.assertIn(styled, out)
        # heading's steel-blue must not appear anywhere in the title region
        self.assertNotIn("38;2;130;170;210", out.splitlines()[0])

    def test_plain_title_still_gets_heading_style(self):
        out = ui.rule(60, title="PLAIN")
        self.assertIn("38;2;130;170;210", out)

    def test_widths_unaffected_by_prestyled_title(self):
        plain = ui.card("DANGER", ["body"], 60, boxed=True)
        styled = ui.card(fmt.style("DANGER", "bad"), ["body"], 60, boxed=True)
        self.assertEqual([ui.visible_len(l) for l in plain.splitlines()],
                         [ui.visible_len(l) for l in styled.splitlines()])


class TestSparkline(unittest.TestCase):
    def test_monotonic_rising(self):
        s = ui.sparkline([1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(ui.visible_len(s), 8)
        self.assertTrue(s.strip()[0] in "▁▂")
        self.assertIn("█", s)

    def test_nan_becomes_space(self):
        s = ui.sparkline([1, float("nan"), 3])
        self.assertEqual(ui.visible_len(s), 3)
        self.assertIn(" ", s)

    def test_flat_series_midlevel(self):
        s = ui.sparkline([5, 5, 5])
        self.assertEqual(ui.visible_len(s), 3)

    def test_empty_series(self):
        self.assertEqual(ui.sparkline([]), "")

    def test_all_nan_series(self):
        self.assertEqual(ui.sparkline([float("nan"), None]), "  ")


class TestBrailleChart(unittest.TestCase):
    def test_returns_height_rows(self):
        lines = ui.braille_chart(list(range(200)), width=40, height=5)
        self.assertEqual(len(lines), 5)
        # braille block chars are U+2800..U+28FF
        joined = "".join(lines)
        self.assertTrue(any(0x2800 <= ord(c) <= 0x28FF for c in joined))

    def test_too_short_returns_empty(self):
        self.assertEqual(ui.braille_chart([1.0], width=40, height=5), [])
        self.assertEqual(ui.braille_chart([], width=40, height=5), [])

    def test_flat_series_does_not_crash(self):
        lines = ui.braille_chart([3.0, 3.0, 3.0, 3.0], width=20, height=4)
        self.assertEqual(len(lines), 4)


class TestEquityCurve(ColorMixin, unittest.TestCase):
    def _rows(self, n=30):
        return [{
            "pnl_pct": (0.1 if i % 3 else -0.15),
            "entry_price": 5.0,
            "ticker": "AAPL",
            "exit_date": f"2026-06-{(i % 28) + 1:02d}",
            "date": None,
        } for i in range(n)]

    def _render(self, rows, rich: bool):
        """Render the panel with check_pnl's import-time capability flags pinned.

        HAS_FMT is evaluated at import, so it cannot be steered by env vars here.
        """
        from src import check_pnl
        with unittest.mock.patch.object(check_pnl, "HAS_FMT", rich), \
                unittest.mock.patch.object(check_pnl, "_HAS_UI_CP", rich):
            buf = io.StringIO()
            with redirect_stdout(buf):
                check_pnl._print_equity_curve(rows, width=100)
            return buf.getvalue()

    def test_renders_curve_and_underwater(self):
        self.force_color_on()
        out = self._render(self._rows(30), rich=True)
        self.assertIn("EQUITY", out.upper())
        self.assertTrue(any(0x2800 <= ord(c) <= 0x28FF for c in out))
        self.assertIn("Underwater", out)

    def test_plain_branch_has_no_braille(self):
        out = self._render(self._rows(30), rich=False)
        self.assertIn("EQUITY", out.upper())
        self.assertIn("max DD", out)
        self.assertFalse(any(0x2800 <= ord(c) <= 0x28FF for c in out))

    def test_underwater_strip_encodes_depth_not_height(self):
        """Deepest drawdown must be the TALLEST bar, since red = bad."""
        self.force_color_on()
        # Rises, then one big loss at the end → deepest drawdown is last.
        rows = [{"pnl_pct": 0.1, "entry_price": 5.0, "ticker": "AAPL",
                 "exit_date": f"2026-06-{i + 1:02d}", "date": None}
                for i in range(11)]
        rows.append({"pnl_pct": -0.9, "entry_price": 5.0, "ticker": "AAPL",
                     "exit_date": "2026-06-28", "date": None})
        out = self._render(rows, rich=True)
        strip = [ln for ln in out.splitlines() if "Underwater" in ln][0]
        bars = [c for c in strip if c in "▁▂▃▄▅▆▇█"]
        self.assertEqual(bars[-1], "█")   # deepest drawdown = full bar
        self.assertEqual(bars[0], "▁")    # at the peak = empty bar

    def test_too_few_trades_prints_nothing(self):
        self.assertEqual(self._render(self._rows(4), rich=True), "")


class TestTermCurve(unittest.TestCase):
    def test_curve_from_picks(self):
        import pandas as pd
        from src import cli_display
        df = pd.DataFrame([
            {"expiration": "2026-07-17", "impliedVolatility": 0.30, "T_years": 9 / 365},
            {"expiration": "2026-07-17", "impliedVolatility": 0.32, "T_years": 9 / 365},
            {"expiration": "2026-08-21", "impliedVolatility": 0.36, "T_years": 44 / 365},
        ])
        curve = cli_display._iv_term_curve_from_picks(df)
        self.assertEqual(len(curve), 2)
        self.assertEqual(curve[0][0], 9)      # sorted by dte ascending
        self.assertAlmostEqual(curve[0][1], 0.31, places=2)  # median of front

    def test_single_expiry_returns_empty(self):
        import pandas as pd
        from src import cli_display
        df = pd.DataFrame([
            {"expiration": "2026-07-17", "impliedVolatility": 0.30, "T_years": 9 / 365},
        ])
        self.assertEqual(cli_display._iv_term_curve_from_picks(df), [])

    def test_missing_columns_returns_empty(self):
        import pandas as pd
        from src import cli_display
        self.assertEqual(cli_display._iv_term_curve_from_picks(pd.DataFrame()), [])
        self.assertEqual(
            cli_display._iv_term_curve_from_picks(pd.DataFrame([{"expiration": "2026-07-17"}])), [])

    def test_absurd_ivs_filtered_out(self):
        import pandas as pd
        from src import cli_display
        df = pd.DataFrame([
            {"expiration": "2026-07-17", "impliedVolatility": 0.0, "T_years": 9 / 365},
            {"expiration": "2026-08-21", "impliedVolatility": 9.9, "T_years": 44 / 365},
        ])
        self.assertEqual(cli_display._iv_term_curve_from_picks(df), [])


if __name__ == "__main__":
    unittest.main()
