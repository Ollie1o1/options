"""Color discipline: green/red mean directional sign only."""
import os
import re
import sys
import unittest

from src import formatting as fmt

ANSI_RE = re.compile(r'\033\[[0-9;]*m')

_SCRIPTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts")


def codes(s):
    return ANSI_RE.findall(s)


def sign_codes():
    """All green/red ANSI prefixes that signal directional sign.

    Covers both the truecolor theme styles and the raw ANSI color constants,
    since decorative coloring in legacy code uses the raw Colors.* values.
    """
    greens = {
        ANSI_RE.findall(fmt.style("x", "good"))[0],
        fmt.Colors.GREEN,
    }
    reds = {
        ANSI_RE.findall(fmt.style("x", "bad"))[0],
        fmt.Colors.RED,
        fmt.Colors.BRIGHT_RED,
    }
    return greens, reds


class StyleSignTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(True)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_positive_uses_good(self):
        pos = fmt.style_sign("+$131", 131)
        self.assertEqual(codes(pos), codes(fmt.style("+$131", "good")))

    def test_negative_uses_bad(self):
        neg = fmt.style_sign("-$12", -12)
        self.assertEqual(codes(neg), codes(fmt.style("-$12", "bad")))

    def test_zero_is_neutral_value(self):
        z = fmt.style_sign("$0", 0)
        self.assertEqual(codes(z), codes(fmt.style("$0", "value")))


class AnalysisLineDisciplineTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(True)
        if _SCRIPTS not in sys.path:
            sys.path.insert(0, _SCRIPTS)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def _lines(self):
        from ui_preview import df
        from src.cli_display import format_analysis_lines
        row = df().iloc[0]
        return format_analysis_lines(row, 0.42, "Discovery scan")

    def _find(self, needle):
        for ln in self._lines():
            if needle in ANSI_RE.sub("", ln):
                return ln
        self.fail(f"no line containing {needle!r}")

    def test_flow_line_has_no_sign_color(self):
        greens, reds = sign_codes()
        flow = self._find("PCR")
        for c in greens | reds:
            self.assertNotIn(c, flow, "PCR/sentiment must not use green/red")

    def test_context_term_has_no_sign_color(self):
        greens, reds = sign_codes()
        ctx = self._find("Term")
        for c in greens | reds:
            self.assertNotIn(c, ctx, "term structure must not use green/red")


class ComparisonDisciplineTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(True)
        if _SCRIPTS not in sys.path:
            sys.path.insert(0, _SCRIPTS)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def _render(self):
        import io
        import contextlib
        from ui_preview import df
        from src.cli_display import print_comparison_table
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            print_comparison_table(df(), mode="Discovery scan")
        return buf.getvalue()

    def test_sign_color_only_on_ev_or_svi(self):
        greens, reds = sign_codes()
        out = self._render()
        reset = fmt.Colors.RESET
        # every green/red-colored segment must be an EV ($) or SVI (CHEAP/RICH) cell
        for code in greens | reds:
            idx = 0
            while True:
                i = out.find(code, idx)
                if i < 0:
                    break
                end = out.find(reset, i)
                seg = ANSI_RE.sub("", out[i:end if end > 0 else len(out)])
                self.assertTrue(
                    any(tok in seg for tok in ("$", "CHEAP", "RICH")),
                    f"sign color on non-EV/SVI cell: {seg!r}")
                idx = i + len(code)

    def test_ev_cell_is_sign_colored(self):
        greens, _ = sign_codes()
        out = self._render()
        # the positive-EV rows should carry a green code somewhere
        self.assertTrue(any(c in out for c in greens), "EV/SVI sign color missing")


if __name__ == "__main__":
    unittest.main()
