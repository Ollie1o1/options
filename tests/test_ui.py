"""Component kit: ANSI-aware rules, kv rows, banners, cards, tables, meters."""
import re
import unittest

from src import formatting as fmt
from src import ui

ANSI_RE = re.compile(r'\033\[[0-9;]*m')


def strip(s):
    return ANSI_RE.sub("", s)


class UiTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def test_visible_len_ignores_ansi(self):
        fmt.set_color_enabled(True)
        colored = fmt.style("abc", "good")
        self.assertEqual(ui.visible_len(colored), 3)

    def test_pad_alignments(self):
        self.assertEqual(ui.pad("ab", 5), "ab   ")
        self.assertEqual(ui.pad("ab", 5, align="right"), "   ab")
        self.assertEqual(ui.pad("abc", 7, align="center"), "  abc  ")

    def test_pad_never_truncates(self):
        self.assertEqual(ui.pad("abcdef", 3), "abcdef")

    def test_rule_plain_and_titled(self):
        self.assertEqual(strip(ui.rule(10)), "─" * 10)
        titled = strip(ui.rule(20, title="GREEKS"))
        self.assertTrue(titled.startswith("─ GREEKS "))
        self.assertEqual(len(titled), 20)

    def test_kv_line_fixed_gutter(self):
        line = strip(ui.kv_line("Flow", ["PCR 0.82", "RSI 58"]))
        self.assertEqual(line, "  " + "Flow".ljust(ui.LABEL_W) + " PCR 0.82  RSI 58")

    def test_kv_line_accepts_string(self):
        self.assertIn("only", strip(ui.kv_line("X", "only")))

    def test_kv_line_drops_empty_segments(self):
        line = strip(ui.kv_line("X", ["a", "", None, "b"]))
        self.assertIn("a  b", line)

    def test_banner_width_and_title(self):
        out = strip(ui.banner("OPTIONS SCREENER", ["ctx line"], width=40))
        lines = out.split("\n")
        self.assertEqual(len(lines[0]), 40)
        self.assertIn("OPTIONS SCREENER", lines[0])
        self.assertIn("═", lines[0])
        self.assertEqual(lines[1], "  ctx line")
        self.assertEqual(lines[-1], "─" * 40)

    def test_card_unboxed_uses_rules(self):
        out = strip(ui.card("T1", ["body"], width=30))
        lines = out.split("\n")
        self.assertTrue(lines[0].startswith("─ T1 "))
        self.assertEqual(lines[1], "body")
        self.assertEqual(lines[-1], "─" * 30)

    def test_card_boxed_aligns_borders(self):
        fmt.set_color_enabled(True)
        out = ui.card("TICKET", [fmt.style("colored", "good"), "plain"],
                      width=30, boxed=True)
        widths = {ui.visible_len(ln) for ln in out.split("\n")}
        self.assertEqual(widths, {30})

    def test_table_alignment_with_ansi_cells(self):
        fmt.set_color_enabled(True)
        cols = [
            {"h": "Tkr", "w": 5},
            {"h": "PoP", "w": 5, "align": "right"},
        ]
        rows = [["NVDA", fmt.style("62%", "good")], ["F", "9%"]]
        out = ui.table(cols, rows)
        plain = [strip(ln) for ln in out.split("\n")]
        self.assertEqual(plain[2], "  NVDA    62%")
        self.assertEqual(plain[3], "  F        9%")

    def test_meter_fill(self):
        self.assertEqual(strip(ui.meter(0.5, width=10)), "█████░░░░░")
        self.assertEqual(strip(ui.meter(0.0, width=4)), "░░░░")
        self.assertEqual(strip(ui.meter(1.2, width=4)), "████")

    def test_clip_no_op_under_limit(self):
        self.assertEqual(ui.clip("short", 40), "short")

    def test_clip_breaks_on_word_boundary(self):
        out = ui.clip("High probability (>65%) • Strong R/R (2.1x)", 40)
        self.assertTrue(out.endswith("…"))
        self.assertLessEqual(ui.visible_len(out), 40)
        # never a hard mid-word cut like "(2…"
        self.assertFalse(out.endswith("(2…"))

    def test_clip_hard_cut_for_single_long_word(self):
        out = ui.clip("supercalifragilistic", 10)
        self.assertEqual(out, "supercali…")

    def test_spinner_noop_on_non_tty(self):
        import io
        buf = io.StringIO()  # StringIO has no isatty -> disabled
        with ui.spinner("loading", stream=buf):
            pass
        self.assertEqual(buf.getvalue(), "")

    def test_spinner_animates_on_tty(self):
        import io

        class FakeTTY(io.StringIO):
            def isatty(self):
                return True

        buf = FakeTTY()
        sp = ui.Spinner("load", stream=buf, interval=0.01)
        sp.start()
        # let it emit at least one frame
        import time
        time.sleep(0.05)
        sp.stop()
        out = buf.getvalue()
        self.assertIn("load", strip(out))
        self.assertTrue(any(f in out for f in ui.SPINNER_FRAMES))


class ScenarioTableTestCase(unittest.TestCase):
    def setUp(self):
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = None

    def _row(self, strike, premium):
        import pandas as pd
        return pd.Series({
            "underlying": 182.40, "strike": strike, "T_years": 0.1,
            "impliedVolatility": 0.42, "type": "call", "premium": premium,
        })

    def test_worthless_grid_shows_note_not_zeros(self):
        from src.trade_analysis import build_scenario_table
        out = build_scenario_table(self._row(480, 11.30), 0.043, 100)
        self.assertIn("worthless", out)
        # collapses to a one-line note, not a grid of zero rows
        self.assertNotIn("Move", out)
        self.assertEqual(out.count("$0.00"), 1)  # only in the note text

    def test_normal_grid_renders_table(self):
        from src.trade_analysis import build_scenario_table
        out = build_scenario_table(self._row(190, 4.20), 0.043, 100)
        self.assertIn("Move", out)
        self.assertNotIn("worthless", out)


if __name__ == "__main__":
    unittest.main()
