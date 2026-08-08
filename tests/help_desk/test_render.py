"""The block renderer: pure, width-disciplined, loud about bad content."""
import unittest

from src import formatting as fmt
from src.help_desk import render


class RenderBlocksTest(unittest.TestCase):
    def setUp(self):
        # Pin the flag itself, never an env var — see the color-discipline
        # tests for why env-based color toggling leaks across the suite.
        self._color = fmt._COLOR_ENABLED
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = self._color

    def test_paragraph_wraps_within_width(self):
        lines = render.render_blocks([("p", "word " * 80)])
        self.assertTrue(lines)
        for ln in lines:
            self.assertLessEqual(len(ln), render.WIDTH)

    def test_paragraph_keeps_deliberate_line_breaks(self):
        lines = render.render_blocks([("p", "first\nsecond")])
        self.assertEqual([ln.strip() for ln in lines], ["first", "second"])

    def test_code_block_is_never_wrapped(self):
        text = "python run.py --mode discover --top 5 --no-ai"
        lines = render.render_blocks([("code", text)])
        self.assertEqual([ln.strip() for ln in lines], [text])

    def test_gap_and_rule_render(self):
        self.assertEqual(render.render_blocks([("gap",)]), [""])
        self.assertEqual(len(render.render_blocks([("rule",)])), 1)

    def test_numbered_list_hangs_indent(self):
        lines = render.render_blocks(
            [("num", ["first item " * 12, "second item"])])
        self.assertTrue(lines[0].lstrip().startswith("1."))
        self.assertTrue(any(ln.lstrip().startswith("2.") for ln in lines))
        # A continuation line must not start with a marker.
        self.assertFalse(lines[1].lstrip().startswith("2."))

    def test_bullet_wraps_and_hangs(self):
        lines = render.render_blocks([("bullet", "point " * 40)])
        self.assertTrue(lines[0].lstrip().startswith("·"))
        for ln in lines:
            self.assertLessEqual(len(ln), render.WIDTH)

    def test_callout_is_boxed_within_width(self):
        lines = render.render_blocks([("callout", "bad", "do not do this " * 8)])
        self.assertTrue(lines[0].startswith("┌"))
        self.assertTrue(lines[-1].startswith("└"))
        for ln in lines:
            self.assertLessEqual(len(ln), render.WIDTH)

    def test_table_stays_within_width(self):
        cols = [{"h": "structure", "w": 20}, {"h": "toll", "w": 30}]
        lines = render.render_blocks(
            [("table", cols, [["two-leg vertical", "27-33% of the credit"]])])
        for ln in lines:
            self.assertLessEqual(len(ln), render.WIDTH)

    def test_consecutive_paragraphs_are_separated(self):
        """Two paragraphs must not render as one wall of text."""
        lines = render.render_blocks([("p", "first"), ("p", "second")])
        self.assertEqual([ln.strip() for ln in lines], ["first", "", "second"])

    def test_consecutive_bullets_stay_tight(self):
        lines = render.render_blocks([("bullet", "one"), ("bullet", "two")])
        self.assertNotIn("", [ln.strip() for ln in lines])

    def test_consecutive_kv_rows_stay_tight(self):
        lines = render.render_blocks([("kv", "a", "one"), ("kv", "b", "two")])
        self.assertEqual(len(lines), 2)

    def test_heading_does_not_get_a_doubled_blank(self):
        lines = render.render_blocks([("p", "text"), ("h", "Next")])
        self.assertEqual([ln.strip() for ln in lines], ["text", "", "NEXT"])

    def test_over_wide_table_cell_is_clipped_not_overflowed(self):
        cols = [{"h": "a", "w": 8}, {"h": "b", "w": 8}]
        lines = render.render_blocks(
            [("table", cols, [["x" * 40, "y"]])])
        for ln in lines:
            self.assertLessEqual(len(ln), 2 + 8 + 1 + 8)

    def test_unknown_block_tag_raises(self):
        with self.assertRaises(ValueError):
            render.render_blocks([("nope", "x")])

    def test_wrong_arity_raises(self):
        with self.assertRaises(ValueError):
            render.render_blocks([("kv", "only-one-value")])

    def test_render_is_pure(self):
        blocks = [("h", "Title"), ("p", "Body text here."),
                  ("bullet", "a point"), ("kv", "label", "value")]
        self.assertEqual(render.render_blocks(blocks),
                         render.render_blocks(blocks))


if __name__ == "__main__":
    unittest.main()
