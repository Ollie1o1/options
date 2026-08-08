"""The chapters are hand-written data, so the tests check the things a human
editing prose will actually get wrong: a malformed block, an over-wide table,
a dropped chapter, and the claims the manual must never quietly lose."""
import unittest

from src import formatting as fmt
from src.help_desk import content, render


class ContentTest(unittest.TestCase):
    def setUp(self):
        self._color = fmt._COLOR_ENABLED
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = self._color

    def _text(self, chapter):
        """All prose in a chapter, flattened — for claim assertions. Recurses,
        because table rows nest two deep and dict cells nest inside those."""
        parts = []

        def walk(node):
            if isinstance(node, str):
                parts.append(node)
            elif isinstance(node, dict):
                for v in node.values():
                    walk(v)
            elif isinstance(node, (list, tuple)):
                for item in node:
                    walk(item)

        for block in chapter.body:
            walk(block[1:])
        return " ".join(parts)

    def test_seven_chapters_with_unique_keys(self):
        keys = [c.key for c in content.CHAPTERS]
        self.assertEqual(len(keys), 7)
        self.assertEqual(len(set(keys)), 7)
        self.assertEqual(
            keys,
            ["start", "picking", "verdict", "friction", "modes", "glossary",
             "trust"])

    def test_every_chapter_has_title_blurb_and_body(self):
        for c in content.CHAPTERS:
            self.assertTrue(c.title.strip(), c.key)
            self.assertTrue(c.blurb.strip(), c.key)
            self.assertTrue(c.body, c.key)

    def test_every_block_is_well_formed(self):
        for c in content.CHAPTERS:
            for block in c.body:
                self.assertIn(block[0], render.BLOCK_ARITY, f"{c.key}: {block!r}")
                self.assertEqual(len(block), render.BLOCK_ARITY[block[0]],
                                 f"{c.key}: {block!r}")

    def test_every_chapter_renders_within_width(self):
        for c in content.CHAPTERS:
            for ln in render.render_blocks(c.body):
                self.assertLessEqual(len(ln), render.WIDTH,
                                     f"{c.key}: {ln!r}")

    def test_every_table_row_matches_its_column_count(self):
        for c in content.CHAPTERS:
            for block in c.body:
                if block[0] != "table":
                    continue
                cols, rows = block[1], block[2]
                for row in rows:
                    self.assertEqual(len(row), len(cols), f"{c.key}: {row!r}")

    def test_every_kv_label_fits_the_gutter(self):
        """The renderer survives an over-long label by giving it its own line,
        but a glossary that silently un-aligns itself is a regression."""
        for c in content.CHAPTERS:
            for block in c.body:
                if block[0] == "kv":
                    self.assertLessEqual(len(block[1]), render.LABEL_W,
                                         f"{c.key}: {block[1]!r}")

    def test_titles_and_blurbs_fit_the_index_columns(self):
        from src.help_desk import menu
        for c in content.CHAPTERS:
            self.assertLessEqual(len(c.title), menu.TITLE_W, c.key)
            self.assertLessEqual(len(c.blurb), menu.BLURB_W, c.key)

    def test_real_money_off_is_stated_in_start_and_trust(self):
        by_key = {c.key: c for c in content.CHAPTERS}
        self.assertIn("Real money is OFF", self._text(by_key["start"]))
        self.assertIn("OFF", self._text(by_key["trust"]))

    def test_picking_chapter_carries_the_five_step_ladder(self):
        picking = {c.key: c for c in content.CHAPTERS}["picking"]
        headings = [b[1] for b in picking.body if b[0] == "h"]
        for step in ("1 ·", "2 ·", "3 ·", "4 ·", "5 ·"):
            self.assertTrue(any(h.startswith(step) for h in headings), step)

    def test_verdict_chapter_quotes_the_refusal_reasons(self):
        """If candidate_verdict.py's wording changes, this must too."""
        text = self._text({c.key: c for c in content.CHAPTERS}["verdict"])
        for reason in ("no two-sided quote on every leg",
                       "credit disappears once the spread is crossed"):
            self.assertIn(reason, text)

    def test_friction_ceiling_matches_the_code(self):
        from src.candidate_verdict import DEFAULT_MAX_FRICTION
        text = self._text({c.key: c for c in content.CHAPTERS}["friction"]
                          ) + self._text(
            {c.key: c for c in content.CHAPTERS}["verdict"])
        self.assertIn(f"{DEFAULT_MAX_FRICTION:.0%}", text)

    def test_modes_chapter_names_every_scan_mode(self):
        """The manual must not drift from the menu it documents. The TEST
        imports the screener; the help package never does."""
        text = self._text({c.key: c for c in content.CHAPTERS}["modes"])
        for mode in ("TICKER", "ALL", "DISCOVER", "SELL", "SPREADS", "IRON",
                     "PORTFOLIO", "MY LIST", "LOTTERY", "INTEL", "SQUEEZE",
                     "PROB LAB", "STRUCTURE"):
            self.assertIn(mode, text, mode)

    def test_glossary_warns_about_the_max_loss_trap(self):
        text = self._text({c.key: c for c in content.CHAPTERS}["glossary"])
        self.assertIn("max_loss", text)
        self.assertIn("Never size off this field", text)


if __name__ == "__main__":
    unittest.main()
