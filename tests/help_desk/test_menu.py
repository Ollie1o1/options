"""The chapter loop, driven through injected I/O so no TTY is needed."""
import unittest

from src import formatting as fmt
from src.help_desk import content, menu


class MenuTest(unittest.TestCase):
    def setUp(self):
        self._color = fmt._COLOR_ENABLED
        fmt.set_color_enabled(False)

    def tearDown(self):
        fmt._COLOR_ENABLED = self._color

    def _run(self, answers):
        out = []
        it = iter(answers)

        def _in(_prompt=""):
            try:
                return next(it)
            except StopIteration:
                raise EOFError
        menu.run_menu(input_fn=_in, output_fn=lambda s="": out.append(str(s)))
        return "\n".join(out)

    def test_back_returns_immediately(self):
        self.assertIn("HELP", self._run(["B"]))

    def test_empty_input_returns(self):
        self.assertIn("HELP", self._run([""]))

    def test_eof_returns_without_raising(self):
        self._run([])

    def test_chapter_by_number_renders_its_title(self):
        text = self._run(["1", "B"])
        self.assertIn(content.CHAPTERS[0].title, text)

    def test_chapter_by_key_renders(self):
        text = self._run(["friction", "B"])
        self.assertIn("FRICTION AND COST", text)

    def test_chapter_by_title_prefix_renders(self):
        text = self._run(["gloss", "B"])
        self.assertIn("GLOSSARY", text)

    def test_out_of_range_number_is_not_a_selection(self):
        text = self._run(["99", "B"])
        self.assertIn("Unknown choice", text)

    def test_read_all_renders_every_chapter(self):
        text = self._run(["A", "B"])
        for chapter in content.CHAPTERS:
            self.assertIn(chapter.title, text)

    def test_unknown_choice_reprints_without_raising(self):
        text = self._run(["zzz", "B"])
        self.assertIn("zzz", text)
        self.assertIn("Unknown choice", text)

    def test_index_lists_every_chapter_and_both_commands(self):
        idx = menu.format_index()
        for chapter in content.CHAPTERS:
            self.assertIn(chapter.title, idx)
            self.assertIn(chapter.blurb, idx)
        self.assertIn("READ ALL", idx)
        self.assertIn("BACK", idx)

    def test_index_stays_within_width(self):
        for line in menu.format_index().splitlines():
            self.assertLessEqual(len(line), menu.WIDTH, repr(line))

    def test_page_lines_splits_by_height(self):
        pages = menu.page_lines([str(i) for i in range(10)], 4)
        self.assertEqual(len(pages), 3)
        self.assertEqual(pages[0], ["0", "1", "2", "3"])
        self.assertEqual(pages[-1], ["8", "9"])

    def test_page_lines_returns_one_page_when_height_is_zero(self):
        lines = ["a", "b", "c"]
        self.assertEqual(menu.page_lines(lines, 0), [lines])

    def test_page_lines_returns_one_page_when_it_already_fits(self):
        lines = ["a", "b", "c"]
        self.assertEqual(menu.page_lines(lines, 10), [lines])

    def test_non_tty_gets_a_single_page(self):
        """Piping the manual must yield the whole chapter, not a stalled pager."""
        self.assertEqual(menu._terminal_height(), 0)


if __name__ == "__main__":
    unittest.main()
