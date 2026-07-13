import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import tempfile
import unittest

from src.desk_kit import hub, shell, theme


class TestTheme(unittest.TestCase):
    def test_light_and_dark_define_identical_keys(self):
        self.assertEqual(set(theme.LIGHT), set(theme.DARK))

    def test_css_tokens_defines_both_blocks(self):
        css = theme.css_tokens()
        self.assertIn(":root {", css)
        self.assertIn('[data-theme="dark"]', css)

    def test_heat_inks_two_hexes(self):
        hl, hd = theme.heat_inks(5.0, 10.0)
        self.assertTrue(hl.startswith("#") and hd.startswith("#"))


class TestShell(unittest.TestCase):
    def test_page_is_self_contained_document(self):
        html = shell.page("T", shell.masthead("TEST", ""), "<p>b</p>")
        self.assertTrue(html.startswith("<!DOCTYPE html>"))
        self.assertIn("desk-theme", html)          # one shared theme key
        self.assertNotIn("http://", html)
        self.assertNotIn("https://", html)          # no CDN, no external fetch
        self.assertNotIn("\x1b[", html)             # no ANSI leakage

    def test_masthead_cross_links(self):
        m = shell.masthead("TEARSHEET", "", where="tearsheets")
        self.assertIn("../index.html", m)
        self.assertIn("../research/latest.html", m)
        self.assertIn("../briefings/latest.html", m)

    def test_masthead_unknown_where_has_no_links(self):
        self.assertNotIn("desklink", shell.masthead("X", "", where=None))

    def test_escaping(self):
        self.assertIn("&lt;s&gt;",
                      shell.card("<s>", "b", span=6))
        self.assertIn("&lt;i&gt;", shell.badge("<i>"))

    def test_tabbar_and_anchor_nav(self):
        t = shell.tabbar([("a", "Alpha"), ("b", "Beta")])
        self.assertIn('data-tabgroup', t)
        self.assertIn('data-tab="a"', t)
        a = shell.anchor_nav([("s1", "One")])
        self.assertIn('data-spynav', a)
        self.assertIn('href="#s1"', a)

    def test_strip_and_rows_table(self):
        s = shell.strip([("A", "1", ""), ("B", "2", "bad")])
        self.assertIn("repeat(2", s)
        self.assertIn("sv b", s)
        rt = shell.rows_table([("k", "v")])
        self.assertIn('class="n"', rt)


class TestHub(unittest.TestCase):
    def test_title_parsing(self):
        self.assertEqual(hub._title("tearsheets", "NVDA_207.5P_20260724.html"),
                         "NVDA 207.5P · exp 2026-07-24")
        self.assertEqual(hub._title("research", "research_20260712_1232_NVDA.html"),
                         "2026-07-12 12:32 · NVDA")
        self.assertEqual(hub._title("research", "research_20260712_1232.html"),
                         "2026-07-12 12:32 · market")
        self.assertEqual(hub._title("briefings", "2026-07-11.html"),
                         "2026-07-11")

    def test_build_index_pure_and_complete(self):
        entries = {"briefings": [{"name": "a.html", "href": "briefings/a.html",
                                  "title": "A", "mtime": 1, "when": "w"}],
                   "research": [], "tearsheets": []}
        html = hub.build_index(entries, generated_at="now")
        self.assertIn("briefings/a.html", html)
        self.assertIn("no research report", html)
        self.assertEqual(html, hub.build_index(entries, generated_at="now"))

    def test_scan_and_write_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            os.makedirs(os.path.join(d, "tearsheets"))
            p = os.path.join(d, "tearsheets", "AAPL_320C_20260821.html")
            with open(p, "w") as f:
                f.write("<html></html>")
            # latest.html and index.html are never listed as reports
            with open(os.path.join(d, "tearsheets", "latest.html"), "w") as f:
                f.write("x")
            out = hub.write_index(d)
            self.assertTrue(os.path.exists(out))
            with open(out) as f:
                idx = f.read()
            self.assertIn("AAPL 320C", idx)
            self.assertNotIn(">latest.html<", idx)

    def test_refresh_latest(self):
        with tempfile.TemporaryDirectory() as d:
            src = os.path.join(d, "2026-07-13.html")
            with open(src, "w") as f:
                f.write("payload")
            alias = hub.refresh_latest(d, src)
            with open(alias) as f:
                self.assertEqual(f.read(), "payload")

    def test_refresh_never_raises(self):
        self.assertIsNone(hub.refresh("/nonexistent/dir/zzz"))


if __name__ == "__main__":
    unittest.main()
