import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest


class TestDeskTab(unittest.TestCase):
    def test_structure_panel_is_registered(self):
        from src.research import collect
        self.assertIn("structure", collect.PANEL_IDS)

    def test_structure_tab_registered_and_builder_exists(self):
        from src.research import render
        self.assertIn("structure", dict(render._TAB_ORDER))
        self.assertIn("structure", render._TAB_BUILDERS)

    def test_tab_hides_itself_when_panel_empty(self):
        from src.research import render
        data = {"meta": {}, "panels": {}, "failures": []}
        self.assertNotIn("structure", dict(render._tabs_present(data)))

    def test_ticker_tab_still_hides_itself(self):
        # regression: _OPTIONAL_TABS refactor must not change ticker behaviour
        from src.research import render
        data = {"meta": {}, "panels": {}, "failures": []}
        self.assertNotIn("ticker", dict(render._tabs_present(data)))

    def test_tab_renders_when_panel_present(self):
        from src.research import render
        data = {"meta": {}, "failures": [], "panels": {"structure": {
            "rows": [{"strategy": "Bull Put", "breakeven_hit": 0.375,
                      "realized_hit": 0.662, "margin": 0.287,
                      "state": "ACTIVE", "n": 68,
                      "ci_includes_zero": False}]}}}
        html = render._tab_structure(data)
        self.assertIn("Bull Put", html)
        self.assertIn("ACTIVE", html)

    def test_untrusted_margin_is_marked_in_html(self):
        from src.research import render
        data = {"meta": {}, "failures": [], "panels": {"structure": {
            "rows": [{"strategy": "Iron Condor", "breakeven_hit": 0.457,
                      "realized_hit": 0.466, "margin": 0.008,
                      "state": "ACTIVE", "n": 73,
                      "ci_includes_zero": True}]}}}
        self.assertIn("~", render._tab_structure(data))


class TestMenuWiring(unittest.TestCase):
    def test_structure_menu_handler_exists(self):
        import src.options_screener as S
        self.assertTrue(hasattr(S, "_run_structure_menu"))
