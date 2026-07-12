import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import unittest

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "desk.json")


def _fixture():
    with open(FIX) as f:
        return json.load(f)


class TestRenderSkeleton(unittest.TestCase):
    def test_full_document(self):
        from src.research.render import render
        html = render(_fixture())
        self.assertIn("<!DOCTYPE html>", html)
        self.assertIn("Research Desk", html)
        for tab in ("market", "volatility", "macro", "ticker"):
            self.assertIn('data-tab="' + tab + '"', html)
            self.assertIn('id="pane-' + tab + '"', html)
        self.assertIn("data-theme", html)       # theme JS
        self.assertIn('id="tip"', html)         # shared tooltip node

    def test_market_tab_content(self):
        from src.research.render import render
        html = render(_fixture())
        self.assertIn("14.2", html)             # VIX KPI
        self.assertIn("RISK_ON", html.replace("RISK ON", "RISK_ON"))
        self.assertIn("CPI", html)              # calendar
        self.assertIn("XLE", html)              # movers
        self.assertIn("+5.2%", html)            # signed mover label

    def test_staleness_banner_when_health_warn(self):
        from src.research.render import render
        html = render(_fixture())
        self.assertIn("auto-log", html)         # stale job named in banner

    def test_dead_panel_degrades_to_placeholder(self):
        from src.research.render import render
        data = _fixture()
        data["panels"]["tape"] = None
        data["failures"] = ["tape: RuntimeError: no index history"]
        html = render(data)
        self.assertIn("no index history", html)
        self.assertIn("<!DOCTYPE html>", html)

    def test_everything_dead_still_renders(self):
        from src.research.render import render
        data = _fixture()
        for pid in list(data["panels"]):
            if pid != "notes":
                data["panels"][pid] = None
        html = render(data)
        self.assertIn("<!DOCTYPE html>", html)
        self.assertNotIn("None", html.replace("none", ""))


if __name__ == "__main__":
    unittest.main()
