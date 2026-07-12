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


class TestVolTab(unittest.TestCase):
    def test_movers_vrp_and_note(self):
        from src.research.render import render
        html = render(_fixture())
        self.assertIn("IV movers", html)
        self.assertIn("+3.0vp", html)      # NVDA d_iv 0.03 -> +3.0vp signed label
        self.assertIn("RICH", html)        # VRP label chip
        self.assertIn("BTC carry note", html)

    def test_dead_vol_panel_placeholder(self):
        from src.research.render import render
        data = _fixture()
        data["panels"]["vol"] = None
        data["failures"] = ["vol: RuntimeError: chain archive stale"]
        html = render(data)
        self.assertIn("chain archive stale", html)


class TestMacroTab(unittest.TestCase):
    def test_pulse_news_signals(self):
        from src.research.render import render
        html = render(_fixture())
        self.assertIn("risk-on", html)
        self.assertIn("What would flip it", html)
        self.assertIn("https://example.com/1", html)
        self.assertIn("CLUSTER BUY", html)
        self.assertIn("XLK", html)          # outlook top

    def test_news_titles_are_escaped(self):
        from src.research.render import render
        html = render(_fixture())
        self.assertNotIn("<script>alert(1)</script>", html)
        self.assertIn("&lt;script&gt;", html)


class TestTickerTab(unittest.TestCase):
    def test_verdict_actions_signals_charts(self):
        from src.research.render import render
        html = render(_fixture())
        self.assertIn("BUY", html)
        self.assertIn("v-buy", html)                 # verdict banner class
        self.assertIn("Buy the 50d-MA retest", html) # primary action
        self.assertIn("above 200d (+21.5%)", html)   # signal detail
        self.assertIn("50d MA", html)                # support band label
        self.assertIn("What to do", html)
        self.assertIn("NVDA headline one", html)

    def test_no_ticker_hides_tab(self):
        from src.research.render import render
        data = _fixture()
        data["panels"]["ticker"] = None
        data["meta"]["symbol"] = None
        html = render(data)
        self.assertNotIn('data-tab="ticker"', html)
        self.assertNotIn('id="pane-ticker"', html)

    def test_missing_chart_degrades(self):
        from src.research.render import render
        data = _fixture()
        data["panels"]["ticker"]["chart"] = None
        data["panels"]["ticker"]["cone"] = []
        data["panels"]["ticker"]["term"] = []
        html = render(data)
        self.assertIn("price history unavailable", html)


if __name__ == "__main__":
    unittest.main()
