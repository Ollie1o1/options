import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import unittest

from src.research import collect as C


def _briefing(sym="NVDA"):
    from src.intel.briefing import Briefing
    from src.intel.signals import Signal
    from src.intel.verdict import Driver, Verdict
    b = Briefing(sym, name="NVIDIA")
    b.state = {"price": 100.0, "ma50": 95.0, "ma200": 90.0, "ret_5d": 0.02,
               "ret_20d": 0.05, "rsi": 55.0, "iv_rank": 0.4,
               "days_to_earnings": 12,
               "_headlines": ["Headline one"], "_bounce": {"bounce_rate": 0.6, "n": 12},
               "_nearest_support": {"level": 92.0, "label": "50d", "pct": -0.08},
               "_nearest_resist": {"level": 110.0, "label": "high", "pct": 0.10},
               "_term_spread": 0.01}
    b.signals = {"trend": Signal("trend", 0.6, "UP", "above 200d")}
    b.verdict = Verdict("BUY", "medium", 0.4,
                        [Driver("trend", "+", "trend up", "A")], note="")
    b.primary_action = "Do the thing."
    b.secondary_action = "Also this."
    b.market_line = "Risk On - VIX 14"
    return b


class TestTickerPanel(unittest.TestCase):
    def test_serializes_briefing(self):
        p = C._panel_ticker("NVDA",
                            _gather_fn=lambda s: _briefing(s),
                            _cone_fn=lambda s: {10: {"p25": 0.2, "median": 0.25,
                                                     "p75": 0.3, "current": 0.27,
                                                     "pctile": 0.6, "min": 0.1,
                                                     "max": 0.5}},
                            _surface_fn=lambda s: [[7, 0.5], [30, 0.45]],
                            _chart_fn=lambda s: {"closes": [1, 2], "dates": ["a", "b"],
                                                 "ma50": [1, 2], "ma200": [],
                                                 "rsi": [50.0]})
        self.assertEqual(p["symbol"], "NVDA")
        self.assertEqual(p["verdict"]["call"], "BUY")
        self.assertEqual(p["signals"][0]["label"], "UP")
        self.assertEqual(p["support"]["level"], 92.0)
        self.assertEqual(p["headlines"], ["Headline one"])
        self.assertEqual(p["cone"][0]["window"], 10)
        self.assertEqual(p["term"], [[7, 0.5], [30, 0.45]])
        json.dumps(p)  # sidecar must be JSON-clean

    def test_failed_briefing_raises(self):
        from src.intel.briefing import Briefing
        bad = Briefing("XX", ok=False, error="no price history")
        with self.assertRaises(RuntimeError):
            C._panel_ticker("XX", _gather_fn=lambda s: bad)

    def test_chart_and_cone_failures_degrade_not_raise(self):
        def boom(s):
            raise ValueError("nope")
        p = C._panel_ticker("NVDA", _gather_fn=lambda s: _briefing(s),
                            _cone_fn=boom, _surface_fn=boom, _chart_fn=boom)
        self.assertIsNone(p["chart"])
        self.assertEqual(p["cone"], [])
        self.assertEqual(p["term"], [])


class TestRsiSeries(unittest.TestCase):
    def test_bounds_and_length(self):
        closes = [100 + (i % 7) - 3 for i in range(60)]
        rsi = C._rsi_series(closes)
        self.assertEqual(len(rsi), len(closes) - 14)
        self.assertTrue(all(0.0 <= v <= 100.0 for v in rsi))
        self.assertEqual(C._rsi_series([1, 2, 3]), [])


class TestBuild(unittest.TestCase):
    def test_build_records_failures_and_stays_json(self):
        def boom():
            raise RuntimeError("dead feed")
        fetchers = [("market", lambda: {"regime": {"vix": 15.0}}),
                    ("tape", boom)]
        data = C.build(symbol="nvda", slow=False, _fetchers=fetchers)
        self.assertEqual(data["meta"]["symbol"], "NVDA")
        self.assertTrue(data["meta"]["base"].endswith("_NVDA"))
        self.assertEqual(data["panels"]["market"], {"regime": {"vix": 15.0}})
        self.assertIsNone(data["panels"]["tape"])
        self.assertTrue(any("tape: RuntimeError" in f for f in data["failures"]))
        self.assertTrue(any("dead feed" in n for n in data["panels"]["notes"]))
        json.dumps(data)

    def test_market_only_build_has_no_ticker(self):
        data = C.build(symbol=None, slow=False, _fetchers=[])
        self.assertIsNone(data["meta"]["symbol"])
        # base is research_YYYYMMDD_HHMM — exactly 2 underscores, no symbol suffix
        self.assertEqual(data["meta"]["base"].count("_"), 2)
        self.assertIsNone(data["panels"]["ticker"])

    def test_default_fetchers_include_ticker_only_with_symbol(self):
        ids = [pid for pid, _ in C._default_fetchers("NVDA")]
        self.assertIn("ticker", ids)
        ids = [pid for pid, _ in C._default_fetchers(None)]
        self.assertNotIn("ticker", ids)


class TestPanelHelpers(unittest.TestCase):
    def test_movers_panel_shapes_rows(self):
        rows = C._panel_movers(_movers_fn=lambda: [("XLE", -0.031, "WEAK"),
                                                   ("NVDA", 0.052, "UPTREND")])
        self.assertEqual(rows[0]["sym"], "XLE")
        self.assertAlmostEqual(rows[0]["ret_5d_pct"], -3.1)

    def test_news_panel_keeps_urls(self):
        items = [{"title": "T1", "source": "reuters.com", "url": "https://x/1",
                  "published": "", "topic": "markets"}]
        p = C._panel_news(_fetch_fn=lambda: items)
        self.assertEqual(p["items"][0]["url"], "https://x/1")
        with self.assertRaises(RuntimeError):
            C._panel_news(_fetch_fn=lambda: [])


if __name__ == "__main__":
    unittest.main()
