import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import unittest
from datetime import datetime

from src.morning import collect


class TestSessionPhase(unittest.TestCase):
    def test_premarket_open_closed_weekend(self):
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 8, 0)), "pre-market")
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 10, 30)), "open")
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 17, 0)), "closed")
        self.assertEqual(collect.session_phase(datetime(2026, 7, 11, 10, 30)), "closed")  # Saturday
        self.assertEqual(collect.session_phase(datetime(2026, 7, 10, 3, 0)), "closed")


class TestSafeHarness(unittest.TestCase):
    def test_failure_is_captured_not_raised(self):
        panels, failures = {}, []
        def boom():
            raise RuntimeError("dead fetch")
        collect._safe("market", boom, panels, failures)
        self.assertIsNone(panels["market"])
        self.assertEqual(len(failures), 1)
        self.assertIn("market: RuntimeError: dead fetch", failures[0])

    def test_success_stores_value(self):
        panels, failures = {}, []
        collect._safe("gate", lambda: {"ok": 1}, panels, failures)
        self.assertEqual(panels["gate"], {"ok": 1})
        self.assertEqual(failures, [])


class TestBuildSkeleton(unittest.TestCase):
    def test_build_meta_and_all_panels_present(self):
        # slow=False and stubbed fetchers: build must never hit the network in tests
        data = collect.build(now=datetime(2026, 7, 10, 9, 0), slow=False,
                             _fetchers=[("market", lambda: {"x": 1})])
        self.assertEqual(data["meta"]["schema"], 1)
        self.assertEqual(data["meta"]["date"], "2026-07-10")
        self.assertEqual(data["meta"]["session"], "pre-market")
        self.assertEqual(data["meta"]["sidecar"], "2026-07-10.json")
        for pid in collect.PANEL_IDS:
            self.assertIn(pid, data["panels"])
        self.assertEqual(data["panels"]["market"], {"x": 1})
        self.assertIsInstance(data["panels"]["notes"], list)

    def test_build_notes_name_failed_panels(self):
        def boom():
            raise ValueError("no db")
        data = collect.build(now=datetime(2026, 7, 10, 9, 0), slow=False,
                             _fetchers=[("vol", boom)])
        self.assertIsNone(data["panels"]["vol"])
        self.assertTrue(any("vol" in n for n in data["panels"]["notes"]))


class TestMarketPanel(unittest.TestCase):
    def test_panel_market_shapes_injected(self):
        regime = {"vix": 14.2, "posture": "RISK_ON", "vix_term_structure": "CONTANGO"}
        dirs = {"SPY": {"last": 620.0, "chg_1d_pct": 0.4, "chg_5d_pct": 1.2,
                        "closes": [610.0, 612.0, 615.0, 618.0, 620.0],
                        "verdict": "BULLISH"}}
        out = collect._panel_market(_regime_fn=lambda: regime,
                                    _dirs_fn=lambda: dirs,
                                    _rates_fn=lambda: {"t10y": 4.2, "t3m": 4.4})
        self.assertEqual(out["regime"]["vix"], 14.2)
        self.assertEqual(out["indexes"][0]["sym"], "SPY")
        self.assertAlmostEqual(out["rates"]["slope"], -0.2, places=6)

    def test_panel_market_rates_failure_is_partial_not_fatal(self):
        def boom():
            raise RuntimeError("fred down")
        out = collect._panel_market(_regime_fn=lambda: {"vix": 15.0},
                                    _dirs_fn=lambda: {}, _rates_fn=boom)
        self.assertEqual(out["rates"], {"t10y": None, "t3m": None, "slope": None})
        self.assertEqual(out["regime"]["vix"], 15.0)

    def test_index_rows_from_closes(self):
        closes = [100.0 + i for i in range(30)]
        rows = collect._index_rows_from_closes({"SPY": closes})
        self.assertEqual(rows["SPY"]["last"], 129.0)
        self.assertAlmostEqual(rows["SPY"]["chg_1d_pct"],
                               (129.0 / 128.0 - 1) * 100, places=6)
        self.assertAlmostEqual(rows["SPY"]["chg_5d_pct"],
                               (129.0 / 124.0 - 1) * 100, places=6)
        self.assertEqual(len(rows["SPY"]["closes"]), 30)


class TestHealthGatePanels(unittest.TestCase):
    def test_panel_health_serializes_report(self):
        class J:
            name, cadence, last_run = "auto-log", "business-daily", "2026-07-09"
            business_days_stale, severity = 1, "OK"
        class R:
            jobs, worst = [J()], "OK"
        out = collect._panel_health(_report_fn=lambda: R())
        self.assertEqual(out["worst"], "OK")
        self.assertEqual(out["jobs"][0]["name"], "auto-log")
        self.assertEqual(out["jobs"][0]["stale_days"], 1)

    def test_panel_gate_carries_target(self):
        ev = {"pooled_ic": 0.1, "p_value": 0.4, "n_oos": 30,
              "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-07-01"}
        out = collect._panel_gate(_evidence_fn=lambda: dict(ev))
        self.assertEqual(out["cohort_n"], 2)
        self.assertGreaterEqual(out["target_n"], 50)


class TestVolPanel(unittest.TestCase):
    def test_panel_vol_caps_sorts_and_passthrough(self):
        movers = [{"symbol": f"S{i}", "iv": 0.3, "d_iv": 0.001 * i} for i in range(20)]
        vrp = [{"symbol": "SPY", "iv": 0.15, "rv": 0.12, "vrp": 3.1, "label": "RICH"}]
        out = collect._panel_vol(_rows_fn=lambda: (movers, vrp))
        self.assertEqual(len(out["movers"]), 8)
        self.assertEqual(out["movers"][0]["symbol"], "S19")  # biggest |d_iv| first
        self.assertEqual(out["vrp"][0]["symbol"], "SPY")
        self.assertEqual(out["n_cov"], 1)
        self.assertIn("carry", out["crypto_note"].lower())


class TestMacroEventsPanel(unittest.TestCase):
    def test_panel_macro_events_shapes(self):
        cal = [{"date": "2026-07-15", "name": "CPI"}]
        out = collect._panel_macro_events(
            _calendar_fn=lambda: cal,
            _pulse_fn=lambda: ("risk tone neutral", ["Headline A", "Headline B"]),
            _earnings_fn=lambda: [{"sym": "NVDA", "date": "2026-08-26"}])
        self.assertEqual(out["calendar"][0]["name"], "CPI")
        self.assertEqual(out["pulse"], "risk tone neutral")
        self.assertEqual(out["earnings"][0]["sym"], "NVDA")

    def test_panel_macro_events_pulse_failure_partial(self):
        def boom():
            raise RuntimeError("rss down")
        out = collect._panel_macro_events(_calendar_fn=lambda: [],
                                          _pulse_fn=boom,
                                          _earnings_fn=lambda: [])
        self.assertIsNone(out["pulse"])
        self.assertEqual(out["headlines"], [])


if __name__ == "__main__":
    unittest.main()
