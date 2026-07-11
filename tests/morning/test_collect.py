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


if __name__ == "__main__":
    unittest.main()
