"""Tests for src/macro_pulse/context.py — pure, offline."""
from __future__ import annotations

import os
import tempfile
import unittest

from src.macro_pulse import context as C


class SectorMapTest(unittest.TestCase):
    def test_geopolitics_maps_to_defense_and_energy(self):
        labels = [lbl for lbl, etf in C.sectors_for("geopolitics")]
        etfs = [etf for lbl, etf in C.sectors_for("geopolitics")]
        self.assertIn("defense", labels)
        self.assertIn("ITA", etfs)
        self.assertIn("XLE", etfs)

    def test_unknown_theme_returns_empty(self):
        self.assertEqual(C.sectors_for("other"), [])
        self.assertEqual(C.sectors_for("nonsense"), [])

    def test_every_worldnews_theme_has_a_mapping(self):
        # 'other' is intentionally unmapped; all real themes must map.
        for theme in ("fed_rates", "inflation", "jobs", "trade_tariffs",
                      "geopolitics", "earnings_tech", "energy", "crypto"):
            self.assertTrue(C.sectors_for(theme),
                            f"{theme} has no sector mapping")

    def test_dataclasses_construct(self):
        tr = C.ThemeRead(theme="energy", score=0.3, n=4, pctile=None, z=None,
                         sectors=C.sectors_for("energy"), top_headline="Oil up")
        self.assertEqual(tr.read, "")
        ctx = C.MacroContext(
            pulse=0.1, pulse_pctile=None, pulse_z=None, lean="NEUTRAL",
            confidence=50, n_items=10, n_sources=4, bull_pct=0.5,
            themes=[tr], event_active=False, event_name=None, event_date=None,
            next_events=[], n_history=0)
        self.assertEqual(ctx.narrative_source, "")


class StatsTest(unittest.TestCase):
    def test_percentile_cold_start_is_none(self):
        self.assertIsNone(C.percentile(0.5, [0.1, 0.2]))  # <5 samples

    def test_percentile_value(self):
        samples = [0.0, 0.1, 0.2, 0.3, 0.4]
        # 0.3 is >= 4 of 5 samples (incl. itself) -> 80th pct
        self.assertEqual(C.percentile(0.3, samples), 80.0)

    def test_zscore_cold_start_and_zero_stdev(self):
        self.assertIsNone(C.zscore(0.5, [0.1, 0.2]))          # <5
        self.assertIsNone(C.zscore(0.5, [0.2] * 6))           # zero stdev

    def test_zscore_value_sign(self):
        z = C.zscore(1.0, [0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        self.assertIsNotNone(z)
        self.assertGreater(z, 0.0)


class HistoryDbTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "macro_pulse.db")

    def test_load_missing_db_returns_empty(self):
        self.assertEqual(C.load_history(self.db), [])

    def test_persist_then_load_roundtrip(self):
        C.persist_reading(0.12, {"geopolitics": 0.3, "fed_rates": -0.1}, self.db)
        C.persist_reading(-0.05, {"geopolitics": 0.1}, self.db)
        rows = C.load_history(self.db, limit=30)
        self.assertEqual(len(rows), 2)
        # newest-first
        self.assertAlmostEqual(rows[0]["pulse"], -0.05)
        self.assertAlmostEqual(rows[1]["pulse"], 0.12)
        self.assertAlmostEqual(rows[1]["themes"]["geopolitics"], 0.3)


class EnrichTest(unittest.TestCase):
    def _agg(self):
        return {
            "pulse": 0.20, "bull_pct": 0.6, "bear_pct": 0.4, "confidence": 70,
            "n_items": 18, "n_sources": 5,
            "themes": {
                "geopolitics": {"score": 0.4, "n": 6},
                "fed_rates": {"score": -0.1, "n": 4},
                "other": {"score": 0.0, "n": 3},
            },
            "top": [
                {"title": "Iran strikes escalate", "source": "reuters.com",
                 "sentiment": -0.6, "theme": "geopolitics"},
                {"title": "Powell signals patience", "source": "wsj.com",
                 "sentiment": -0.2, "theme": "fed_rates"},
            ],
        }

    def test_cold_start_percentiles_none(self):
        ctx = C.enrich(self._agg(), db_path="/nonexistent/no.db")
        self.assertIsNone(ctx.pulse_pctile)
        self.assertEqual(ctx.n_history, 0)
        # themes exclude 'other', carry sector tags, sorted by item count
        names = [t.theme for t in ctx.themes]
        self.assertIn("geopolitics", names)
        self.assertNotIn("other", names)
        geo = next(t for t in ctx.themes if t.theme == "geopolitics")
        self.assertTrue(geo.sectors)
        self.assertEqual(geo.top_headline, "Iran strikes escalate")

    def test_percentiles_from_history(self):
        db = os.path.join(tempfile.mkdtemp(), "h.db")
        for p in (-0.3, -0.1, 0.0, 0.05, 0.1, 0.15):
            C.persist_reading(p, {"geopolitics": 0.0}, db)
        ctx = C.enrich(self._agg(), db_path=db)
        self.assertIsNotNone(ctx.pulse_pctile)   # >=5 samples
        self.assertEqual(ctx.pulse_pctile, 100.0)  # 0.20 above all history
        self.assertEqual(ctx.n_history, 6)


if __name__ == "__main__":
    unittest.main()
