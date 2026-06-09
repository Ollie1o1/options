"""Tests for src/intel/reliability.py — pure math + grading + cache, offline."""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

from src.intel import reliability as R


class SpearmanTests(unittest.TestCase):
    def test_perfect_positive(self):
        self.assertAlmostEqual(R._spearman([1, 2, 3, 4], [10, 20, 30, 40]), 1.0)

    def test_perfect_negative(self):
        self.assertAlmostEqual(R._spearman([1, 2, 3, 4], [40, 30, 20, 10]), -1.0)

    def test_degenerate_none(self):
        self.assertIsNone(R._spearman([1, 1, 1], [1, 2, 3]))


class SignalSkillTests(unittest.TestCase):
    def test_predictive_signal_high_hit_rate(self):
        # signal sign always matches forward sign
        sig = [1, 1, -1, -1, 1, -1]
        fwd = [0.02, 0.01, -0.03, -0.02, 0.04, -0.01]
        sk = R.signal_skill(sig, fwd)
        self.assertEqual(sk["hit_rate"], 1.0)
        self.assertGreater(sk["ic"], 0.5)

    def test_flat_signals_excluded_from_hitrate(self):
        sig = [0, 0, 1, -1]
        fwd = [0.5, -0.5, 0.1, -0.1]
        sk = R.signal_skill(sig, fwd)
        self.assertEqual(sk["directional_n"], 2)  # the two zeros ignored

    def test_empty(self):
        sk = R.signal_skill([], [])
        self.assertEqual(sk["n"], 0)


class GradeTests(unittest.TestCase):
    def test_strong_signal_reliable(self):
        g = R._grade({"ic": 0.08, "hit_rate": 0.6, "n": 500, "directional_n": 500})
        self.assertEqual(g["tag"], "reliable")
        self.assertGreater(g["weight"], 0)

    def test_no_edge_zero_weight(self):
        g = R._grade({"ic": 0.0, "hit_rate": 0.5, "n": 500, "directional_n": 500})
        self.assertEqual(g["weight"], 0.0)
        self.assertIn("context only", g["tag"])

    def test_thin_sample_capped(self):
        g = R._grade({"ic": 0.09, "hit_rate": 0.6, "n": 50, "directional_n": 50})
        self.assertLessEqual(g["weight"], 0.5)
        self.assertTrue(g["tag"].startswith("ok"))


class ComputeReliabilityTests(unittest.TestCase):
    def test_trending_series_trend_earns_weight(self):
        # Noisy but persistent uptrend → trend signal should show positive IC.
        up = [100.0 + i * 0.4 + 4.0 * math.sin(i / 5.0) for i in range(600)]
        down = [400.0 - i * 0.3 + 4.0 * math.sin(i / 6.0) for i in range(600)]
        rel = R.compute_reliability({"UP": up, "DOWN": down}, step=3)
        self.assertIn("trend", rel)
        self.assertIsNotNone(rel["trend"]["ic"])
        # context-only signals stay zero-weight
        self.assertEqual(rel["news"]["weight"], 0.0)

    def test_context_only_always_zero(self):
        up = [100.0 + i * 0.4 for i in range(400)]
        rel = R.compute_reliability({"UP": up})
        for nm in ("news", "analyst"):
            self.assertEqual(rel[nm]["weight"], 0.0)


class CacheTests(unittest.TestCase):
    def test_uses_fresh_cache(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rel.json")
            payload = {"_meta": {"computed_at": 9e18, "forward_days": 10},
                       "trend": {"weight": 0.7, "tag": "reliable"}}
            with open(path, "w") as f:
                json.dump(payload, f)
            called = {"n": 0}

            def _fetch():
                called["n"] += 1
                return {}
            out = R.load_or_compute_reliability(cache_path=path, fetch=_fetch)
            self.assertEqual(out["trend"]["weight"], 0.7)
            self.assertEqual(called["n"], 0)  # cache hit, no fetch

    def test_neutral_fallback_when_no_data(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "rel.json")
            out = R.load_or_compute_reliability(
                cache_path=path, force=True, fetch=lambda: {})
            self.assertEqual(out["trend"]["weight"], 0.0)


if __name__ == "__main__":
    unittest.main()
