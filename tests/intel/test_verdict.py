"""Tests for src/intel/verdict.py — pure, offline."""
from __future__ import annotations

import unittest

from src.intel.signals import Signal
from src.intel import verdict as V


def _sig(name, value, directional=True):
    return Signal(name, value, "", "", directional=directional)


def _rel(**weights):
    base = {n: {"weight": 0.0, "tag": "weak — context only"}
            for n in ("trend", "momentum", "bounce", "rsi", "support", "news", "analyst")}
    for k, w in weights.items():
        base[k] = {"weight": w, "tag": "reliable"}
    return base


class VerdictTests(unittest.TestCase):
    def test_bullish_signals_buy(self):
        sigs = {"trend": _sig("trend", 0.8), "momentum": _sig("momentum", 0.6),
                "earnings": _sig("earnings", 0.0, directional=False)}
        v = V.decide(sigs, _rel(trend=1.0, momentum=1.0))
        self.assertEqual(v.call, "BUY")
        self.assertGreater(v.composite, V.BUY_AT)

    def test_bearish_signals_avoid(self):
        sigs = {"trend": _sig("trend", -0.9), "momentum": _sig("momentum", -0.8)}
        v = V.decide(sigs, _rel(trend=1.0, momentum=1.0))
        self.assertEqual(v.call, "AVOID")

    def test_zero_weight_signals_ignored(self):
        # Strong bullish reading but the signals earned no weight → no edge.
        sigs = {"trend": _sig("trend", 0.9), "momentum": _sig("momentum", 0.9)}
        v = V.decide(sigs, _rel())  # all zero weight
        self.assertEqual(v.composite, 0.0)
        self.assertEqual(v.call, "NEUTRAL")

    def test_earnings_caps_buy_to_wait(self):
        sigs = {"trend": _sig("trend", 0.9), "momentum": _sig("momentum", 0.7),
                "earnings": _sig("earnings", -1.0, directional=False)}
        v = V.decide(sigs, _rel(trend=1.0, momentum=1.0))
        self.assertEqual(v.call, "WAIT")
        self.assertIn("earnings", v.note)
        self.assertNotEqual(v.confidence, "high")

    def test_drivers_ranked_weighted_first(self):
        sigs = {"trend": _sig("trend", 0.5), "news": _sig("news", -0.9)}
        v = V.decide(sigs, _rel(trend=1.0))  # news zero weight
        self.assertEqual(v.drivers[0].name, "trend")     # weighted ranks first
        self.assertEqual(v.drivers[0].glyph, "+")
        news_driver = [d for d in v.drivers if d.name == "news"][0]
        self.assertEqual(news_driver.glyph, "~")          # context only

    def test_confidence_low_when_no_weight(self):
        sigs = {"trend": _sig("trend", 0.9)}
        v = V.decide(sigs, _rel())
        self.assertEqual(v.confidence, "low")


if __name__ == "__main__":
    unittest.main()
