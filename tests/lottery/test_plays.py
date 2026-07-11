"""Tests for the lottery play-archetype classifier."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.lottery import plays as P


class TestClassifyPlay(unittest.TestCase):
    def test_catalyst_when_event_and_cheap_iv(self):
        row = {"type": "call", "earnings_dte": 6, "iv_rank_score": 0.30}
        self.assertEqual(P.classify_play(row), P.CATALYST)

    def test_event_but_rich_iv_is_not_catalyst(self):
        # rich IV into an event is a crush trap, not a CATALYST play
        row = {"type": "call", "earnings_dte": 6, "iv_rank_score": 0.85, "rsi_14": 50}
        self.assertNotEqual(P.classify_play(row), P.CATALYST)

    def test_squeeze(self):
        row = {"type": "call", "short_interest": 0.28, "Unusual_Whale": 1.0, "rsi_14": 50}
        self.assertEqual(P.classify_play(row), P.SQUEEZE)

    def test_breakout(self):
        row = {"type": "call", "adx_14": 30, "underlying": 105, "sma_50": 100,
               "high_20d": 104, "rsi_14": 62}
        self.assertEqual(P.classify_play(row), P.BREAKOUT)

    def test_bounce(self):
        row = {"type": "call", "rsi_14": 28, "underlying": 90, "sma_20": 100, "rvol": 1.6}
        self.assertEqual(P.classify_play(row), P.BOUNCE)

    def test_crash(self):
        row = {"type": "put", "rsi_14": 72, "underlying": 95, "sma_20": 100, "rvol": 1.5}
        self.assertEqual(P.classify_play(row), P.CRASH)

    def test_put_on_bouncing_name_is_not_bounce(self):
        # BOUNCE is a call-only archetype; a put here must not read as BOUNCE
        row = {"type": "put", "rsi_14": 28, "underlying": 90, "sma_20": 100, "rvol": 1.6}
        self.assertNotEqual(P.classify_play(row), P.BOUNCE)

    def test_default_longshot(self):
        row = {"type": "call", "rsi_14": 50, "adx_14": 15}
        self.assertEqual(P.classify_play(row), P.LONGSHOT)

    def test_safe_on_empty_row(self):
        self.assertEqual(P.classify_play({}), P.LONGSHOT)


if __name__ == "__main__":
    unittest.main()
