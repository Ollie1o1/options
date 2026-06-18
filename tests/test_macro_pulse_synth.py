"""Tests for src/macro_pulse/synth.py — AI mocked, offline."""
from __future__ import annotations

import json
import unittest

from src.macro_pulse import context as C
from src.macro_pulse import synth as S


def _ctx():
    tr = C.ThemeRead(theme="geopolitics", score=0.4, n=6, pctile=92.0, z=1.8,
                     sectors=C.sectors_for("geopolitics"),
                     top_headline="Iran strikes escalate")
    return C.MacroContext(
        pulse=0.2, pulse_pctile=95.0, pulse_z=1.9, lean="BULLISH-LEAN",
        confidence=70, n_items=18, n_sources=5, bull_pct=0.6, themes=[tr],
        event_active=True, event_name="FOMC", event_date="2026-06-17",
        next_events=[], n_history=10)


class _FakeScorer:
    def __init__(self, raw):
        self._raw = raw

    def safe_chat_complete(self, system, user, max_tokens=400):
        return self._raw


class SynthTest(unittest.TestCase):
    def test_ai_success_fills_narrative(self):
        raw = json.dumps({
            "headline": "Geopolitics dominates; risk-off into FOMC.",
            "themes": [{"theme": "geopolitics",
                        "read": "Escalation lifts defense and energy.",
                        "sectors": ["ITA", "XLE"]}],
            "what_would_flip": "An Iran ceasefire headline.",
        })
        out = S.narrate(_ctx(), scorer=_FakeScorer(raw))
        self.assertEqual(out.narrative_source, "ai")
        self.assertIn("FOMC", out.headline)
        self.assertTrue(out.themes[0].read)
        self.assertTrue(out.what_would_flip)

    def test_ai_failure_falls_back_deterministic(self):
        out = S.narrate(_ctx(), scorer=_FakeScorer(None))  # AI returns None
        self.assertEqual(out.narrative_source, "deterministic")
        self.assertTrue(out.headline)          # template still fills it
        self.assertTrue(out.themes[0].read)

    def test_malformed_ai_falls_back(self):
        out = S.narrate(_ctx(), scorer=_FakeScorer("not json at all"))
        self.assertEqual(out.narrative_source, "deterministic")
        self.assertTrue(out.headline)


if __name__ == "__main__":
    unittest.main()
