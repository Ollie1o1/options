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
        self.calls = 0

    def safe_chat_complete(self, system, user, max_tokens=400):
        self.calls += 1
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


class SynthNoAiTest(unittest.TestCase):
    """The macro panel is shown BEFORE the user answers the AI-ranking prompt,
    so building it must not fire an AI request. `use_ai=False` keeps the panel
    fully functional on the deterministic template with zero network calls."""

    def test_use_ai_false_makes_no_ai_call_even_with_a_scorer(self):
        scorer = _FakeScorer(json.dumps({"headline": "should never be used"}))
        out = S.narrate(_ctx(), scorer=scorer, use_ai=False)
        self.assertEqual(scorer.calls, 0)
        self.assertEqual(out.narrative_source, "deterministic")
        self.assertTrue(out.headline)

    def test_use_ai_false_does_not_construct_a_default_scorer(self):
        # scorer=None must NOT lazily build an AIScorer (that is the network hop
        # the user saw fire before the prompt).
        import src.ai_scorer as ai_scorer

        built = []
        orig = ai_scorer.AIScorer.__init__

        def _spy(self, *a, **k):
            built.append(1)
            return orig(self, *a, **k)

        ai_scorer.AIScorer.__init__ = _spy
        try:
            out = S.narrate(_ctx(), scorer=None, use_ai=False)
        finally:
            ai_scorer.AIScorer.__init__ = orig
        self.assertEqual(built, [])
        self.assertEqual(out.narrative_source, "deterministic")

    def test_use_ai_true_is_still_the_default(self):
        scorer = _FakeScorer(json.dumps({"headline": "AI read", "themes": [],
                                         "what_would_flip": "x"}))
        out = S.narrate(_ctx(), scorer=scorer)
        self.assertEqual(scorer.calls, 1)
        self.assertEqual(out.narrative_source, "ai")


class BuildContextNoAiTest(unittest.TestCase):
    def test_build_context_use_ai_false_never_calls_ai(self):
        from src.macro_pulse import orchestrator as O

        scorer = _FakeScorer(json.dumps({"headline": "nope"}))
        # Empty news → deterministic path regardless, but assert no AI attempt.
        O.build_context(fetch_fn=lambda: [], scorer=scorer, use_ai=False,
                        persist=False, cache_db=":memory:", db_path=":memory:")
        self.assertEqual(scorer.calls, 0)


if __name__ == "__main__":
    unittest.main()
