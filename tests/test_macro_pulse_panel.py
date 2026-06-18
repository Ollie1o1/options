"""Tests for src/macro_pulse/panel.py — pure render, offline."""
from __future__ import annotations

import unittest

from src.macro_pulse import context as C
from src.macro_pulse import panel as P


def _ctx(empty=False, cold=False):
    if empty:
        return C.MacroContext(
            pulse=0.0, pulse_pctile=None, pulse_z=None, lean="NEUTRAL",
            confidence=0, n_items=0, n_sources=0, bull_pct=0.5, themes=[],
            event_active=False, event_name=None, event_date=None,
            next_events=[], n_history=0, headline="No strong themes.",
            narrative_source="deterministic")
    tr = C.ThemeRead(theme="geopolitics", score=0.4, n=6,
                     pctile=None if cold else 92.0, z=None if cold else 1.8,
                     sectors=C.sectors_for("geopolitics"),
                     top_headline="Iran strikes escalate",
                     read="Escalation lifts defense and energy.")
    return C.MacroContext(
        pulse=0.2, pulse_pctile=None if cold else 95.0,
        pulse_z=None if cold else 1.9, lean="BULLISH-LEAN", confidence=70,
        n_items=18, n_sources=5, bull_pct=0.6, themes=[tr],
        event_active=True, event_name="FOMC", event_date="2026-06-17",
        next_events=[{"name": "CPI", "date": "2026-07-14"}], n_history=12,
        headline="Geopolitics dominates; risk-off into FOMC.",
        what_would_flip="An Iran ceasefire headline.", narrative_source="ai")


class PanelTest(unittest.TestCase):
    def test_full_render_contains_key_fields(self):
        out = P.render(_ctx())
        self.assertIn("MACRO PULSE", out.upper())
        self.assertIn("+0.20", out)
        self.assertIn("95", out)              # pulse percentile
        self.assertIn("geopolitics", out)
        self.assertIn("ITA", out)             # sector tag
        self.assertIn("FOMC", out)            # event flag
        self.assertIn("Geopolitics dominates", out)  # AI headline
        self.assertIn("ceasefire", out)       # what-would-flip
        self.assertIn("risk", out.lower())    # honest-read footer
        self.assertIn("ai", out.lower())      # narrative source tag

    def test_cold_start_shows_building_history(self):
        out = P.render(_ctx(cold=True))
        self.assertIn("building", out.lower())

    def test_empty_render_does_not_crash(self):
        out = P.render(_ctx(empty=True))
        self.assertIn("MACRO PULSE", out.upper())
        self.assertIn("No strong themes", out)


if __name__ == "__main__":
    unittest.main()
