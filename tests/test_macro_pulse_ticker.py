"""Tests for src/macro_pulse/ticker.py — sector focus + render, offline."""
from __future__ import annotations

import unittest

from src.macro_pulse import context as C
from src.macro_pulse import ticker as T


def _ctx():
    def tr(name, score, sectors, head):
        return C.ThemeRead(theme=name, score=score, n=5, pctile=None, z=None,
                           sectors=C.sectors_for(name), top_headline=head,
                           read=f"{name} read")
    return C.MacroContext(
        pulse=0.05, pulse_pctile=None, pulse_z=None, lean="NEUTRAL",
        confidence=70, n_items=40, n_sources=10, bull_pct=0.5,
        themes=[
            tr("earnings_tech", 0.29, "earnings_tech", "Chip earnings soar"),
            tr("geopolitics", -0.11, "geopolitics", "NATO warning"),
            tr("fed_rates", -0.02, "fed_rates", "Fed holds"),
            tr("energy", 0.09, "energy", "Oil steady"),
        ],
        event_active=False, event_name=None, event_date=None,
        next_events=[], n_history=0,
        headline="Tech optimism vs neutral macro.",
        narrative_source="ai")


class SectorThemeTest(unittest.TestCase):
    def test_technology_maps_to_earnings_tech(self):
        self.assertIn("earnings_tech", T.themes_for_sector("Technology"))

    def test_energy_maps_to_energy_and_geopolitics(self):
        s = T.themes_for_sector("Energy")
        self.assertIn("energy", s)
        self.assertIn("geopolitics", s)

    def test_unknown_sector_returns_broad_only(self):
        s = T.themes_for_sector("Nonsense Sector")
        self.assertEqual(set(s), set(T.BROAD_THEMES))

    def test_ticker_override_crypto(self):
        # COIN's yfinance sector is Financial Services, but it's crypto-driven
        s = T.themes_for_ticker("COIN", "Financial Services")
        self.assertIn("crypto", s)


class FocusTest(unittest.TestCase):
    def test_focus_keeps_sector_relevant_first(self):
        ctx = T.focus(_ctx(), "Technology")
        names = [t.theme for t in ctx.themes]
        # earnings_tech is sector-specific -> must lead
        self.assertEqual(names[0], "earnings_tech")
        # broad themes still present
        self.assertIn("fed_rates", names)

    def test_focus_drops_irrelevant_specific_theme(self):
        # Technology focus should not foreground 'energy' (a non-broad,
        # non-tech specific theme)
        ctx = T.focus(_ctx(), "Technology")
        names = [t.theme for t in ctx.themes]
        self.assertNotIn("energy", names)


class RenderTest(unittest.TestCase):
    def test_render_ticker_contains_symbol_sector_and_news(self):
        out = T.render_ticker(_ctx(), "AAPL", "Technology")
        self.assertIn("AAPL", out)
        self.assertIn("Technology", out.title())  # case-insensitive-ish
        self.assertIn("earnings_tech", out)
        self.assertIn("Chip earnings soar", out)   # relevant headline shown
        self.assertIn("risk", out.lower())         # honest-read footer

    def test_render_discovery_is_general(self):
        out = T.render_ticker(_ctx(), None, None)
        # general mode -> market-wide; no single-ticker header
        self.assertIn("MARKET", out.upper())


if __name__ == "__main__":
    unittest.main()
