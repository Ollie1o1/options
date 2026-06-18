"""Tests for src/macro_pulse/rank.py — AI mocked, offline."""
from __future__ import annotations

import json
import unittest

from src.macro_pulse import context as C
from src.macro_pulse import rank as R


def _ctx():
    def tr(name, score):
        return C.ThemeRead(theme=name, score=score, n=5, pctile=None, z=None,
                           sectors=C.sectors_for(name), top_headline=f"{name} hl",
                           read=f"{name} read")
    return C.MacroContext(
        pulse=0.05, pulse_pctile=None, pulse_z=None, lean="NEUTRAL",
        confidence=70, n_items=40, n_sources=10, bull_pct=0.5,
        themes=[tr("earnings_tech", 0.30), tr("fed_rates", -0.02),
                tr("geopolitics", -0.11), tr("energy", 0.10)],
        event_active=False, event_name=None, event_date=None,
        next_events=[], n_history=0, headline="h", narrative_source="ai")


_SECTORS = {"AAPL": "Technology", "JPM": "Financial Services",
            "XOM": "Energy"}


class _FakeScorer:
    def __init__(self, raw):
        self._raw = raw

    def safe_chat_complete(self, system, user, max_tokens=400):
        return self._raw


class DeterministicTest(unittest.TestCase):
    def test_tech_ranks_above_neutral_financial(self):
        rows = R.rank_tickers(["AAPL", "JPM"], _ctx(), sectors=_SECTORS,
                              scorer=None)
        by = {r.symbol: r for r in rows}
        self.assertGreater(by["AAPL"].score, by["JPM"].score)
        self.assertEqual(rows[0].symbol, "AAPL")        # sorted desc
        self.assertEqual(by["AAPL"].lean, "TAILWIND")
        self.assertTrue(by["AAPL"].reason)
        self.assertEqual(rows[0].narrative_source, "deterministic")

    def test_every_symbol_present(self):
        rows = R.rank_tickers(["AAPL", "JPM", "XOM"], _ctx(), sectors=_SECTORS)
        self.assertEqual({r.symbol for r in rows}, {"AAPL", "JPM", "XOM"})

    def test_reason_agrees_with_lean(self):
        # A net-headwind name must cite a headwind driver, not an offsetting
        # positive contributor.
        rows = R.rank_tickers(["XOM"], _ctx(), sectors=_SECTORS)
        xom = rows[0]
        if xom.lean == "HEADWIND":
            self.assertIn("headwind", xom.reason)
        elif xom.lean == "TAILWIND":
            self.assertIn("tailwind", xom.reason)


class AiTest(unittest.TestCase):
    def test_ai_order_and_reasons_used(self):
        raw = json.dumps({"ranking": [
            {"symbol": "JPM", "lean": "HEADWIND", "reason": "rate pressure"},
            {"symbol": "AAPL", "lean": "TAILWIND", "reason": "chip strength"},
        ]})
        rows = R.rank_tickers(["AAPL", "JPM"], _ctx(), sectors=_SECTORS,
                              scorer=_FakeScorer(raw))
        self.assertEqual(rows[0].symbol, "JPM")          # AI order honored
        self.assertEqual(rows[0].reason, "rate pressure")
        self.assertEqual(rows[0].narrative_source, "ai")

    def test_ai_failure_falls_back(self):
        rows = R.rank_tickers(["AAPL", "JPM"], _ctx(), sectors=_SECTORS,
                              scorer=_FakeScorer(None))
        self.assertEqual(rows[0].narrative_source, "deterministic")
        self.assertEqual(rows[0].symbol, "AAPL")


class RenderTest(unittest.TestCase):
    def test_render_table_has_symbols_and_tag(self):
        rows = R.rank_tickers(["AAPL", "JPM"], _ctx(), sectors=_SECTORS)
        out = R.render_ranking(rows)
        self.assertIn("AAPL", out)
        self.assertIn("JPM", out)
        self.assertIn("TAILWIND", out.upper())


if __name__ == "__main__":
    unittest.main()
