"""Both HTML surfaces must show the same recent-action context as the terminal,
and must tolerate a pre-upgrade cache that lacks the fields.

The two renderers differ in entry point and fixture shape:
  morning  — _zone_signals(p),  p is the panel dict directly
  research — _tab_macro(data),  reads via _panel(data, "signals")

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.outlook.test_recent_html -v
"""
from __future__ import annotations

import unittest


def _rows(lagging=True):
    return [{"ticker": "SMH", "direction": "BULLISH", "conviction": 99,
             "lagging": lagging, "ret_21d": -0.130, "excess_21d": -14.8}]


class ResearchDeskTests(unittest.TestCase):
    def _data(self, outlook):
        return {"panels": {"signals": {"uoa": [], "insider": [],
                                       "outlook": outlook}}}

    def test_lagging_row_shows_the_recent_move(self):
        from src.research.render import _tab_macro
        html = _tab_macro(self._data({"top": _rows(), "bottom": []}))
        self.assertIn("14.8pp", html.replace("−", "-"))

    def test_unflagged_row_gets_no_note(self):
        from src.research.render import _tab_macro
        html = _tab_macro(self._data({"top": _rows(lagging=False), "bottom": []}))
        self.assertNotIn("pp vs SPY", html.replace("−", "-"))

    def test_pre_upgrade_rows_render_without_error(self):
        from src.research.render import _tab_macro
        html = _tab_macro(self._data(
            {"top": [{"ticker": "SMH", "direction": "BULLISH"}], "bottom": []}))
        self.assertIn("SMH", html)


class MorningBriefingTests(unittest.TestCase):
    def test_lagging_row_shows_the_recent_move(self):
        from src.morning.render import _zone_signals
        html = _zone_signals({"outlook": {"top": _rows(), "bottom": [],
                                          "as_of": "2026-07-28 12:00 UTC"}})
        self.assertIn("14.8pp", html.replace("−", "-"))

    def test_unflagged_row_gets_no_note(self):
        from src.morning.render import _zone_signals
        html = _zone_signals({"outlook": {"top": _rows(lagging=False),
                                          "bottom": [], "as_of": None}})
        self.assertNotIn("pp vs SPY", html.replace("−", "-"))

    def test_pre_upgrade_rows_render_without_error(self):
        from src.morning.render import _zone_signals
        html = _zone_signals({"outlook": {"top": [{"ticker": "SMH"}],
                                          "bottom": [], "as_of": None}})
        self.assertIn("SMH", html)
