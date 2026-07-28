"""The recent-action context must ride along with the ranking without changing
it. This is the guard on the validated IC: if attaching context ever alters a
score, direction, conviction or driver string, this test fails.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.outlook.test_recent_wiring -v
"""
from __future__ import annotations

import unittest

from src.outlook.backtest import _features
from src.outlook.engine import DEFAULT_OUTLOOK_CONFIG, rank_universe
from src.outlook.recent import recent_context


def _series(n, start, end):
    """Linear price path from `start` to `end` over n bars."""
    return [start + (end - start) * i / (n - 1) for i in range(n)]


class ScoreIsUnchangedTests(unittest.TestCase):
    def test_attaching_context_does_not_alter_ranking_fields(self):
        cols = {
            "SPY": _series(300, 100.0, 110.0),
            "AAA": _series(300, 100.0, 150.0),
            "BBB": _series(300, 100.0, 90.0),
        }
        bench = cols["SPY"]
        t = len(bench) - 1
        feats = {tk: _features(cols[tk], bench, t) for tk in ("AAA", "BBB")}
        baseline = rank_universe(feats, DEFAULT_OUTLOOK_CONFIG)

        enriched = [dict(r) for r in baseline]
        for r in enriched:
            r.update(recent_context(cols[r["ticker"]], bench, t))

        for base, rich in zip(baseline, enriched):
            for key in ("ticker", "score", "direction", "conviction", "drivers"):
                self.assertEqual(base[key], rich[key],
                                 f"{key} changed when context was attached")

    def test_context_keys_are_present_on_every_row(self):
        cols = {"SPY": _series(300, 100.0, 110.0), "AAA": _series(300, 100.0, 150.0)}
        bench = cols["SPY"]
        t = len(bench) - 1
        feats = {"AAA": _features(cols["AAA"], bench, t)}
        rows = rank_universe(feats, DEFAULT_OUTLOOK_CONFIG)
        for r in rows:
            r.update(recent_context(cols[r["ticker"]], bench, t))
            for key in ("ret_5d", "ret_21d", "excess_5d", "excess_21d", "lagging"):
                self.assertIn(key, r)


class ConfigTests(unittest.TestCase):
    def test_threshold_is_tunable_from_config(self):
        self.assertIn("recent_lag_threshold_pp", DEFAULT_OUTLOOK_CONFIG)

    def test_config_default_matches_the_module_default(self):
        from src.outlook.recent import DEFAULT_LAG_THRESHOLD_PP
        self.assertEqual(DEFAULT_OUTLOOK_CONFIG["recent_lag_threshold_pp"],
                         DEFAULT_LAG_THRESHOLD_PP)


class LiveOutlookShapeTests(unittest.TestCase):
    """live_outlook must attach context to the rows it returns. Exercised with
    injected price columns so the test stays offline."""

    def test_rows_carry_context(self):
        from src.outlook import backtest as bt

        # SMH rises all year, then gives back 13% in the final month — the
        # 2026-07-28 shape. A gently-sloping line would not clear -5.2pp over
        # 21 bars, so the drop has to be concentrated at the end.
        smh = _series(279, 100.0, 150.0)
        smh += [150.0 * (1.0 - 0.13 * i / 21) for i in range(1, 22)]
        cols = {"SPY": _series(300, 100.0, 110.0),
                "XLK": _series(300, 100.0, 150.0),
                "SMH": smh}
        orig = bt._aligned_closes
        bt._aligned_closes = lambda tickers, period="max": ([], cols)
        try:
            rows = bt.live_outlook(DEFAULT_OUTLOOK_CONFIG, universe=["XLK", "SMH"])
        finally:
            bt._aligned_closes = orig

        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertIn("lagging", r)
            self.assertIn("excess_21d", r)
        smh = next(r for r in rows if r["ticker"] == "SMH")
        self.assertTrue(smh["lagging"], "SMH underperforms SPY and should flag")
