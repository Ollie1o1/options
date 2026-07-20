"""Tests for the long-term discovery scan (src/longterm/discover.py).

Pure-function tests only in this file's first section (universe sourcing,
context math, ladder suggestion, narrative rendering) — network-touching
functions (universe's actual Finviz call inside finviz_tickers, deep_context,
scan) are thin wrappers tested by inspection/mocking only where noted, never
against the live network, matching the convention set by
tests/longterm/test_data.py for src/longterm/data.py's fetch_snapshots.
"""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import discover as DSC


class TestSectorFilters(unittest.TestCase):
    def test_every_filter_value_is_nonempty_and_has_quality_gate(self):
        for keyword, f_params in DSC.SECTOR_FILTERS.items():
            self.assertTrue(f_params, f"{keyword} has an empty filter string")
            self.assertIn("cap_midover", f_params, f"{keyword} missing quality gate")

    def test_keywords_are_uppercase(self):
        for keyword in DSC.SECTOR_FILTERS:
            self.assertEqual(keyword, keyword.upper())


class TestUniverse(unittest.TestCase):
    def test_unknown_keyword_raises_with_valid_list(self):
        with self.assertRaises(ValueError) as ctx:
            DSC.universe("NOT_A_SECTOR")
        msg = str(ctx.exception)
        self.assertIn("SEMICONDUCTORS", msg)  # a real keyword should be listed

    def test_keyword_is_case_insensitive(self):
        with mock.patch("src.squeeze.universe.finviz_tickers",
                        return_value=["MU", "AMD"]) as mocked:
            result = DSC.universe("semiconductors", limit=10)
        self.assertEqual(result, ["MU", "AMD"])
        mocked.assert_called_once()
        called_f_params = mocked.call_args.args[0]
        self.assertEqual(called_f_params, DSC.SECTOR_FILTERS["SEMICONDUCTORS"])

    def test_passes_limit_through(self):
        with mock.patch("src.squeeze.universe.finviz_tickers",
                        return_value=[]) as mocked:
            DSC.universe("TECH", limit=7)
        self.assertEqual(mocked.call_args.kwargs.get("limit")
                         or mocked.call_args.args[2], 7)

    def test_empty_result_on_fetch_failure_does_not_raise(self):
        # finviz_tickers itself never raises (degrades to [] internally) —
        # confirm universe() doesn't add its own crash path on top.
        with mock.patch("src.squeeze.universe.finviz_tickers", return_value=[]):
            result = DSC.universe("BANKS")
        self.assertEqual(result, [])


from src.longterm.zones import Snapshot


def _rising_closes(n=300, start=100.0, daily_return=0.003):
    """Steadily rising series — cheap fixture for shape/wiring tests where
    the exact trajectory doesn't matter, only that enough history exists."""
    closes = [start]
    for _ in range(n - 1):
        closes.append(closes[-1] * (1.0 + daily_return))
    return closes


def _drawdown_snapshot():
    """A name 260 trading days deep with a clean 30% drop from its high in
    the last 60 days, then flat — gives fast_context() real supports/bounce
    data to wire through, and a known, hand-checkable drawdown_pct."""
    closes = _rising_closes(n=240, start=100.0, daily_return=0.004)  # up to ~256
    peak = closes[-1]
    # Sharp 30% drop over 10 days, then flat for 50 days (recent swing low
    # + enough post-drop history for bounce_stats to have samples).
    drop_step = (0.70 ** (1 / 10))
    for _ in range(10):
        closes.append(closes[-1] * drop_step)
    flat = closes[-1]
    closes.extend([flat] * 50)
    return Snapshot(ticker="TST", spot=closes[-1], high_52w=peak,
                    low_52w=min(closes), ma200=sum(closes[-200:]) / 200,
                    daily_sigma=0.02, closes=closes)


class TestFastContext(unittest.TestCase):
    def test_drawdown_matches_snapshot_high(self):
        snap = _drawdown_snapshot()
        c = DSC.fast_context(snap)
        expected = (snap.spot / snap.high_52w - 1.0) * 100
        self.assertAlmostEqual(c.drawdown_pct, expected, places=6)
        self.assertLess(c.drawdown_pct, 0)  # it's a drawdown, must be negative

    def test_ma200_distance_matches_snapshot(self):
        snap = _drawdown_snapshot()
        c = DSC.fast_context(snap)
        expected = (snap.spot / snap.ma200 - 1.0) * 100
        self.assertAlmostEqual(c.ma200_distance_pct, expected, places=6)

    def test_ma200_distance_none_when_snapshot_lacks_ma200(self):
        snap = _drawdown_snapshot()
        snap.ma200 = None
        c = DSC.fast_context(snap)
        self.assertIsNone(c.ma200_distance_pct)

    def test_momentum_present_with_enough_history(self):
        snap = _drawdown_snapshot()  # 300 closes, > 252 needed for mom_12_1
        c = DSC.fast_context(snap)
        self.assertIsNotNone(c.momentum_12_1)

    def test_momentum_none_with_short_history(self):
        closes = _rising_closes(n=100)
        snap = Snapshot(ticker="TST", spot=closes[-1], high_52w=max(closes),
                        low_52w=min(closes), ma200=None, daily_sigma=0.02,
                        closes=closes)
        c = DSC.fast_context(snap)
        self.assertIsNone(c.momentum_12_1)

    def test_supports_and_bounce_are_wired_through(self):
        snap = _drawdown_snapshot()
        c = DSC.fast_context(snap)
        # levels.support_resistance_levels always returns a 50d MA support
        # entry for spot below it — this fixture's post-drop spot is below
        # its own 50d MA average of the flat tail, so at least one support.
        self.assertIsInstance(c.supports, list)
        self.assertIsInstance(c.bounce, dict)
        self.assertIn("by_horizon", c.bounce)

    def test_suggested_ladder_first_tranche_is_spot(self):
        snap = _drawdown_snapshot()
        c = DSC.fast_context(snap)
        self.assertAlmostEqual(c.suggested_ladder[0].level, snap.spot, places=2)

    def test_suggested_ladder_equal_weights_sum_to_one(self):
        snap = _drawdown_snapshot()
        c = DSC.fast_context(snap)
        total_weight = sum(t.weight for t in c.suggested_ladder)
        self.assertAlmostEqual(total_weight, 1.0, places=6)


class TestSuggestLadder(unittest.TestCase):
    def test_uses_real_supports_when_available(self):
        supports = [{"label": "50d MA", "level": 90.0, "pct": -0.10},
                    {"label": "200d MA", "level": 80.0, "pct": -0.20}]
        ladder = DSC.suggest_ladder(100.0, supports)
        levels = [t.level for t in ladder]
        self.assertEqual(levels, [100.0, 90.0, 80.0])

    def test_uses_at_most_two_supports_below_spot(self):
        # Three supports available — ladder caps at 3 tranches total
        # (spot + 2 supports), matching the design's "tranches 2-3" rule.
        supports = [{"label": "50d MA", "level": 95.0, "pct": -0.05},
                    {"label": "200d MA", "level": 90.0, "pct": -0.10},
                    {"label": "60d low", "level": 80.0, "pct": -0.20}]
        ladder = DSC.suggest_ladder(100.0, supports)
        self.assertEqual(len(ladder), 3)
        self.assertEqual([t.level for t in ladder], [100.0, 95.0, 90.0])

    def test_falls_back_to_percentage_steps_with_no_supports(self):
        ladder = DSC.suggest_ladder(100.0, [])
        levels = [t.level for t in ladder]
        self.assertEqual(levels, [100.0, 90.0, 80.0])  # spot, -10%, -20%

    def test_ignores_resistance_levels_above_spot(self):
        # support_resistance_levels never returns a "support" above spot by
        # construction, but guard suggest_ladder against a malformed input
        # anyway rather than silently producing an inverted ladder.
        supports = [{"label": "bad data", "level": 110.0, "pct": 0.10}]
        ladder = DSC.suggest_ladder(100.0, supports)
        levels = [t.level for t in ladder]
        self.assertEqual(levels, [100.0, 90.0, 80.0])  # fell back cleanly

    def test_equal_weights(self):
        ladder = DSC.suggest_ladder(100.0, [])
        for t in ladder:
            self.assertAlmostEqual(t.weight, 1.0 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
