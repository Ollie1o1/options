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
        # finviz_tickers can itself return [] (e.g. a genuinely empty
        # screen) — confirm universe() passes that straight through.
        with mock.patch("src.squeeze.universe.finviz_tickers", return_value=[]):
            result = DSC.universe("BANKS")
        self.assertEqual(result, [])

    def test_raising_finviz_fetch_degrades_to_empty_list_not_raise(self):
        # finviz_tickers has NO try/except of its own around its network
        # calls (unlike get_squeeze_universe, which wraps a call to it) —
        # a live Finviz failure raises straight out of finviz_tickers.
        # universe() must catch that itself and degrade to [], never
        # propagate, so a live-network hiccup can't crash the launcher.
        with mock.patch("src.squeeze.universe.finviz_tickers",
                        side_effect=RuntimeError("connection reset")):
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


class TestDeepContext(unittest.TestCase):
    def test_insider_read_returns_none_without_cik(self):
        with mock.patch("src.insider.edgar.cik_for", return_value=None):
            result = DSC._insider_read("ZZZZ")
        self.assertIsNone(result)

    def test_insider_read_wires_cluster_score(self):
        fake_score = {"n_buyers": 2, "score": 0.85, "label": "CLUSTER BUY"}
        with mock.patch("src.insider.edgar.cik_for", return_value=12345), \
             mock.patch("src.insider.edgar.recent_form4",
                        return_value=[{"accession": "a", "document": "d", "filed": "2026-07-01"}]), \
             mock.patch("src.insider.edgar.fetch_form4_xml", return_value="<xml/>"), \
             mock.patch("src.insider.parse.parse_form4", return_value=[{"code": "P"}]), \
             mock.patch("src.insider.signal.cluster_score", return_value=fake_score):
            result = DSC._insider_read("MU")
        self.assertEqual(result, fake_score)

    def test_insider_read_degrades_on_exception(self):
        with mock.patch("src.insider.edgar.cik_for", side_effect=RuntimeError("boom")):
            result = DSC._insider_read("MU")
        self.assertIsNone(result)

    def test_earnings_read_none_without_key(self):
        # resolve_api_key returns None with no FINNHUB_API_KEY/config — the
        # real next_earnings_date already returns None in that case; confirm
        # this wrapper doesn't crash and passes that through.
        with mock.patch("src.earnings_provider.resolve_api_key", return_value=None), \
             mock.patch("src.earnings_provider.next_earnings_date", return_value=None):
            result = DSC._earnings_read("MU")
        self.assertIsNone(result)

    def test_earnings_read_converts_datetime_to_days(self):
        import datetime as dt
        future = dt.datetime.combine(dt.date.today() + dt.timedelta(days=12),
                                     dt.time())
        with mock.patch("src.earnings_provider.resolve_api_key", return_value="k"), \
             mock.patch("src.earnings_provider.next_earnings_date", return_value=future):
            result = DSC._earnings_read("MU")
        self.assertEqual(result, 12)

    def test_fundamentals_read_extracts_named_fields_only(self):
        fake_info = {
            "trailingPE": 18.2, "forwardPE": 15.1, "profitMargins": 0.22,
            "revenueGrowth": 0.11, "earningsGrowth": 0.08, "returnOnEquity": 0.31,
            "irrelevantField": "should not appear",
        }
        fake_ticker = mock.MagicMock()
        fake_ticker.info = fake_info
        with mock.patch("yfinance.Ticker", return_value=fake_ticker):
            result = DSC._fundamentals_read("MU")
        self.assertEqual(result, {
            "trailingPE": 18.2, "forwardPE": 15.1, "profitMargins": 0.22,
            "revenueGrowth": 0.11, "earningsGrowth": 0.08, "returnOnEquity": 0.31,
        })
        self.assertNotIn("irrelevantField", result)

    def test_fundamentals_read_degrades_on_exception(self):
        with mock.patch("yfinance.Ticker", side_effect=RuntimeError("timeout")):
            result = DSC._fundamentals_read("MU")
        self.assertIsNone(result)

    def test_deep_context_combines_all_three_independently(self):
        with mock.patch.object(DSC, "_insider_read", return_value={"score": 0.8}), \
             mock.patch.object(DSC, "_earnings_read", return_value=5), \
             mock.patch.object(DSC, "_fundamentals_read", return_value={"trailingPE": 20.0}):
            result = DSC.deep_context("MU")
        self.assertEqual(result.ticker, "MU")
        self.assertEqual(result.insider, {"score": 0.8})
        self.assertEqual(result.earnings_days, 5)
        self.assertEqual(result.fundamentals, {"trailingPE": 20.0})

    def test_deep_context_survives_one_source_failing(self):
        with mock.patch.object(DSC, "_insider_read", side_effect=RuntimeError("boom")), \
             mock.patch.object(DSC, "_earnings_read", return_value=5), \
             mock.patch.object(DSC, "_fundamentals_read", return_value={"trailingPE": 20.0}):
            result = DSC.deep_context("MU")
        self.assertIsNone(result.insider)
        self.assertEqual(result.earnings_days, 5)
        self.assertEqual(result.fundamentals, {"trailingPE": 20.0})


from src.longterm.discover import CandidateRead, DeepRead


def _candidate(ticker="MU", drawdown=-32.4, spot=760.0):
    return CandidateRead(
        ticker=ticker, spot=spot, drawdown_pct=drawdown,
        ma200_distance_pct=-8.5, momentum_12_1=-0.18,
        supports=[{"label": "200d MA", "level": 700.0, "pct": -0.079}],
        bounce={"trailing_return": -0.15, "lookback_days": 5,
               "by_horizon": {20: {"n": 41, "bounce_rate": 0.65,
                                   "median": 0.04, "p25": -0.02, "p75": 0.09}}},
        suggested_ladder=[],
    )


class TestInsightLine(unittest.TestCase):
    def test_includes_ticker_and_drawdown(self):
        line = DSC.insight_line(_candidate(), None)
        self.assertIn("MU", line)
        self.assertIn("32", line)  # drawdown magnitude appears

    def test_includes_nearest_support_label(self):
        line = DSC.insight_line(_candidate(), None)
        self.assertIn("200d MA", line)

    def test_includes_bounce_odds_with_sample_size(self):
        line = DSC.insight_line(_candidate(), None)
        self.assertIn("41", line)  # n= is present, not hidden

    def test_none_deep_read_omits_deep_clauses_without_crashing(self):
        line = DSC.insight_line(_candidate(), None)
        self.assertIsInstance(line, str)
        self.assertNotIn("None", line)  # no raw None leaking into the sentence

    def test_partial_deep_read_omits_only_missing_fields(self):
        deep = DeepRead(ticker="MU", insider={"n_buyers": 2, "buy_value": 340_000.0,
                                              "label": "CLUSTER BUY"},
                        earnings_days=None, fundamentals=None)
        line = DSC.insight_line(_candidate(), deep)
        self.assertIn("insider", line.lower())
        self.assertNotIn("None", line)

    def test_full_deep_read_includes_all_clauses(self):
        deep = DeepRead(
            ticker="MU",
            insider={"n_buyers": 2, "buy_value": 340_000.0, "label": "CLUSTER BUY"},
            earnings_days=12,
            fundamentals={"trailingPE": 18.0, "forwardPE": 15.0,
                         "profitMargins": 0.22, "revenueGrowth": 0.11,
                         "earningsGrowth": 0.08, "returnOnEquity": 0.31},
        )
        line = DSC.insight_line(_candidate(), deep)
        self.assertIn("12", line)     # earnings days
        self.assertIn("18", line)     # P/E
        self.assertIn("340", line)    # insider buy value (in thousands or raw)


class TestScan(unittest.TestCase):
    def test_ranks_by_drawdown_most_negative_first(self):
        from src.longterm.zones import Snapshot

        def snap(ticker, spot, high):
            return Snapshot(ticker=ticker, spot=spot, high_52w=high, low_52w=spot * 0.9,
                            ma200=spot, daily_sigma=0.02, closes=[spot] * 60)

        snaps = {
            "AAA": snap("AAA", 90.0, 100.0),   # -10%
            "BBB": snap("BBB", 60.0, 100.0),   # -40%
            "CCC": snap("CCC", 80.0, 100.0),   # -20%
        }
        with mock.patch.object(DSC, "universe", return_value=list(snaps)), \
             mock.patch("src.longterm.data.fetch_snapshots", return_value=snaps), \
             mock.patch.object(DSC, "deep_context",
                               side_effect=lambda t: DeepRead(ticker=t)):
            results = DSC.scan("TECH", universe_limit=10, board_limit=10, deep_limit=2)
        tickers_in_order = [c.ticker for c, _ in results]
        self.assertEqual(tickers_in_order, ["BBB", "CCC", "AAA"])

    def test_deep_read_only_for_deep_limit_candidates(self):
        from src.longterm.zones import Snapshot

        def snap(ticker, spot):
            return Snapshot(ticker=ticker, spot=spot, high_52w=100.0, low_52w=spot * 0.9,
                            ma200=spot, daily_sigma=0.02, closes=[spot] * 60)

        snaps = {t: snap(t, 100.0 - i) for i, t in enumerate(["A", "B", "C", "D"])}
        with mock.patch.object(DSC, "universe", return_value=list(snaps)), \
             mock.patch("src.longterm.data.fetch_snapshots", return_value=snaps), \
             mock.patch.object(DSC, "deep_context",
                               side_effect=lambda t: DeepRead(ticker=t)) as mocked_deep:
            results = DSC.scan("TECH", universe_limit=10, board_limit=10, deep_limit=2)
        deep_present = [deep is not None for _, deep in results]
        self.assertEqual(deep_present, [True, True, False, False])
        self.assertEqual(mocked_deep.call_count, 2)

    def test_missing_snapshot_for_a_universe_ticker_is_skipped_not_crashed(self):
        from src.longterm.zones import Snapshot
        snaps = {"A": Snapshot(ticker="A", spot=90.0, high_52w=100.0, low_52w=80.0,
                               ma200=90.0, daily_sigma=0.02, closes=[90.0] * 60)}
        with mock.patch.object(DSC, "universe", return_value=["A", "MISSING"]), \
             mock.patch("src.longterm.data.fetch_snapshots", return_value=snaps), \
             mock.patch.object(DSC, "deep_context",
                               side_effect=lambda t: DeepRead(ticker=t)):
            results = DSC.scan("TECH")
        self.assertEqual([c.ticker for c, _ in results], ["A"])


if __name__ == "__main__":
    unittest.main()
