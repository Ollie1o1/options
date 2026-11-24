"""Discovery explanations must be reproducible observations, not new signals."""
import copy
import json
import unittest
from unittest.mock import patch

import pandas as pd

from src import discovery as D


def history(values, start="2026-10-01"):
    return pd.DataFrame({"Close": values}, index=pd.bdate_range(start, periods=len(values)))


def row(**over):
    return dict({"symbol": "TEST", "type": "call", "strategy_name": "Long Call",
                 "ret_5d": .04, "impliedVolatility": .4, "hv_30d": .3,
                 "quote_source": "yfinance", "quote_as_of": "2026-10-08T20:00:00Z",
                 "quote_freshness": "delayed", "expiration": "2026-12-18",
                 "ev_per_contract": 10}, **over)


def context():
    return {"mode": "Discovery scan", "as_of": "2026-10-08", "vix": 20,
            "market_observations": {"spy": D.history_observations(history([100, 99, 98, 97, 96, 95])),
                                    "vix": D.history_observations(history([15, 16, 17, 18, 19, 20]))},
            "ticker_contexts": {"TEST": {"earnings_date": "2026-11-01"}}}


class HistoryTests(unittest.TestCase):
    def test_exact_five_session_endpoints(self):
        out = D.history_observations(history([10, 11, 12, 13, 14, 15]))
        self.assertAlmostEqual(out["returns"]["5"]["value"], .5)
        self.assertEqual(out["returns"]["5"]["start"], "2026-10-01")
        self.assertEqual(out["as_of"], "2026-10-08")

    def test_missing_candle_is_not_dropped_to_manufacture_a_return(self):
        out = D.history_observations(history([10, 11, float("nan"), 13, 14, 15]))
        self.assertNotIn("5", out["returns"])
        self.assertIn("1", out["returns"])

    def test_short_history_has_no_five_session_claim(self):
        self.assertNotIn("5", D.history_observations(history([10, 11, 12, 13, 14]))["returns"])

    def test_bad_endpoint_and_unordered_history_are_refused(self):
        for h in (history([10, float("inf")]), history([10, 0]), history([10, 11]).iloc[::-1]):
            self.assertEqual(D.history_observations(h), {})

    def test_duplicate_dates_are_refused(self):
        h = history([10, 11])
        h.index = [h.index[0], h.index[0]]
        self.assertEqual(D.history_observations(h), {})

    def test_sector_uses_matching_dates_and_percentage_point_difference(self):
        stock = history([1, 100, 102, 104, 106, 108])
        sector = history([200, 202, 204, 206, 208], start="2026-10-02")
        aligned = D.relative_observations(stock, sector)
        self.assertAlmostEqual(aligned["ticker_return"], .08)
        self.assertAlmostEqual(aligned["sector_return"], .04)
        self.assertEqual(aligned["sessions"], 4)
        ctx = context()
        ctx["ticker_contexts"]["TEST"]["sector_perf"] = {"sector_etf": "XLK", "aligned_returns": aligned}
        obs = next(o for o in D.build(row(), ctx)["observations"] if o["id"] == "sector_relative")
        self.assertIn("+4.0 percentage points", obs["fact"])
        self.assertEqual(obs["horizon"], "2026-10-02 to 2026-10-08")

    def test_invalid_sector_history_does_not_become_neutral(self):
        self.assertEqual(D.relative_observations(history([1, float("nan"), 3]), history([1, 2, 3])), {})


class EvidenceTests(unittest.TestCase):
    def test_disagreement_is_explained_without_voting(self):
        brief = D.build(row(), context())
        self.assertIn("stock_momentum", brief["supporting"])
        self.assertIn("market_momentum", brief["conflicting"])
        self.assertIn("disagree", brief["summary"])
        self.assertNotIn("confidence", brief)
        self.assertNotIn("score", brief)

    def test_vix_is_context_even_when_it_rises(self):
        brief = D.build(row(), context())
        vix = next(o for o in brief["observations"] if o["id"] == "vix")
        self.assertEqual(vix["alignment"], "context")
        self.assertIn("5-session change +33.3%", vix["fact"])
        self.assertIn("not an up/down forecast", vix["implication"])

    def test_missing_vix_history_does_not_invent_trend(self):
        brief = D.build(row(), {"vix": 20})
        self.assertTrue(any("5-session VIX trend" in x for x in brief["unknowns"]))
        self.assertNotIn("change", next(o for o in brief["observations"] if o["id"] == "vix")["fact"])

    def test_dated_price_history_wins_over_row_momentum(self):
        ctx = context()
        ctx["ticker_contexts"]["TEST"]["price_observations"] = D.history_observations(history([10, 9, 8, 7, 6, 5]))
        brief = D.build(row(ret_5d=.99), ctx)
        stock = next(o for o in brief["observations"] if o["id"] == "stock_momentum")
        self.assertIn("-50.0%", stock["fact"])
        self.assertEqual(stock["as_of"], "2026-10-08")

    def test_short_put_exposure_is_not_mistaken_for_long_put(self):
        brief = D.build(row(type="put", strategy_name="short_put"), context())
        self.assertIn("stock_momentum", brief["supporting"])
        fallback = D.build(row(type="put", strategy_name=None), dict(context(), mode="Premium Selling"))
        self.assertIn("stock_momentum", fallback["supporting"])

    def test_long_put_alignment_reverses(self):
        brief = D.build(row(type="put", strategy_name="long_put"), context())
        self.assertIn("stock_momentum", brief["conflicting"])
        self.assertIn("market_momentum", brief["supporting"])

    def test_legacy_mismatched_sector_fields_are_not_compared(self):
        ctx = context()
        ctx["ticker_contexts"]["TEST"]["sector_perf"] = {"ticker_return": .2, "sector_return": -.2}
        brief = D.build(row(), ctx)
        self.assertFalse(any(o["id"] == "sector_relative" for o in brief["observations"]))
        self.assertTrue(any("matching window" in x for x in brief["unknowns"]))

    def test_nan_and_pandas_missing_are_json_safe_and_not_neutral_facts(self):
        brief = D.build(row(ret_5d=float("nan"), impliedVolatility=float("inf"), hv_30d=pd.NA,
                            strategy_name=pd.NA, quote_source=pd.NA, quote_as_of=pd.NaT,
                            quote_freshness=pd.NA, ev_per_contract=float("nan")))
        json.dumps(brief, allow_nan=False)
        self.assertEqual(brief["observations"], [])
        self.assertTrue(brief["unknowns"])

    def test_event_is_only_shown_inside_the_known_window(self):
        for earnings, show in (("2026-11-01", True), ("2026-01-01", False), ("2027-01-01", False)):
            ctx = context()
            ctx["ticker_contexts"]["TEST"]["earnings_date"] = earnings
            brief = D.build(row(), ctx)
            self.assertEqual(any(o["id"] == "earnings" for o in brief["observations"]), show)

    def test_generation_date_never_substitutes_for_source_timestamp(self):
        ctx = dict(context(), as_of="2030-01-01")
        stock = next(o for o in D.build(row(), ctx)["observations"] if o["id"] == "stock_momentum")
        self.assertIsNone(stock["as_of"])

    def test_old_observations_do_not_count_as_directional_agreement(self):
        brief = D.build(row(), dict(context(), as_of="2026-11-24"))
        market = next(o for o in brief["observations"] if o["id"] == "market_momentum")
        self.assertEqual(market["alignment"], "context")
        self.assertTrue(any("calendar days old" in s for s in brief["unknowns"]))

    def test_future_observations_are_excluded(self):
        brief = D.build(row(), dict(context(), as_of="2026-10-01"))
        self.assertFalse(any(o["id"] == "market_momentum" for o in brief["observations"]))
        self.assertTrue(any("dated after this scan" in s for s in brief["unknowns"]))

    def test_builder_does_not_mutate_or_fetch(self):
        r, ctx = row(), context()
        before = copy.deepcopy((r, ctx))
        with patch("socket.create_connection", side_effect=AssertionError("network forbidden")):
            first = D.build(r, ctx)
            self.assertEqual(first, D.build(r, ctx))
        self.assertEqual((r, ctx), before)
        json.dumps(first, allow_nan=False)

    def test_model_ev_is_always_labelled_as_assumption_dependent(self):
        obs = next(o for o in D.build(row())["observations"] if o["id"] == "model_ev")
        self.assertIn("not verified future profit", obs["implication"])
        self.assertEqual(obs["alignment"], "context")

    def test_empty_news_default_zero_is_not_a_neutral_sentiment_read(self):
        ctx = context()
        ctx["ticker_contexts"]["TEST"]["news_data"] = {"items": [], "aggregate_sentiment": 0}
        brief = D.build(row(), ctx)
        self.assertFalse(any(o["id"] == "news_tone" for o in brief["observations"]))
        self.assertTrue(any("sentiment is unknown" in s for s in brief["unknowns"]))

    def test_news_and_analysts_remain_context_despite_positive_tone(self):
        from types import SimpleNamespace
        ctx = context()
        ctx["ticker_contexts"]["TEST"]["news_data"] = SimpleNamespace(
            items=[SimpleNamespace(source="Reuters", published="2026-10-08T10:00:00Z")],
            aggregate_sentiment=.9, fetched_at="2026-10-08T11:00:00Z",
            analyst_changes=[{"action": "upgrade", "date": "2026-10-08"}])
        brief = D.build(row(), ctx)
        for key in ("news_tone", "analyst_changes"):
            obs = next(o for o in brief["observations"] if o["id"] == key)
            self.assertEqual(obs["alignment"], "context")
        json.dumps(brief, allow_nan=False)

    def test_measured_zero_news_tone_is_preserved(self):
        ctx = context()
        ctx["ticker_contexts"]["TEST"]["news_data"] = {
            "items": [{"source": "Reuters", "published": "2026-10-08"}], "aggregate_sentiment": 0}
        brief = D.build(row(), ctx)
        obs = next(o for o in brief["observations"] if o["id"] == "news_tone")
        self.assertIn("+0.00", obs["fact"])

    def test_compact_output_keeps_unknowns_and_evidence_limit(self):
        labels = dict(D.lines(D.build(row()), compact=True))
        self.assertIn("Unknowns", labels)
        self.assertIn("Evidence", labels)
        self.assertIn("Investigate", labels)


class IntegrationTests(unittest.TestCase):
    def test_market_fetch_captures_dated_evidence_and_reuses_cache(self):
        from src import data_fetching as F
        from unittest.mock import Mock
        spy, vix = Mock(), Mock()
        spy.history.return_value = history(list(range(100, 160)))
        vix.history.return_value = history([15, 16, 17, 18, 19, 22])
        provider = Mock()
        provider.Ticker.side_effect = lambda symbol, **kw: spy if symbol == "SPY" else vix
        with patch.object(F, "_market_context_cache", {"value": None, "ts": 0}), \
             patch.object(F, "yf", provider), patch.object(F, "_init_yfinance"), \
             patch.object(F, "_init_yf_session"), patch.object(F, "_init_request_cache"), \
             patch.object(F, "_yf_throttle"), patch.object(F, "check_macro_risk", return_value=False), \
             patch.object(F, "check_yield_spike", return_value=(False, 0)):
            legacy = F.get_market_context()
            self.assertEqual(legacy, ("Bull", "High", False, 0))
            obs = F.get_cached_market_observations()
            self.assertAlmostEqual(obs["vix"]["returns"]["5"]["value"], 22 / 15 - 1)
            self.assertEqual(obs["vix"]["as_of"], "2026-10-08")
            self.assertEqual(F.get_market_context(), legacy)
            vix.history.assert_called_once_with(period="1mo")

    def test_sector_fetch_adds_aligned_evidence_without_changing_legacy_inputs(self):
        from src import data_fetching as F
        from unittest.mock import Mock
        stock = history([1, 100, 102, 104, 106, 108])
        sector = history([200, 202, 204, 206, 208], start="2026-10-02")
        provider = Mock()
        provider.Ticker.return_value.history.return_value = sector
        with patch.object(F, "yf", provider), patch.object(F, "_init_yfinance"), \
             patch.object(F, "_init_yf_session"):
            result = F.get_sector_performance("TEST", stock)
        self.assertAlmostEqual(result["ticker_return"], 107)
        self.assertAlmostEqual(result["aligned_returns"]["ticker_return"], .08)
        self.assertEqual(result["aligned_returns"]["window_start"], "2026-10-02")

    def test_cache_reader_is_read_only_and_labels_staleness(self):
        from src import data_fetching as F
        cache = {"value": ("Bull", "Low", False, 0), "ts": 1,
                 "observations": {"vix": {"last": 20}}}
        with patch.object(F, "_market_context_cache", cache):
            obs = F.get_cached_market_observations()
            self.assertTrue(obs["cache_stale"])
            obs["vix"]["last"] = 90
            self.assertEqual(cache["observations"]["vix"]["last"], 20)

    def test_html_escapes_evidence_and_preserves_legacy_sidecars(self):
        from src.tearsheet.render import _discovery_section
        self.assertEqual(_discovery_section({}), "")
        brief = D.build(row(symbol="<script>bad()</script>"), context())
        output = _discovery_section({"discovery": brief})
        self.assertNotIn("<script>", output)
        self.assertIn("&lt;script&gt;", output)
        self.assertIn("Sources and observation windows", output)

    def test_saved_brief_renders_identically_after_json_roundtrip(self):
        from src.tearsheet.render import _discovery_section
        data = {"discovery": D.build(row(), context())}
        self.assertEqual(_discovery_section(data), _discovery_section(json.loads(json.dumps(data))))
