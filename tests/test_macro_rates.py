"""Tests for src.macro_rates — FRED no-key rates/macro ingestion."""
import os
import unittest

from src import macro_rates as mr


class TestParseFredCsv(unittest.TestCase):
    def test_parses_date_value_rows_and_skips_header(self):
        csv = "observation_date,DGS10\n2026-06-16,4.43\n2026-06-17,4.49\n"
        obs = mr.parse_fred_csv(csv)
        self.assertEqual(obs, [("2026-06-16", 4.43), ("2026-06-17", 4.49)])

    def test_treats_dot_as_missing_none(self):
        csv = "observation_date,DGS10\n2026-06-15,.\n2026-06-16,4.43\n"
        obs = mr.parse_fred_csv(csv)
        self.assertEqual(obs, [("2026-06-15", None), ("2026-06-16", 4.43)])

    def test_handles_legacy_DATE_header(self):
        csv = "DATE,VALUE\n2026-06-17,4.49\n"
        obs = mr.parse_fred_csv(csv)
        self.assertEqual(obs, [("2026-06-17", 4.49)])

    def test_blank_and_malformed_lines_ignored(self):
        csv = "observation_date,DGS10\n2026-06-17,4.49\n\ngarbage\n"
        obs = mr.parse_fred_csv(csv)
        self.assertEqual(obs, [("2026-06-17", 4.49)])


class TestLatestValid(unittest.TestCase):
    def test_returns_last_non_missing(self):
        obs = [("2026-06-15", 4.40), ("2026-06-16", None), ("2026-06-17", None)]
        self.assertEqual(mr.latest_valid(obs), ("2026-06-15", 4.40))

    def test_empty_returns_none_pair(self):
        self.assertEqual(mr.latest_valid([]), (None, None))


class TestFetchSnapshot(unittest.TestCase):
    def _fake_fetcher(self, series_id):
        data = {
            "DGS10": "observation_date,DGS10\n2026-06-17,4.49\n",
            "DGS2": "observation_date,DGS2\n2026-06-17,3.90\n",
            "T10Y2Y": "observation_date,T10Y2Y\n2026-06-17,0.59\n",
            "DFF": "observation_date,DFF\n2026-06-17,4.33\n",
            "VIXCLS": "observation_date,VIXCLS\n2026-06-17,17.20\n",
        }
        return data[series_id]

    def test_builds_snapshot_from_injected_fetcher(self):
        snap = mr.fetch_rates_snapshot(fetcher=self._fake_fetcher, use_cache=False)
        self.assertAlmostEqual(snap.dgs10, 4.49)
        self.assertAlmostEqual(snap.dgs2, 3.90)
        self.assertAlmostEqual(snap.t10y2y, 0.59)
        self.assertAlmostEqual(snap.dff, 4.33)
        self.assertAlmostEqual(snap.vixcls, 17.20)
        self.assertEqual(snap.as_of["DGS10"], "2026-06-17")

    def test_missing_series_is_none_not_crash(self):
        def partial(series_id):
            if series_id == "DGS10":
                return "observation_date,DGS10\n2026-06-17,4.49\n"
            raise RuntimeError("network down")
        snap = mr.fetch_rates_snapshot(fetcher=partial, use_cache=False)
        self.assertAlmostEqual(snap.dgs10, 4.49)
        self.assertIsNone(snap.dgs2)

    def test_fail_fast_when_first_series_unreachable(self):
        """If the very first series throws (source down), don't hammer the
        rest — bounds worst-case latency to a single timeout."""
        calls = []

        def always_down(series_id):
            calls.append(series_id)
            raise RuntimeError("timeout")

        # yahoo also down: tests the genuine all-sources-unreachable path.
        snap = mr.fetch_rates_snapshot(fetcher=always_down, use_cache=False,
                                       yahoo_fetcher=lambda _s: None)
        self.assertEqual(len(calls), 1)  # aborted after first failure
        self.assertIsNone(snap.dgs10)


class TestCacheNeverStoresFailure(unittest.TestCase):
    def setUp(self):
        import tempfile
        fd, self.tmp = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.unlink(self.tmp)
        self._orig = mr._CACHE_PATH
        mr._CACHE_PATH = self.tmp

    def tearDown(self):
        mr._CACHE_PATH = self._orig
        if os.path.exists(self.tmp):
            os.unlink(self.tmp)

    def test_all_none_fetch_is_not_cached(self):
        def down(_sid):
            raise RuntimeError("timeout")
        # both FRED and yahoo down -> a total failure must not poison the cache.
        mr.fetch_rates_snapshot(fetcher=down, use_cache=True,
                                yahoo_fetcher=lambda _s: None)
        self.assertIsNone(mr._read_cache())

    def test_partial_success_is_cached(self):
        def one(sid):
            if sid == "DGS10":
                return "observation_date,DGS10\n2026-06-17,4.49\n"
            raise RuntimeError("down")
        mr.fetch_rates_snapshot(fetcher=one, use_cache=True)
        cached = mr._read_cache()
        self.assertIsNotNone(cached)
        self.assertAlmostEqual(cached["dgs10"], 4.49)


class TestYahooFallback(unittest.TestCase):
    """When FRED is unreachable, fall back to yfinance Treasury yields."""

    def _fake_yahoo(self, symbol):
        # CBOE yield indices quote 10x the percent (e.g. ^TNX 43.9 == 4.39%).
        return {"^TNX": 43.94, "^IRX": 36.82, "^FVX": 41.63,
                "^TYX": 48.61, "^VIX": 19.2}.get(symbol)

    def test_fallback_builds_snapshot_from_yahoo(self):
        snap = mr.yahoo_rates_fallback(fetcher=self._fake_yahoo)
        self.assertEqual(snap.source, "yahoo")
        self.assertAlmostEqual(snap.dgs10, 4.394, places=3)
        self.assertAlmostEqual(snap.dgs3mo, 3.682, places=3)
        self.assertAlmostEqual(snap.vixcls, 19.2, places=3)

    def test_fallback_computes_10y_3m_slope(self):
        snap = mr.yahoo_rates_fallback(fetcher=self._fake_yahoo)
        self.assertAlmostEqual(snap.t10y3m, 4.394 - 3.682, places=3)

    def test_fallback_handles_already_percent_quotes(self):
        # Some yfinance builds return ^TNX already in percent (4.39, not 43.9).
        def already_pct(symbol):
            return {"^TNX": 4.39, "^IRX": 3.68, "^VIX": 18.9}.get(symbol)
        snap = mr.yahoo_rates_fallback(fetcher=already_pct)
        self.assertAlmostEqual(snap.dgs10, 4.39, places=3)
        self.assertAlmostEqual(snap.dgs3mo, 3.68, places=3)

    def test_fetch_snapshot_uses_yahoo_when_fred_down(self):
        def fred_down(_sid):
            raise RuntimeError("timeout")
        snap = mr.fetch_rates_snapshot(fetcher=fred_down, use_cache=False,
                                       yahoo_fetcher=self._fake_yahoo)
        self.assertEqual(snap.source, "yahoo")
        self.assertAlmostEqual(snap.dgs10, 4.394, places=3)

    def test_fred_success_does_not_trigger_yahoo(self):
        def fred_ok(sid):
            return f"observation_date,{sid}\n2026-06-17,4.49\n"

        def yahoo_should_not_run(_sym):
            raise AssertionError("yahoo fallback should not be called")
        snap = mr.fetch_rates_snapshot(fetcher=fred_ok, use_cache=False,
                                       yahoo_fetcher=yahoo_should_not_run)
        self.assertEqual(snap.source, "FRED")


class TestFormatPanel(unittest.TestCase):
    def test_panel_states_facts(self):
        snap = mr.RatesSnapshot(dgs10=4.49, dgs2=3.90, t10y2y=0.59,
                                dff=4.33, vixcls=17.20,
                                as_of={"DGS10": "2026-06-17"})
        lines = mr.format_rates_panel(snap, width=80)
        body = "\n".join(lines)
        self.assertIn("4.49", body)   # 10Y
        self.assertIn("0.59", body)   # 2s10s

    def test_panel_shows_yahoo_source_and_slope(self):
        snap = mr.RatesSnapshot(dgs10=4.39, dgs2=None, t10y2y=None,
                                dff=None, vixcls=19.2, dgs3mo=3.68,
                                t10y3m=0.71, source="yahoo",
                                as_of={"DGS10": "2026-06-25"})
        body = "\n".join(mr.format_rates_panel(snap, width=80))
        self.assertIn("4.39", body)        # 10Y
        self.assertIn("3.68", body)        # 3M
        self.assertIn("0.71", body)        # 10y-3m slope
        self.assertIn("Yahoo", body)       # source marked
        self.assertIn("19.2", body)        # VIX

    def test_inverted_curve_labeled(self):
        snap = mr.RatesSnapshot(dgs10=4.0, dgs2=4.5, t10y2y=-0.50,
                                dff=4.33, vixcls=17.2, as_of={})
        body = "\n".join(mr.format_rates_panel(snap, width=80))
        self.assertIn("inverted", body.lower())

    def test_all_missing_renders_without_crash(self):
        snap = mr.RatesSnapshot(None, None, None, None, None, {})
        lines = mr.format_rates_panel(snap, width=80)
        self.assertTrue(any("n/a" in ln.lower() or "—" in ln for ln in lines))


if __name__ == "__main__":
    unittest.main()
