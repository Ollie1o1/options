"""Offline regression tests for revisions, failures and honest data coverage."""
import copy
import hashlib
import json
import sqlite3
import tempfile
import unittest
import zlib
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from src.breakout.data import load_series, upsert_ohlcv
from src.breakout.refresh import completed_session, fetch_yahoo, refresh_universe
from src.discovery_research import collection
from src.discovery_research.coverage import price_coverage, snapshot, write_report

NOW = datetime(2026, 11, 24, 23, tzinfo=timezone.utc)
DAYS = ["2026-11-20", "2026-11-23", "2026-11-24"]


def rows(days=DAYS, scale=1):
    return [{"date": d, "close": (100 + i) * scale, "high": (101 + i) * scale,
             "low": (99 + i) * scale, "volume": 1000,
             "source_close": 110 + i, "dividend": 0., "split": 0.} for i, d in enumerate(days)]


class RefreshTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.db = str(Path(self.tmp.name) / "prices.db")

    def refresh(self, symbols=None, fetcher=None, **kwargs):
        return refresh_universe(self.db, symbols or ["SPY"], fetcher=fetcher or (lambda s, start: rows()),
                                now=NOW, pause=0, **kwargs)

    def query(self, sql, params=()):
        with closing(sqlite3.connect(self.db)) as con:
            return con.execute(sql, params).fetchall()

    def test_legacy_rebase_preserves_old_values_and_provider_evidence(self):
        upsert_ohlcv(self.db, "SPY", rows(DAYS[:2], scale=4))
        fetch = Mock(return_value=rows())
        result = self.refresh(fetcher=fetch)
        self.assertTrue(result["complete"])
        self.assertEqual(result["revised"], 2)
        self.assertEqual(result["inserted"], 1)
        fetch.assert_called_once_with("SPY", None)
        self.assertEqual(self.query("SELECT close FROM ohlcv_revisions ORDER BY date"), [(400.,), (404.,)])
        payload, digest, accepted = self.query("SELECT payload,sha256,accepted FROM price_fetches")[0]
        encoded = zlib.decompress(payload)
        self.assertEqual(hashlib.sha256(encoded).hexdigest(), digest)
        self.assertEqual(json.loads(encoded), rows())
        self.assertEqual(accepted, 1)

    def test_full_refresh_repairs_a_legacy_null_without_failing_after_commit(self):
        upsert_ohlcv(self.db, "SPY", rows())
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute("UPDATE ohlcv SET close=NULL WHERE date=?", (DAYS[0],))
        result = self.refresh()
        self.assertTrue(result["complete"])
        self.assertEqual(self.query("SELECT close FROM ohlcv_revisions"), [(None,)])
        self.assertEqual(load_series(self.db, "SPY").close[0], 100.)

    def test_incremental_request_contains_overlap_and_is_idempotent(self):
        self.refresh()
        fetch = Mock(return_value=rows())
        result = self.refresh(fetcher=fetch)
        fetch.assert_called_once_with("SPY", "2026-10-20")
        self.assertEqual((result["inserted"], result["revised"]), (0, 0))
        self.assertEqual(len(load_series(self.db, "SPY").dates), 3)
        self.assertEqual(len(self.query("SELECT * FROM price_fetches")), 2)

    def test_adjustment_revision_requires_full_refetch_and_updates_earlier_history(self):
        self.refresh()
        revised = rows(scale=.5)
        fetch = Mock(return_value=revised)
        result = self.refresh(fetcher=fetch)
        self.assertEqual(fetch.call_args_list[0].args, ("SPY", "2026-10-20"))
        self.assertEqual(fetch.call_args_list[1].args, ("SPY", None))
        self.assertEqual(result["revised"], 3)
        self.assertTrue(result["complete"])
        self.assertEqual(self.query("SELECT accepted FROM price_fetches ORDER BY id"), [(1,), (0,), (1,)])

    def test_failed_full_refetch_never_stitches_differently_adjusted_series(self):
        self.refresh()
        before = self.query("SELECT * FROM ohlcv")
        fetch = Mock(side_effect=[rows(scale=.5), [], []])
        result = self.refresh(fetcher=fetch)
        self.assertFalse(result["complete"])
        self.assertEqual(result["symbols"]["SPY"]["status"], "failed")
        self.assertEqual(before, self.query("SELECT * FROM ohlcv"))
        self.assertEqual(len(self.query("SELECT * FROM ohlcv_revisions")), 0)

    def test_partial_full_response_does_not_delete_or_rewrite_old_sessions(self):
        self.refresh()
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute("UPDATE price_state SET last_full='2026-10-01T00:00:00+00:00'")
        result = self.refresh(fetcher=lambda s, start: rows(DAYS[1:]))
        self.assertFalse(result["complete"])
        self.assertIn("omitted", result["symbols"]["SPY"]["error"])
        self.assertEqual(len(load_series(self.db, "SPY").dates), 3)

    def test_monthly_revalidation_does_not_rely_only_on_overlap(self):
        self.refresh()
        with closing(sqlite3.connect(self.db)) as con, con:
            con.execute("UPDATE price_state SET last_full='2026-10-01T00:00:00+00:00'")
        fetch = Mock(return_value=rows())
        self.refresh(fetcher=fetch)
        fetch.assert_called_once_with("SPY", None)

    def test_bad_rows_and_partial_day_are_rejected_and_retryable(self):
        self.refresh()
        before = self.query("SELECT * FROM ohlcv")
        for mutate in (lambda r: r[0].update(close=float("nan")),
                       lambda r: r[0].update(high=1),
                       lambda r: r[0].update(date="2026-11-28"),
                       lambda r: r.append(copy.deepcopy(r[0]))):
            invalid = rows()
            mutate(invalid)
            out = self.refresh(fetcher=lambda s, start: invalid)
            self.assertFalse(out["complete"])
            self.assertEqual(out["symbols"]["SPY"]["status"], "failed")
            self.assertEqual(self.query("SELECT * FROM ohlcv"), before)

    def test_empty_response_and_exception_are_not_successful_zero_updates(self):
        for response in ([], RuntimeError("feed down")):
            fetch = Mock(side_effect=response) if isinstance(response, Exception) else Mock(return_value=response)
            result = self.refresh(fetcher=fetch)
            self.assertFalse(result["complete"])
            self.assertEqual(fetch.call_count, 2)
            self.assertTrue(result["symbols"]["SPY"]["error"])

    def test_one_symbol_failure_preserves_other_success_and_requires_retry(self):
        def fetch(sym, start):
            if sym == "BAD":
                raise RuntimeError("feed down")
            return rows()
        result = self.refresh(["SPY", "BAD", "A"], fetch)
        self.assertFalse(result["complete"])
        self.assertEqual(result["symbols"]["A"]["status"], "updated")
        self.assertEqual(len(load_series(self.db, "A").dates), 3)

    def test_dead_feed_has_bounded_requests(self):
        fetch = Mock(side_effect=RuntimeError("feed down"))
        out = self.refresh([f"S{i}" for i in range(20)], fetch)
        self.assertFalse(out["complete"])
        self.assertEqual(fetch.call_count, 10)
        self.assertEqual(len(out["symbols"]), 20)

    def test_latest_date_does_not_hide_an_internal_gap(self):
        out = self.refresh(fetcher=lambda s, start: rows([DAYS[0], DAYS[2]]))
        self.assertFalse(out["complete"])
        self.assertEqual(out["symbols"]["SPY"]["status"], "quality_review")

    def test_stale_response_can_preserve_useful_rows_but_never_completes(self):
        out = self.refresh(fetcher=lambda s, start: rows(DAYS[:2]))
        self.assertFalse(out["complete"])
        self.assertEqual(out["symbols"]["SPY"]["status"], "stale")
        self.assertEqual(len(load_series(self.db, "SPY").dates), 2)

    def test_first_seen_does_not_change_on_reobservation(self):
        self.refresh()
        first = self.query("SELECT first_seen FROM ohlcv_evidence")
        refresh_universe(self.db, ["SPY"], now=datetime(2026, 11, 25, 23, tzinfo=timezone.utc),
                         fetcher=lambda s, start: rows(), pause=0)
        self.assertEqual(self.query("SELECT first_seen FROM ohlcv_evidence"), first)

    def test_coverage_distinguishes_stale_missing_gaps_and_unverified(self):
        upsert_ohlcv(self.db, "SPY", rows())
        upsert_ohlcv(self.db, "A", rows([DAYS[0], DAYS[2]]))
        upsert_ohlcv(self.db, "OLD", rows(DAYS[:1]))
        cov = price_coverage(self.db, ["SPY", "A", "OLD", "NONE"], "2026-11-01", NOW)
        records = {r["symbol"]: r for r in cov["symbols"]}
        self.assertEqual(records["SPY"]["status"], "unverified")
        self.assertEqual(records["A"]["missing_spy_dates"], [DAYS[1]])
        self.assertEqual(records["OLD"]["status"], "stale")
        self.assertEqual(records["NONE"]["status"], "missing")
        self.assertIsNone(records["NONE"]["last"])
        self.assertIsNone(records["SPY"]["source"])

    def test_absent_archives_are_unknown_without_creating_databases(self):
        cov = snapshot(self.db, self.db + ".news", self.db + ".analysts", ["SPY"], "2026-01-01", NOW)
        self.assertEqual(cov["prices"]["status"], "unknown")
        self.assertEqual(cov["news"]["status"], "unknown")
        self.assertFalse(Path(self.db).exists())
        self.assertFalse(Path(self.db + ".news").exists())
        html = write_report(cov, self.tmp.name).read_text()
        self.assertIn("Expected completed session", html)
        self.assertIn("Follow-up actions", html)
        self.assertIn("unknown", html)

    def test_extreme_move_remains_flagged_instead_of_inferred_split(self):
        raw = rows()
        raw[-1].update(close=10, high=11, low=9)
        self.refresh(fetcher=lambda s, start: raw)
        cov = price_coverage(self.db, ["SPY"], "2026-11-01", NOW)
        self.assertEqual(cov["symbols"][0]["status"], "quality_review")
        self.assertEqual(load_series(self.db, "SPY").close[-1], 10)
        self.assertEqual(cov["symbols"][0]["large_moves"][0]["date"], DAYS[-1])


class TimingAndProviderTests(unittest.TestCase):
    def test_conservative_cutoff_holidays_weekends_dst_and_unknown_year(self):
        cases = [(datetime(2026, 11, 24, 17, 59), "2026-11-23"),
                 (datetime(2026, 11, 24, 18), "2026-11-24"),
                 (datetime(2026, 11, 26, 19), "2026-11-25"),
                 (datetime(2026, 11, 29, 19), "2026-11-27"),
                 (datetime(2026, 7, 3, 23, tzinfo=timezone.utc), "2026-07-02"),
                 (datetime(2027, 7, 3, 23, tzinfo=timezone.utc), None)]
        for now, expected in cases:
            self.assertEqual(completed_session(now), expected)

    def test_provider_explicit_adjustment_actions_end_and_local_dates(self):
        frame = pd.DataFrame({"Close": [200.], "Adj Close": [100.], "High": [220.], "Low": [180.],
                              "Volume": [1000.], "Dividends": [2.], "Stock Splits": [4.]},
                             index=pd.DatetimeIndex(["2026-11-24"], tz="America/New_York"))
        ticker = Mock()
        ticker.history.return_value = frame
        with patch("yfinance.Ticker", return_value=ticker):
            out = fetch_yahoo("SPY", "2026-10-01", "2026-11-24")
        args = ticker.history.call_args.kwargs
        self.assertEqual(args["start"], "2026-10-01")
        self.assertEqual(args["end"], "2026-11-25")
        self.assertFalse(args["auto_adjust"])
        self.assertFalse(args["repair"])
        self.assertTrue(args["actions"])
        self.assertEqual((out[0]["date"], out[0]["close"], out[0]["high"], out[0]["low"]),
                         ("2026-11-24", 100., 110., 90.))
        self.assertEqual((out[0]["source_close"], out[0]["dividend"], out[0]["split"]), (200., 2., 4.))

    def test_collection_due_only_after_close_and_incomplete_retries(self):
        self.assertFalse(collection.due({}, datetime(2026, 11, 24, 17, 59)))
        self.assertTrue(collection.due({}, datetime(2026, 11, 24, 18)))
        self.assertFalse(collection.due({"last_discovery_prices": "2026-11-24"}, datetime(2026, 11, 24, 19)))
        self.assertFalse(collection.due({}, datetime(2026, 11, 26, 19)))
        self.assertFalse(collection.due({}, datetime(2027, 6, 1, 19)))


class CollectionIntegrationTests(unittest.TestCase):
    def test_collection_reuses_existing_symbols_and_preserves_findings(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "prices.db")
            cfg = Path(d) / "spec.json"
            cfg.write_text(json.dumps({"universe": ["SPY"], "evaluation": {"start": "2026-11-01"}}))
            upsert_ohlcv(db, "EXISTING", rows())
            latest = Path(d) / "latest.json"
            latest.write_text('original dated findings')
            from src.breakout import refresh
            with patch.object(collection, "refresh_universe", side_effect=lambda db, symbols, now: refresh.refresh_universe(
                    db, symbols, now=now, fetcher=lambda s, start: rows(), pause=0)), \
                    patch("src.discovery_research.engine.evaluate", side_effect=AssertionError("Must not evaluate outcomes")):
                out = collection.collect(config_path=str(cfg), prices_path=db, news_path=db + ".news",
                                         analysts_path=db + ".analysts", output_dir=d, now=NOW)
            self.assertTrue(out["complete"])
            self.assertEqual(set(out["collection"]["symbols"]), {"SPY", "EXISTING"})
            self.assertEqual(latest.read_text(), 'original dated findings')
            self.assertTrue((Path(d) / "coverage.html").exists())

    def test_calendar_catches_gap_missing_from_benchmark_itself(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "prices.db")
            upsert_ohlcv(db, "SPY", rows([DAYS[0], DAYS[2]]))
            cov = price_coverage(db, ["SPY"], "2026-11-01", NOW)
            self.assertEqual(cov["symbols"][0]["missing_spy_dates"], [DAYS[1]])
            self.assertEqual(cov["symbols"][0]["status"], "quality_review")

    def test_one_floating_point_unit_at_the_high_is_valid(self):
        with tempfile.TemporaryDirectory() as d:
            raw = rows()
            raw[0]["high"] = raw[0]["close"] - 1e-14
            out = refresh_universe(str(Path(d) / "prices.db"), ["SPY"], now=NOW,
                                   fetcher=lambda s, start: raw, pause=0)
            self.assertTrue(out["complete"])

    def test_tiny_revision_is_retained_even_without_a_full_rebase(self):
        with tempfile.TemporaryDirectory() as d:
            db = str(Path(d) / "prices.db")
            refresh_universe(db, ["SPY"], now=NOW, fetcher=lambda s, start: rows(), pause=0)
            changed = rows()
            changed[0]["close"] += 1e-8
            fetch = Mock(return_value=changed)
            out = refresh_universe(db, ["SPY"], now=NOW, fetcher=fetch, pause=0)
            self.assertTrue(out["complete"])
            self.assertEqual(fetch.call_count, 1)
            self.assertEqual(out["revised"], 1)
            with closing(sqlite3.connect(db)) as con:
                self.assertEqual(con.execute("SELECT close FROM ohlcv_revisions").fetchall(), [(100.,)])


class MissingSchemaTests(unittest.TestCase):
    def test_existing_empty_database_still_produces_unknown_coverage(self):
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / 'empty.db'
            with closing(sqlite3.connect(db)):
                pass
            config = Path(d) / 'config.json'
            config.write_text(json.dumps({'universe': ['SPY'], 'evaluation': {'start': '2026-01-01'}}))
            out = collection.collect(config_path=str(config), prices_path=str(db), news_path=str(db),
                                     analysts_path=str(db), output_dir=d, update=False, now=NOW)
            self.assertFalse(out['complete'])
            self.assertEqual(out['coverage']['prices']['status'], 'unknown')
            self.assertIn('schema', out['coverage']['prices']['error'])
            self.assertTrue((Path(d) / 'coverage.html').exists())


if __name__ == "__main__":
    unittest.main()
