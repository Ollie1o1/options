"""Research must earn its claims with time-correct, paired held-out evidence."""
import copy
import json
import math
import sqlite3
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.discovery_research import data as D, engine as E
from src.discovery_research.report import render


def spec():
    return {"first_test": "2021-01-01", "test_block_sessions": 60,
            "train_window_sessions": 300, "min_train_dates": 50,
            "min_test_dates": 20, "min_test_years": 1, "ridge_alpha": 1.0,
            "bootstrap_repetitions": 300, "bootstrap_seed": 42, "bootstrap_block_dates": 4,
            "family_alpha": .05}


def trial():
    return {"id": "synthetic", "question": "synthetic fixture only", "target": "forward_return",
            "horizon": 5, "population": "SPY", "baseline": ["base"], "candidate": ["candidate"]}


def synthetic_dataset(n=850):
    dates = pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d").tolist()
    rng = np.random.default_rng(42)
    a, b = rng.normal(size=n), rng.normal(size=n)
    data = pd.DataFrame({"date": dates[:-6], "symbol": "SPY", "index": np.arange(n - 6),
                         "base": a[:-6], "candidate": b[:-6],
                         "target": 2 * b[:-6] + rng.normal(scale=.1, size=n - 6),
                         "entry_date": dates[1:-5], "end_date": dates[6:]})
    return data, dates


class TimingTests(unittest.TestCase):
    def test_next_session_close_entry_and_exact_horizon(self):
        dates = pd.bdate_range("2020-01-01", periods=40).strftime("%Y-%m-%d")
        prices = pd.DataFrame({"SPY": np.arange(100, 140, dtype=float)}, index=dates)
        features = pd.DataFrame([{"date": dates[25], "index": 25, "symbol": "SPY", "base": 1, "candidate": 2}])
        out = D.make_dataset(prices, features, trial(), {}, entry_lag=1)
        self.assertEqual(out.iloc[0]["entry_date"], dates[26])
        self.assertEqual(out.iloc[0]["end_date"], dates[31])
        self.assertAlmostEqual(out.iloc[0]["target"], 131 / 126 - 1)

    def test_features_are_invariant_to_future_price_mutation(self):
        dates = pd.bdate_range("2020-01-01", periods=70).strftime("%Y-%m-%d")
        prices = pd.DataFrame({"SPY": np.linspace(100, 150, 70), "A": np.linspace(200, 300, 70),
                               "B": np.linspace(150, 180, 70)}, index=dates)
        vix = pd.Series(np.linspace(15, 25, 70), index=dates)
        original = D.make_features(prices, vix, pd.DataFrame(), pd.DataFrame(), {"g": ["A", "B"]})
        prices.loc[dates[50]:, "B"] *= 2
        vix.loc[dates[50]:] *= 2
        changed = D.make_features(prices, vix, pd.DataFrame(), pd.DataFrame(), {"g": ["A", "B"]})
        pd.testing.assert_frame_equal(original.loc[original["date"] < dates[50]].reset_index(drop=True),
                                      changed.loc[changed["date"] < dates[50]].reset_index(drop=True))

    def test_peer_feature_excludes_the_target(self):
        dates = pd.bdate_range("2020-01-01", periods=30).strftime("%Y-%m-%d")
        prices = pd.DataFrame({"SPY": np.full(30, 100.), "A": np.linspace(100, 300, 30),
                               "B": np.full(30, 200.)}, index=dates)
        features = D.make_features(prices, pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(), {"g": ["A", "B"]})
        target = features.loc[features["symbol"] == "A"]
        self.assertTrue((target["peer_excess5"] == 0).all())

    def test_vix_spike_is_not_misclassified_as_an_equity_split(self):
        dates = pd.bdate_range("2020-01-01", periods=30).strftime("%Y-%m-%d")
        prices = pd.DataFrame({"SPY": np.linspace(100, 110, 30)}, index=dates)
        vix = pd.Series([15.] * 29 + [40.], index=dates)
        features = D.make_features(prices, vix, pd.DataFrame(), pd.DataFrame(), {})
        self.assertAlmostEqual(features.iloc[-1]["vix_change5"], 40 / 15 - 1)

    def test_news_uses_first_seen_not_publication_or_grade_dates(self):
        calendar = ["2026-06-19", "2026-06-22", "2026-06-23"]
        news = pd.DataFrame([{"symbol": "A", "date": "2026-06-21", "sentiment": .5,
                              "relevance": 1, "published": "2020-01-01"}])
        events = D._event_features(news, pd.DataFrame(), calendar)
        self.assertNotIn(("2026-06-19", "A"), events)
        self.assertIn(("2026-06-22", "A"), events)

    def test_no_archived_news_is_missing_not_zero(self):
        self.assertEqual(D._event_features(pd.DataFrame(), pd.DataFrame(), ["2020-01-01"]), {})

    def test_syndicated_weekend_headline_counts_once_on_the_feature_session(self):
        news = pd.DataFrame([{"symbol": "A", "date": day, "sentiment": .5,
                              "relevance": 1, "normalized": "same news"}
                             for day in ("2026-06-20", "2026-06-21")])
        events = D._event_features(news, pd.DataFrame(), ["2026-06-19", "2026-06-22"])
        self.assertAlmostEqual(events[("2026-06-22", "A")]["news_activity"], math.log(2))

    def test_nan_inside_target_window_does_not_bridge_the_gap(self):
        dates = pd.bdate_range("2020-01-01", periods=40).strftime("%Y-%m-%d")
        prices = pd.DataFrame({"SPY": np.arange(100, 140, dtype=float)}, index=dates)
        prices.loc[dates[28], "SPY"] = np.nan
        features = pd.DataFrame([{"date": dates[25], "index": 25, "symbol": "SPY", "base": 1, "candidate": 2}])
        self.assertTrue(D.make_dataset(prices, features, trial(), {}).empty)

    def test_magnitude_target_is_realized_vol_not_absolute_return(self):
        dates = pd.bdate_range("2020-01-01", periods=10).strftime("%Y-%m-%d")
        values = np.array([100, 100, 101, 99, 102, 98, 101, 100, 102, 99.])
        prices = pd.DataFrame({"SPY": values}, index=dates)
        features = pd.DataFrame([{"date": dates[0], "index": 0, "symbol": "SPY", "base": 1, "candidate": 2}])
        t = dict(trial(), target="log_future_realized_vol")
        dataset = D.make_dataset(prices, features, t, {})
        expected = np.diff(np.log(values[1:7])).std(ddof=1) * math.sqrt(252)
        self.assertAlmostEqual(dataset.iloc[0]["target"], math.log(expected))

    def test_relationship_map_does_not_claim_forward_predictability(self):
        rng = np.random.default_rng(51)
        n = 300
        market, shared = rng.normal(0, .01, n), rng.normal(0, .01, n)
        values = {"SPY": 100 * np.exp(np.cumsum(market)),
                  "A": 100 * np.exp(np.cumsum(market + shared)),
                  "B": 100 * np.exp(np.cumsum(market + shared + rng.normal(0, .001, n)))}
        prices = pd.DataFrame(values, index=pd.bdate_range("2020-01-01", periods=n).strftime("%Y-%m-%d"))
        rel = D.relationship_snapshot(prices, {"g": ["A", "B"]})
        self.assertGreater(rel[0]["market_adjusted_correlation"], .9)
        self.assertEqual(rel[0]["observations"], 252)
        self.assertIn("not causation, lead-lag or forecast", rel[0]["interpretation"])


class EvaluationTests(unittest.TestCase):
    def test_strict_label_end_purge_and_nonoverlapping_tests(self):
        dataset, calendar = synthetic_dataset()
        out = E.evaluate(dataset, calendar, trial(), spec())
        self.assertGreater(out["test_dates"], 20)
        for fold in out["folds"]:
            if fold["status"] == "evaluated":
                self.assertLess(fold["train_label_end"], fold["test_start"])
        last_end = ""
        for pred in sorted(out["predictions"], key=lambda r: r["date"]):
            self.assertGreater(pred["date"], last_end)
            last_end = pred["end_date"]

    def test_equal_weighted_dates_not_correlated_stock_rows(self):
        dataset, calendar = synthetic_dataset()
        base = E.evaluate(dataset, calendar, trial(), spec())
        duplicate = dataset.copy()
        duplicate["symbol"] = "COPY"
        doubled = E.evaluate(pd.concat([dataset, duplicate]), calendar, trial(), spec())
        self.assertEqual(base["test_dates"], doubled["test_dates"])
        self.assertEqual(doubled["test_rows"], base["test_rows"] * 2)

    def test_future_labels_cannot_change_earlier_fold_predictions(self):
        dataset, calendar = synthetic_dataset()
        before = E.evaluate(dataset, calendar, trial(), spec())
        altered = dataset.copy()
        boundary = before["folds"][0]["test_end"]
        altered.loc[altered["date"] > boundary, "target"] *= -100
        after = E.evaluate(altered, calendar, trial(), spec())
        old = [(r["date"], r["candidate"]) for r in before["predictions"] if r["date"] <= boundary]
        new = [(r["date"], r["candidate"]) for r in after["predictions"] if r["date"] <= boundary]
        self.assertEqual(old, new)

    def test_scaling_uses_train_not_test(self):
        x = np.arange(10, dtype=float)[:, None]
        y = 2 * x[:, 0]
        a = E.predict_ridge(x, y, np.array([[11.]]))
        b = E.predict_ridge(x, y, np.array([[11.], [1e9]]))
        self.assertAlmostEqual(a[0], b[0])

    def test_zero_variance_feature_does_not_create_infinite_forecast(self):
        result = E.predict_ridge(np.ones((20, 2)), np.arange(20.), np.ones((3, 2)))
        self.assertTrue(np.all(np.isfinite(result)))

    def test_holm_accounts_for_all_trials_including_insufficient(self):
        self.assertEqual(E.holm_adjust([.01, .02, 1., 1.]), [.04, .06, 1., 1.])

    def test_insufficient_dates_cannot_be_promoted_by_a_tiny_p(self):
        out = {"status": "INSUFFICIENT", "p_value": 1e-12}
        result = E.finalize_family([out], spec())[0]
        self.assertEqual(result["status"], "INSUFFICIENT")
        self.assertEqual(result["adjusted_p"], 1)

    def test_year_consistency_is_required_even_with_small_p(self):
        r = {"status": "AWAITING_FAMILY_ADJUSTMENT", "p_value": .001,
             "error_improvement_ci": [.1, .2],
             "per_year": {"2022": {"mean_improvement": -1}, "2023": {"mean_improvement": -1},
                          "2024": {"mean_improvement": 10}}}
        self.assertEqual(E.finalize_family([r], spec())[0]["status"], "NOT_ESTABLISHED")

    def test_block_interval_is_seeded_and_null_centered(self):
        values = np.arange(1, 101.)
        a = E.block_mean_interval(values, seed=42)
        self.assertEqual(a, E.block_mean_interval(values, seed=42))
        self.assertGreater(a["ci"][0], 0)
        self.assertLess(a["p_value"], .01)

    def test_negative_improvement_does_not_pass_positive_evidence_test(self):
        out = E.block_mean_interval(-np.arange(1, 101.))
        self.assertGreater(out["p_value"], .9)


class InputReportTests(unittest.TestCase):
    def test_research_menu_opens_saved_findings_without_rerunning_the_study(self):
        import contextlib
        import io
        from unittest.mock import patch
        from src.launcher import _research_menu
        with patch("builtins.input", side_effect=["5", "B"]), \
             patch("src.discovery_research.__main__.main", return_value=0) as entry, \
             contextlib.redirect_stdout(io.StringIO()):
            _research_menu()
        entry.assert_called_once_with(["--view"])

    def test_index_csv_records_fingerprint_and_dates(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "vix.csv"
            p.write_text("DATE,OPEN,HIGH,LOW,CLOSE\n01/02/2020,10,10,10,10\n01/03/2020,12,12,12,12\n")
            series, coverage = D.load_cboe_csv(str(p))
            self.assertEqual(series["2020-01-03"], 12)
            self.assertEqual(len(coverage["sha256"]), 64)

    def test_news_loader_deduplicates_and_uses_archival_time(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "news.db"
            with sqlite3.connect(p) as con:
                con.execute("CREATE TABLE news_archive (symbol,archived_at,sentiment,relevance,headline)")
                con.executemany("INSERT INTO news_archive VALUES (?,?,?,?,?)", [
                    ("A", "2026-06-22T22:00:00Z", .5, 1, "New guidance!"),
                    ("A", "2026-06-22T23:00:00Z", .5, 1, "NEW GUIDANCE")])
            frame, coverage = D.load_news(str(p), ["A"])
            self.assertEqual(len(frame), 1)
            self.assertEqual(frame.iloc[0]["date"], "2026-06-22")
            self.assertIn("first archived_at", coverage["availability"])

    def test_missing_sources_are_visible_without_creating_databases(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "absent.db"
            frame, coverage = D.load_news(str(p), ["A"])
            self.assertTrue(frame.empty)
            self.assertIn("error", coverage)
            self.assertFalse(p.exists())

    def test_report_escapes_content_and_explains_insufficient_evidence(self):
        data = {"results": [{"id": "<script>alert(123)</script>", "question": "fixture", "status": "INSUFFICIENT",
                             "test_dates": 0, "test_rows": 0, "adjusted_p": 1, "reason": "<script>alert(123)</script>"}],
                "limitations": ["fixture"], "coverage": {}}
        html = render(data)
        self.assertNotIn("<script>alert(123)</script>", html)
        self.assertIn("&lt;script&gt;", html)
        self.assertIn("not trading returns", html)

    def test_registered_family_is_fixed_and_uses_next_session_entry(self):
        path = Path(__file__).resolve().parents[1] / "configs/discovery_research_v1.json"
        prereg = json.loads(path.read_text())
        self.assertEqual(len(prereg["trials"]), 8)
        self.assertEqual(prereg["evaluation"]["entry_lag_sessions"], 1)
        self.assertEqual(prereg["evaluation"]["bootstrap_block_dates"], 4)

    def test_complete_offline_run_writes_reproducible_audit_and_report(self):
        from src.discovery_research.__main__ import run
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            dates = pd.bdate_range("2020-01-01", periods=800).strftime("%Y-%m-%d").tolist()
            rng = np.random.default_rng(42)
            prices = 100 * np.exp(np.cumsum(rng.normal(0, .01, len(dates))))
            price_db, news_db, analyst_db = root / "prices.db", root / "news.db", root / "analysts.db"
            with sqlite3.connect(price_db) as con:
                con.execute("CREATE TABLE ohlcv (ticker,date,close)")
                con.executemany("INSERT INTO ohlcv VALUES (?,?,?)", [("SPY", day, float(v)) for day, v in zip(dates, prices)])
            with sqlite3.connect(news_db) as con:
                con.execute("CREATE TABLE news_archive (symbol,archived_at,sentiment,relevance,headline)")
            with sqlite3.connect(analyst_db) as con:
                con.execute("CREATE TABLE analyst_actions (symbol,first_seen,action)")
            config = {"id": "fixture-only", "universe": ["SPY"], "peer_groups": {}, "limitations": ["synthetic"],
                      "evaluation": dict(spec(), start="2020-01-01", entry_lag_sessions=1),
                      "trials": [dict(trial(), baseline=["ret5"], candidate=["log_rv20"]) ]}
            path = root / "spec.json"
            path.write_text(json.dumps(config))
            result = run(str(path), str(price_db), str(news_db), str(analyst_db), str(root / "missing.csv"), str(root / "out"))
            saved = json.loads((root / "out" / "latest.json").read_text())
            self.assertEqual(saved, result)
            self.assertTrue((root / "out" / "latest.html").exists())
            self.assertIn("source_code_sha256", saved)
            self.assertIn(saved["results"][0]["status"], ("INSUFFICIENT", "NOT_ESTABLISHED", "RESEARCH_SUPPORT"))
