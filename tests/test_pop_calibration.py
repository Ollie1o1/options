"""Tests for src/pop_calibration.py — the calibrated probability of profit.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_pop_calibration -v

Every statistic is checked against data with a KNOWN answer. The two tests
that matter most are `test_walk_forward_never_trains_on_its_own_day` and
`test_ship_check_refuses_a_flat_curve`: the first is the only thing standing
between this module and a leaked result, and the second is the only thing
standing between the operator and a seventh decorative number.
"""
import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import pop_calibration as pc


def _planted(beta, n=3000, seed=1, strategies=("Bull Put",)):
    """Rows whose win probability really is sigmoid(intercept + beta * z)."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    p = 1.0 / (1.0 + np.exp(-(beta * z)))
    wins = rng.random(size=n) < p
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 10, "D")
    return pd.DataFrame({
        "abs_delta": z,
        "dte": np.zeros(n),
        "entry_iv": np.zeros(n),
        "iv_rank_score": np.zeros(n),
        "credit_to_width": np.zeros(n),
        "strategy": [strategies[i % len(strategies)] for i in range(n)],
        "entry_date": dates.strftime("%Y-%m-%d"),
        "won": wins.astype(int),
    })


class TestFit(unittest.TestCase):
    def test_a_planted_coefficient_is_recovered(self):
        model = pc.fit(_planted(beta=1.5), features=["abs_delta"])
        self.assertAlmostEqual(model.coefficient("abs_delta"), 1.5, delta=0.2)

    def test_no_relationship_reads_near_zero(self):
        model = pc.fit(_planted(beta=0.0), features=["abs_delta"])
        self.assertAlmostEqual(model.coefficient("abs_delta"), 0.0, delta=0.2)

    def test_a_constant_feature_does_not_blow_up_the_fit(self):
        """pop_score spans five points inside a strategy. Near-constant columns
        are the normal case here, not the pathological one."""
        df = _planted(beta=1.0)
        df["abs_delta"] = 0.53
        model = pc.fit(df, features=["abs_delta"])
        self.assertTrue(np.isfinite(model.coefficient("abs_delta")))


class TestPredict(unittest.TestCase):
    def test_predictions_are_probabilities(self):
        df = _planted(beta=2.0)
        p = pc.predict(pc.fit(df, features=["abs_delta"]), df)
        self.assertEqual(len(p), len(df))
        self.assertTrue(((p >= 0.0) & (p <= 1.0)).all())

    def test_a_higher_feature_gives_a_higher_probability(self):
        df = _planted(beta=2.0)
        model = pc.fit(df, features=["abs_delta"])
        p = pc.predict(model, df)
        lo = p[df["abs_delta"] < df["abs_delta"].median()].mean()
        hi = p[df["abs_delta"] > df["abs_delta"].median()].mean()
        self.assertGreater(hi, lo)


class TestWalkForward(unittest.TestCase):
    def test_walk_forward_never_trains_on_its_own_day(self):
        """THE test. A model that has seen the row it is predicting, or
        anything from that row's day, reports a skill it does not have."""
        oos = pc.walk_forward(_planted(beta=1.5), features=["abs_delta"],
                              seed_n=300, step=50)
        self.assertGreater(len(oos), 0)
        self.assertTrue((oos["trained_through"] < oos["entry_date"]).all())

    def test_every_row_after_the_seed_gets_a_prediction(self):
        df = _planted(beta=1.5, n=1000)
        oos = pc.walk_forward(df, features=["abs_delta"], seed_n=300, step=50)
        # Rows sharing the seed boundary's date are held back, so allow slack.
        self.assertGreater(len(oos), 600)
        self.assertLessEqual(len(oos), 700)

    def test_a_planted_signal_survives_out_of_sample(self):
        oos = pc.walk_forward(_planted(beta=2.0), features=["abs_delta"],
                              seed_n=300, step=50)
        lo = oos[oos["predicted"] < 0.5]["won"].mean()
        hi = oos[oos["predicted"] > 0.5]["won"].mean()
        self.assertGreater(hi, lo + 0.15)

    def test_too_few_rows_yields_an_empty_frame_not_a_crash(self):
        oos = pc.walk_forward(_planted(beta=1.0, n=50), features=["abs_delta"],
                              seed_n=300, step=50)
        self.assertEqual(len(oos), 0)


class TestReliability(unittest.TestCase):
    def test_a_known_rate_is_recovered_in_its_bucket(self):
        oos = pd.DataFrame({
            "predicted": [0.65] * 100,
            "won": [1] * 70 + [0] * 30,
        })
        rel = pc.reliability(oos, min_n=20)
        row = rel[rel["bucket_lo"] == 0.6].iloc[0]
        self.assertEqual(row["n"], 100)
        self.assertAlmostEqual(row["realised"], 0.70, places=6)
        self.assertLess(row["ci_lo"], 0.70)
        self.assertGreater(row["ci_hi"], 0.70)

    def test_a_thin_bucket_is_reported_but_not_qualifying(self):
        oos = pd.DataFrame({"predicted": [0.65] * 5, "won": [1] * 5})
        rel = pc.reliability(oos, min_n=20)
        self.assertEqual(len(rel), 1)
        self.assertFalse(bool(rel.iloc[0]["qualifies"]))


class TestShipCheck(unittest.TestCase):
    def test_ship_check_refuses_a_flat_curve(self):
        """Every bucket wins at the same rate: the number orders nothing."""
        rows = []
        for lo in (0.3, 0.4, 0.5, 0.6):
            rows += [{"predicted": lo + 0.05, "won": int(i < 50)} for i in range(100)]
        ok, reason = pc.ship_check(pc.reliability(pd.DataFrame(rows), min_n=20))
        self.assertFalse(ok)
        self.assertIn("slope", reason.lower())

    def test_ship_check_accepts_a_curve_that_tracks_its_prediction(self):
        rows = []
        for lo in (0.3, 0.4, 0.5, 0.6, 0.7):
            k = int(round(lo * 200))
            rows += [{"predicted": lo + 0.05, "won": int(i < k)} for i in range(200)]
        ok, _ = pc.ship_check(pc.reliability(pd.DataFrame(rows), min_n=20))
        self.assertTrue(ok)

    def test_too_few_qualifying_buckets_refuses(self):
        oos = pd.DataFrame({"predicted": [0.65] * 100, "won": [1] * 70 + [0] * 30})
        ok, reason = pc.ship_check(pc.reliability(oos, min_n=20))
        self.assertFalse(ok)
        self.assertIn("bucket", reason.lower())


def _planted_return(beta, n=3000, seed=3, noise=0.20):
    """Rows whose expected return on risk really is beta * z."""
    rng = np.random.default_rng(seed)
    z = rng.normal(size=n)
    ret = beta * z + rng.normal(scale=noise, size=n)
    dates = pd.to_datetime("2026-01-01") + pd.to_timedelta(np.arange(n) // 10, "D")
    return pd.DataFrame({
        "abs_delta": z,
        "dte": np.zeros(n), "entry_iv": np.zeros(n),
        "iv_rank_score": np.zeros(n), "credit_to_width": np.zeros(n),
        "strategy": ["Bull Put"] * n,
        "entry_date": dates.strftime("%Y-%m-%d"),
        "won": (ret > 0).astype(int),
        "ret_on_risk": ret,
    })


class TestReturnModel(unittest.TestCase):
    """Win rate is not profit. Measured out-of-sample on the real book, the
    0.4-0.5 bucket wins 44.3% at PF 0.66 while the 0.3-0.4 bucket wins 36.5%
    at PF 1.13. A probability that is honest about winning can still mislead
    about money, so the expected return gets its own model and own guard."""

    def test_a_planted_return_coefficient_is_recovered(self):
        model = pc.fit_return(_planted_return(beta=0.5), features=["abs_delta"])
        self.assertAlmostEqual(model.coefficient("abs_delta"), 0.5, delta=0.05)

    def test_no_relationship_reads_near_zero(self):
        model = pc.fit_return(_planted_return(beta=0.0), features=["abs_delta"])
        self.assertAlmostEqual(model.coefficient("abs_delta"), 0.0, delta=0.05)

    def test_walk_forward_on_returns_never_trains_on_its_own_day(self):
        oos = pc.walk_forward(_planted_return(beta=0.5), features=["abs_delta"],
                              seed_n=300, step=50, target="ret_on_risk")
        self.assertGreater(len(oos), 0)
        self.assertTrue((oos["trained_through"] < oos["entry_date"]).all())

    def test_a_planted_return_signal_survives_out_of_sample(self):
        oos = pc.walk_forward(_planted_return(beta=0.5), features=["abs_delta"],
                              seed_n=300, step=50, target="ret_on_risk")
        med = oos["predicted"].median()
        lo = oos[oos["predicted"] <= med]["actual"].mean()
        hi = oos[oos["predicted"] > med]["actual"].mean()
        self.assertGreater(hi, lo + 0.3)

    def test_return_reliability_recovers_a_known_mean(self):
        oos = pd.DataFrame({"predicted": np.linspace(-1, 1, 500),
                            "actual": np.linspace(-1, 1, 500)})
        rel = pc.return_reliability(oos, n_buckets=5)
        self.assertEqual(len(rel), 5)
        self.assertTrue((rel["n"] > 0).all())
        np.testing.assert_allclose(rel["mean_predicted"], rel["mean_actual"],
                                   atol=1e-9)

    def test_rows_with_no_recorded_return_are_dropped_not_zeroed(self):
        """A trade whose capital at risk was never recorded has no return.
        Training it as 0% invents a data point at exactly the value that
        flattens a slope."""
        df = _planted_return(beta=1.0, n=600)
        clean = pc.fit_return(df, features=["abs_delta"])
        poisoned = df.copy()
        # Half the rows lose their denominator; the true relationship is
        # unchanged, so the coefficient must be too.
        poisoned.loc[poisoned.index[::2], "ret_on_risk"] = np.nan
        model = pc.fit_return(poisoned, features=["abs_delta"])
        self.assertAlmostEqual(model.coefficient("abs_delta"),
                               clean.coefficient("abs_delta"), delta=0.1)
        self.assertEqual(model.n_train, 300)

    def test_ship_check_return_refuses_a_flat_relationship(self):
        rng = np.random.default_rng(7)
        oos = pd.DataFrame({"predicted": rng.normal(size=2000),
                            "actual": rng.normal(size=2000)})
        ok, reason = pc.ship_check_return(pc.return_reliability(oos))
        self.assertFalse(ok)
        self.assertIn("slope", reason.lower())

    def test_ship_check_return_accepts_a_real_relationship(self):
        rng = np.random.default_rng(8)
        x = rng.normal(size=4000)
        oos = pd.DataFrame({"predicted": x, "actual": x + rng.normal(scale=0.3, size=4000)})
        ok, _ = pc.ship_check_return(pc.return_reliability(oos))
        self.assertTrue(ok)


class TestLoadTrainingSet(unittest.TestCase):
    """A test that names the real ledger is a test that migrates it."""

    def _ledger(self, path):
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE trades (
            entry_id INTEGER PRIMARY KEY, date TEXT, expiration TEXT,
            strategy_name TEXT, status TEXT, pnl_usd REAL,
            entry_delta REAL, entry_iv REAL, iv_rank_score REAL,
            net_credit REAL, spread_width REAL, capital_at_risk REAL)""")
        rows = [
            (1, "2026-05-01", "2026-06-01", "Bull Put", "CLOSED", 55.0, -0.28, 0.31, 0.4, 1.2, 5.0, 380.0),
            (2, "2026-05-02", "2026-06-01", "Bull Put", "CLOSED", -80.0, -0.34, 0.36, 0.5, 1.4, 5.0, 360.0),
            (3, "2026-05-03", "2026-06-01", "Long Call", "CLOSED", 20.0, 0.42, 0.44, 0.6, None, None, 200.0),
            (4, "2026-05-04", "2026-06-01", "Bull Put", "OPEN", None, -0.30, 0.33, 0.4, 1.1, 5.0, 390.0),
            (5, "2026-05-05", "2026-06-01", "Bull Put", "CLOSED", None, -0.30, 0.33, 0.4, 1.1, 5.0, 390.0),
        ]
        conn.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        conn.commit()
        conn.close()

    def test_only_closed_rows_with_an_outcome_are_loaded(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.db")
            self._ledger(path)
            df = pc.load_training_set(path)
        self.assertEqual(len(df), 3)
        self.assertEqual(set(df["won"]), {0, 1})

    def test_derived_columns_are_present_and_signed_correctly(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.db")
            self._ledger(path)
            df = pc.load_training_set(path)
        row = df[df["entry_date"] == "2026-05-01"].iloc[0]
        self.assertAlmostEqual(row["abs_delta"], 0.28, places=6)
        self.assertAlmostEqual(row["dte"], 31.0, places=6)
        self.assertAlmostEqual(row["credit_to_width"], 1.2 / 5.0, places=6)
        self.assertEqual(row["won"], 1)

    def test_a_missing_credit_to_width_is_zero_not_dropped(self):
        """Single-leg rows have no credit or width. Dropping them would throw
        away 383 of the 909 closed trades."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.db")
            self._ledger(path)
            df = pc.load_training_set(path)
        row = df[df["strategy"] == "Long Call"].iloc[0]
        self.assertEqual(row["credit_to_width"], 0.0)

    def test_ret_on_risk_is_pnl_over_capital_at_risk(self):
        """Naming the denominator is not optional here. Premium, credit and
        capital at risk coincide only for long premium, and the same trades
        give opposite answers on the wrong one."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.db")
            self._ledger(path)
            df = pc.load_training_set(path)
        row = df[df["entry_date"] == "2026-05-01"].iloc[0]
        self.assertAlmostEqual(row["ret_on_risk"], 55.0 / 380.0, places=9)
        loser = df[df["entry_date"] == "2026-05-02"].iloc[0]
        self.assertAlmostEqual(loser["ret_on_risk"], -80.0 / 360.0, places=9)

    def test_a_row_without_capital_at_risk_has_no_return(self):
        """NULL means not recorded, never zero. A missing denominator must
        not become a 0% return that drags an average toward the middle."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "ledger.db")
            conn = sqlite3.connect(path)
            conn.execute("""CREATE TABLE trades (
                entry_id INTEGER PRIMARY KEY, date TEXT, expiration TEXT,
                strategy_name TEXT, status TEXT, pnl_usd REAL,
                entry_delta REAL, entry_iv REAL, iv_rank_score REAL,
                net_credit REAL, spread_width REAL, capital_at_risk REAL)""")
            conn.execute("INSERT INTO trades VALUES "
                         "(1,'2026-05-01','2026-06-01','Bull Put','CLOSED',"
                         "55.0,-0.28,0.31,0.4,1.2,5.0,NULL)")
            conn.commit()
            conn.close()
            df = pc.load_training_set(path)
        self.assertEqual(len(df), 1)
        self.assertTrue(pd.isna(df.iloc[0]["ret_on_risk"]))
        self.assertEqual(df.iloc[0]["won"], 1)

    def test_an_unreadable_database_returns_empty_not_an_exception(self):
        df = pc.load_training_set("/nonexistent/nowhere.db")
        self.assertEqual(len(df), 0)


class TestModelArtifact(unittest.TestCase):
    def test_a_saved_model_predicts_identically_when_reloaded(self):
        df = _planted(beta=1.5)
        model = pc.fit(df, features=["abs_delta"])
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.json")
            pc.save_model(model, path, shipped=True, reason="test")
            reloaded = pc.load_model(path)
        self.assertIsNotNone(reloaded)
        assert reloaded is not None
        np.testing.assert_allclose(pc.predict(model, df),
                                   pc.predict(reloaded, df), rtol=1e-9)

    def test_a_missing_artifact_loads_as_none(self):
        self.assertIsNone(pc.load_model("/nonexistent/model.json"))

    def test_an_unshipped_artifact_loads_as_none(self):
        """The guard has to hold at the read side too, or a flat model still
        reaches the board."""
        df = _planted(beta=1.5)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.json")
            pc.save_model(pc.fit(df, features=["abs_delta"]), path,
                          shipped=False, reason="slope CI contains zero")
            self.assertIsNone(pc.load_model(path))


class TestDescribeRow(unittest.TestCase):
    """Rendering is asserted on OUTPUT. A source grep is not a rendering
    test — this repo has shipped a column that was never drawn."""

    def _model_and_rel(self):
        df = _planted(beta=2.0)
        model = pc.fit(df, features=["abs_delta"])
        oos = pc.walk_forward(df, features=["abs_delta"], seed_n=300, step=50)
        return model, pc.reliability(oos)

    def test_the_line_carries_the_prediction_and_its_evidence(self):
        model, rel = self._model_and_rel()
        line = pc.describe_row({"abs_delta": 1.0, "strategy": "Bull Put"},
                               model=model, rel=rel)
        self.assertIsNotNone(line)
        assert line is not None
        self.assertIn("%", line)
        self.assertRegex(line, r"\d+ out-of-sample")
        self.assertRegex(line, r"\[\d+%, \d+%\]")

    def test_no_model_renders_nothing(self):
        self.assertIsNone(pc.describe_row({"abs_delta": 0.3}, model=None,
                                          rel=pd.DataFrame()))

    def test_a_bucket_with_no_support_renders_nothing(self):
        """Predicting into a region the model was never CHECKED in must show
        nothing, not a bare number carrying implied authority. On the real
        book the 0.8-0.9 and 0.9-1.0 buckets hold 3 and 1 predictions."""
        model, rel = self._model_and_rel()
        row = {"abs_delta": 1.0, "strategy": "Bull Put"}
        p = float(pc.predict(model, pd.DataFrame([pc.row_features(row)
                                                  | {"strategy": "Bull Put"}]))[0])
        # Withdraw support from exactly the bucket this row lands in.
        starved = rel.copy()
        hit = np.isclose(starved["bucket_lo"], np.floor(p * 10) / 10)
        self.assertTrue(hit.any(), "row must land in a known bucket")
        starved.loc[hit, "qualifies"] = False

        self.assertIsNotNone(pc.describe_row(row, model=model, rel=rel))
        self.assertIsNone(pc.describe_row(row, model=model, rel=starved))

    def test_scan_row_column_names_are_understood(self):
        """Scan rows do not use ledger column names. A mapping that silently
        read every feature as zero would render a confident constant."""
        model, rel = self._model_and_rel()
        a = pc.row_features({"delta": -0.30, "days_to_expiry": 45,
                             "impliedVolatility": 0.32, "iv_rank": 0.6,
                             "net_credit": 1.2, "spread_width": 5.0})
        self.assertAlmostEqual(a["abs_delta"], 0.30, places=6)
        self.assertAlmostEqual(a["dte"], 45.0, places=6)
        self.assertAlmostEqual(a["entry_iv"], 0.32, places=6)
        self.assertAlmostEqual(a["credit_to_width"], 0.24, places=6)


class TestBoardRendering(unittest.TestCase):
    """The board is asserted on its OUTPUT. This repo has shipped a column
    that a source grep proved present and a render proved absent."""

    def setUp(self):
        from src import cli_display, formatting
        self.cd, self.fmt = cli_display, formatting
        self._colour = formatting._COLOR_ENABLED
        formatting._COLOR_ENABLED = False
        cli_display._CAL_CACHE.clear()

    def tearDown(self):
        self.fmt._COLOR_ENABLED = self._colour
        self.cd._CAL_CACHE.clear()

    def _row(self):
        return pd.Series({
            "prob_profit": 0.62, "delta": -0.30, "days_to_expiry": 45,
            "impliedVolatility": 0.32, "iv_rank": 0.55, "net_credit": 1.20,
            "spread_width": 5.0, "strategy": "Bull Put",
        })

    def test_a_shipped_model_puts_its_line_on_the_board(self):
        df = _planted(beta=2.0)
        model = pc.fit(df, features=list(pc.DEFAULT_FEATURES))
        oos = pc.walk_forward(df, seed_n=300, step=50)
        rel = pc.reliability(oos)
        self.cd._CAL_CACHE.update(model=model, rel=rel, mod=pc)

        out = " ".join(str(x) for x in self.cd.format_analysis_lines(
            self._row(), chain_iv_median=0.30, mode="Premium Selling"))
        self.assertIn("CalPoP", out)
        self.assertIn("out-of-sample analogues", out)

    def test_no_shipped_model_draws_nothing(self):
        """A refused model must leave the board exactly as it was."""
        self.cd._CAL_CACHE.update(model=None, rel=None, mod=None)
        out = " ".join(str(x) for x in self.cd.format_analysis_lines(
            self._row(), chain_iv_median=0.30, mode="Premium Selling"))
        self.assertNotIn("CalPoP", out)
        self.assertIn("PoP:", out)  # the rest of the board is untouched


class TestReport(unittest.TestCase):
    """The report is what turns a fit into a shipped artifact, so its refusal
    path needs a test as much as its success path does."""

    def _ledger(self, path, n=500, signal=True):
        rng = np.random.default_rng(11)
        conn = sqlite3.connect(path)
        conn.execute("""CREATE TABLE trades (
            entry_id INTEGER PRIMARY KEY, date TEXT, expiration TEXT,
            strategy_name TEXT, status TEXT, pnl_usd REAL,
            entry_delta REAL, entry_iv REAL, iv_rank_score REAL,
            net_credit REAL, spread_width REAL, capital_at_risk REAL)""")
        base = pd.to_datetime("2026-01-01")
        for i in range(n):
            delta = float(rng.uniform(0.05, 0.60))
            p = 1.0 - delta if signal else 0.5
            pnl = 100.0 if rng.random() < p else -100.0
            d = (base + pd.Timedelta(days=i // 5)).strftime("%Y-%m-%d")
            conn.execute("INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                         (i + 1, d, "2026-12-31", "Bull Put", "CLOSED", pnl,
                          -delta, 0.3, 0.5, 1.0, 5.0, 500.0))
        conn.commit()
        conn.close()

    def test_a_real_signal_produces_a_shipped_artifact(self):
        from scripts import pop_calibration_report as rep
        with tempfile.TemporaryDirectory() as d:
            db, art = os.path.join(d, "l.db"), os.path.join(d, "m.json")
            self._ledger(db, signal=True)
            out = rep.run(db, art, seed_n=200, step=50)
            self.assertTrue(out["pop"]["shipped"], out["pop"]["reason"])
            self.assertIsNotNone(pc.load_model(art))

    def test_noise_produces_an_artifact_that_refuses_to_load(self):
        from scripts import pop_calibration_report as rep
        with tempfile.TemporaryDirectory() as d:
            db, art = os.path.join(d, "l.db"), os.path.join(d, "m.json")
            self._ledger(db, signal=False)
            out = rep.run(db, art, seed_n=200, step=50)
            self.assertFalse(out["pop"]["shipped"])
            self.assertTrue(os.path.exists(art))
            self.assertIsNone(pc.load_model(art))

    def test_an_empty_ledger_refuses_rather_than_raising(self):
        from scripts import pop_calibration_report as rep
        with tempfile.TemporaryDirectory() as d:
            db, art = os.path.join(d, "l.db"), os.path.join(d, "m.json")
            conn = sqlite3.connect(db)
            conn.execute("CREATE TABLE trades (entry_id INTEGER, date TEXT, "
                         "expiration TEXT, strategy_name TEXT, status TEXT, "
                         "pnl_usd REAL, entry_delta REAL, entry_iv REAL, "
                         "iv_rank_score REAL, net_credit REAL, "
                         "spread_width REAL, capital_at_risk REAL)")
            conn.commit()
            conn.close()
            out = rep.run(db, art, seed_n=200, step=50)
            self.assertFalse(out["pop"]["shipped"])


if __name__ == "__main__":
    unittest.main()
