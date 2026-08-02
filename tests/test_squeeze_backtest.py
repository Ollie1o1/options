"""Tests for the squeeze point-in-time backtest harness."""
import math
import os
import sqlite3
import tempfile
import unittest

import numpy as np

from src.squeeze.backtest import finra, panel, shares, study


class TrendTest(unittest.TestCase):
    def test_matches_live_tolerance(self):
        # ±5% band, same as src/short_interest.py
        self.assertEqual(panel._trend(110, 100), "rising")
        self.assertEqual(panel._trend(90, 100), "falling")
        self.assertEqual(panel._trend(102, 100), "flat")

    def test_missing_inputs_are_none(self):
        self.assertIsNone(panel._trend(None, 100))
        self.assertIsNone(panel._trend(100, 0))


class CandidateDatesTest(unittest.TestCase):
    def test_covers_mid_month_and_month_end_weekdays_only(self):
        got = finra.candidate_dates("2024-03-01", "2024-03-31")
        self.assertIn("2024-03-15", got)
        self.assertIn("2024-03-28", got)   # 3/29 was Good Friday, 3/30-31 weekend
        for d in got:
            import datetime as dt
            self.assertLess(dt.date.fromisoformat(d).weekday(), 5)


def _make_price_db(path, symbol="TEST", n=200, start_price=100.0, spike_at=None):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE px (date TEXT, symbol TEXT, close REAL, volume REAL,"
                 " PRIMARY KEY (symbol,date))")
    rows = []
    price = start_price
    for i in range(n):
        if spike_at is not None and i == spike_at:
            price *= 1.5
        rows.append((f"2020-{1 + i // 28:02d}-{1 + i % 28:02d}", symbol, price, 1_000_000.0))
        price *= 1.001
    conn.executemany("INSERT INTO px VALUES (?,?,?,?)", rows)
    conn.commit()
    conn.close()
    return [r[0] for r in rows]


class PriceBookTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "px.db")
        self.dates = _make_price_db(self.db)
        self.book = panel.PriceBook(self.db)

    def test_features_need_full_trailing_window(self):
        # Too early in the series: trailing vol window is not satisfied.
        self.assertIsNone(self.book.features("TEST", self.dates[10]))
        self.assertIsNotNone(self.book.features("TEST", self.dates[120]))

    def test_asof_lookup_never_sees_the_future(self):
        f = self.book.features("TEST", self.dates[120])
        # Spot must equal the bar at that date, not a later one.
        conn = sqlite3.connect(self.db)
        expect = conn.execute("SELECT close FROM px WHERE date=?", (self.dates[120],)).fetchone()[0]
        conn.close()
        self.assertAlmostEqual(f["spot"], expect, places=6)

    def test_forward_uses_path_max_not_endpoint(self):
        db2 = os.path.join(self.tmp, "px2.db")
        _make_price_db(db2, n=200, spike_at=130)
        book = panel.PriceBook(db2)
        f = book.features("TEST", self.dates[125])
        fwd = book.forward("TEST", f["_i"], 21)
        mx, end, mn = fwd.max_up, fwd.end, fwd.min_dn
        # A mid-path 50% spike must show in the max even though drift is tiny.
        self.assertGreater(mx, 0.4)
        self.assertGreater(mx, end - 0.001)

    def test_forward_returns_none_past_series_end(self):
        f = self.book.features("TEST", self.dates[190])
        self.assertIsNone(self.book.forward("TEST", f["_i"], 42))

    def test_publication_lag_shifts_the_entry_bar(self):
        # FINRA publishes ~8 business days after the settlement date, so entry
        # must sit that far forward or the study trades unpublished data.
        base = self.book.features("TEST", self.dates[120], lag=0)
        lagged = self.book.features("TEST", self.dates[120], lag=8)
        self.assertEqual(lagged["_i"], base["_i"] + 8)

    def test_lagged_entry_skips_a_squeeze_that_happened_before_publication(self):
        db2 = os.path.join(self.tmp, "px3.db")
        _make_price_db(db2, n=200, spike_at=124)
        book = panel.PriceBook(db2)
        # Spike lands 4 bars after settlement — inside the publication gap.
        no_lag = book.features("TEST", self.dates[120], lag=0)
        lagged = book.features("TEST", self.dates[120], lag=8)
        mx_no_lag = book.forward("TEST", no_lag["_i"], 21).max_up
        mx_lagged = book.forward("TEST", lagged["_i"], 21).max_up
        # Trading on unpublished data would have captured the move; a real
        # entry after publication would not.
        self.assertGreater(mx_no_lag, 0.4)
        self.assertLess(mx_lagged, 0.1)


class SharesLookupTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "s.db")
        conn = shares.connect(self.db)
        conn.executemany(
            "INSERT INTO shares_out(symbol,filed,end_date,shares) VALUES(?,?,?,?)",
            [("AAA", "2020-02-01", "2019-12-31", 1000.0),
             ("AAA", "2021-02-01", "2020-12-31", 2000.0)])
        conn.commit()
        conn.close()
        self.lookup = shares.SharesLookup(self.db)

    def test_uses_filed_date_not_period_end(self):
        # The 2020-12-31 figure was not public until 2021-02-01.
        self.assertEqual(self.lookup.get("AAA", "2021-01-15"), 1000.0)
        self.assertEqual(self.lookup.get("AAA", "2021-02-15"), 2000.0)

    def test_before_first_filing_is_none(self):
        self.assertIsNone(self.lookup.get("AAA", "2019-06-01"))

    def test_unknown_symbol_is_none(self):
        self.assertIsNone(self.lookup.get("ZZZ", "2021-01-01"))


def _panel_rows(n_dates=8, per_date=60, setup_edge=0.0, seed=3):
    """Synthetic panel; SETUP rows get *setup_edge* added to their z-score."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        date = f"2020-{d + 1:02d}-15"
        for i in range(per_date):
            is_setup = i < per_date // 3
            z = rng.normal(0, 1) + (setup_edge if is_setup else 0.0)
            rows.append({
                "date": date, "symbol": f"S{i}",
                "grade": "SETUP" if is_setup else "NONE",
                "points": 5 if is_setup else 1,
                "z_21": z, "max_21": z * 0.1, "end_21": z * 0.05, "min_21": -abs(z) * 0.05,
            })
    return rows


class StudyTest(unittest.TestCase):
    def test_tail_rate_counts_threshold_crossings(self):
        rows = [{"z_21": 1.0}, {"z_21": 2.5}, {"z_21": 3.0}]
        rate, n = study.tail_rate(rows, 21, k=2.0)
        self.assertEqual(n, 3)
        self.assertAlmostEqual(rate, 2 / 3)

    def test_no_edge_gives_ci_spanning_zero(self):
        rows = _panel_rows(n_dates=20, per_date=300, setup_edge=0.0)
        lb = study.lift_bootstrap(rows, 21, n_boot=400)
        self.assertLessEqual(lb["ci_lo"], 0.0)
        self.assertGreaterEqual(lb["ci_hi"], 0.0)
        self.assertGreater(lb["p_le_zero"], 0.05)

    def test_real_edge_is_detected(self):
        lb = study.lift_bootstrap(_panel_rows(setup_edge=1.5, per_date=200), 21, n_boot=300)
        self.assertGreater(lb["observed"], 0.0)
        self.assertLess(lb["p_le_zero"], 0.05)

    def test_bootstrap_resamples_dates_not_rows(self):
        # With one date, a date-clustered bootstrap cannot vary the sample, so
        # every draw reproduces the observed lift and the CI collapses.
        rows = _panel_rows(n_dates=1, per_date=300, setup_edge=1.0)
        lb = study.lift_bootstrap(rows, 21, n_boot=200)
        self.assertAlmostEqual(lb["ci_lo"], lb["ci_hi"], places=9)

    def test_describe_reports_counts(self):
        d = study.describe(_panel_rows(), 21)
        self.assertGreater(d["n"], 0)
        self.assertIn("p_2sig", d)


class GradeTest(unittest.TestCase):
    def _row(self, ret_5d):
        return {"si_ratio": 0.30, "dtc": 6.0, "trend": "rising",
                "ret_5d": ret_5d, "rvol": 2.0, "spot": 10.0}

    def test_ret5d_scale_now_controls_the_momentum_rule(self):
        # The fraction/percent mismatch that killed the old "late shorts" rule
        # applies to the momentum rule too: +12% arrives as 0.12 and cannot
        # clear a >= 10.0 threshold. as_written reproduces that dead behaviour;
        # as_intended reproduces the fixed adapter (detector._ret_5d_as_percent).
        dead = panel.grade([self._row(0.12)], ret5d_scale="as_written")[0]
        live = panel.grade([self._row(0.12)], ret5d_scale="as_intended")[0]
        self.assertEqual(live["points"], dead["points"] + 2)

    def test_negative_week_scores_nothing_either_way(self):
        # The "late shorts pressing" bonus is retired: a -15% week is shown as
        # evidence but never scored, whichever scale is used.
        dead = panel.grade([self._row(-0.15)], ret5d_scale="as_written")[0]
        live = panel.grade([self._row(-0.15)], ret5d_scale="as_intended")[0]
        self.assertEqual(live["points"], dead["points"])

    def test_si_scale_widens_the_gate(self):
        low = panel.grade([{"si_ratio": 0.12, "dtc": 6.0, "trend": "rising",
                            "ret_5d": None, "rvol": 2.0, "spot": 10.0}], si_scale=1.0)[0]
        high = panel.grade([{"si_ratio": 0.12, "dtc": 6.0, "trend": "rising",
                             "ret_5d": None, "rvol": 2.0, "spot": 10.0}], si_scale=1.5)[0]
        self.assertEqual(low["grade"], "NONE")       # 12% of shares out: below gate
        self.assertNotEqual(high["grade"], "NONE")   # 18% of assumed float: in

    def test_grade_uses_production_detector(self):
        # Guard against the harness drifting into its own copy of the rules.
        from src.squeeze import detector
        self.assertIs(panel.assess_squeeze, detector.assess_squeeze)


if __name__ == "__main__":
    unittest.main()
