"""tests/test_gate_rd_test.py

Pure computation and temp-sqlite-fixture tests only — data/candidates.db is
never opened here.
"""
from __future__ import annotations

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.gate_rd_test import (  # noqa: E402
    CUTOFF,
    relative_spread,
    running_variable,
)


def _bull_put_row(short_bid, short_ask, long_bid, long_ask, **extra):
    blob = {
        "short_bid": short_bid, "short_ask": short_ask,
        "long_bid": long_bid, "long_ask": long_ask,
        "entry_delta": extra.get("entry_delta", -0.20),
    }
    return {
        "strategy_name": "Bull Put",
        "round_trip_pct": extra.get("round_trip_pct"),
        "features_json": json.dumps(blob),
    }


class RunningVariableTests(unittest.TestCase):
    def test_uses_stored_value_when_present(self):
        row = _bull_put_row(1.0, 1.1, 0.4, 0.5, round_trip_pct=0.18)
        rtp, status = running_variable(row)
        self.assertEqual(status, "measured")
        self.assertAlmostEqual(rtp, 0.18)

    def test_recomputes_from_features_json_when_stored_is_null(self):
        # short 1.00/1.10 sell, long 0.40/0.50 buy: mid credit = 1.05-0.45=0.60
        # crossed (sell at bid, buy at ask): 1.00-0.50=0.50; slip=0.10; round trip 2*0.10/0.60=0.333..
        row = _bull_put_row(1.00, 1.10, 0.40, 0.50)
        rtp, status = running_variable(row)
        self.assertEqual(status, "recomputed")
        self.assertGreater(rtp, CUTOFF)  # this one refuses on friction

    def test_credit_gone_is_reported_and_excluded(self):
        # candidate_verdict checks credit_gone against the "limit" fill
        # (35% of half-spread conceded, execution_truth.DEFAULT_LIMIT_K),
        # not a full cross. short 0.50/1.50 (mid=1.00, half=0.50), long
        # 0.50/1.30 (mid=0.90, half=0.40): mid credit = 1.00-0.90 = 0.10
        # (is_credit=True), but limit net_reward = 0.10 - 0.35*(0.50+0.40)
        # = -0.215 <= 0 -> the credit evaporates once actually filled.
        row = _bull_put_row(0.50, 1.50, 0.50, 1.30)
        rtp, status = running_variable(row)
        self.assertEqual(status, "credit_gone")
        self.assertIsNone(rtp)

    def test_unpriceable_when_a_leg_is_unquoted(self):
        blob = {"short_bid": 1.0, "short_ask": 1.1, "entry_delta": -0.2}
        row = {"strategy_name": "Bull Put", "round_trip_pct": None,
              "features_json": json.dumps(blob)}
        rtp, status = running_variable(row)
        self.assertEqual(status, "unpriceable")
        self.assertIsNone(rtp)


class RelativeSpreadTests(unittest.TestCase):
    def test_computed_from_short_leg_quote(self):
        row = _bull_put_row(1.00, 1.10, 0.40, 0.50)
        # short mid = 1.05, spread = 0.10 -> relative = 0.10/1.05
        rs = relative_spread(row)
        self.assertAlmostEqual(rs, 0.10 / 1.05, places=6)

    def test_none_when_short_leg_unquoted(self):
        row = {"strategy_name": "Bull Put",
              "features_json": json.dumps({"long_bid": 0.4, "long_ask": 0.5})}
        self.assertIsNone(relative_spread(row))


import sqlite3
import tempfile

from scripts.gate_rd_test import attach_outcome, fetch_band_rows


def _make_candidates_db(path, rows):
    """rows: list of dicts with keys rowid(implicit), contract_key, symbol,
    ts, expiration, strategy_name, round_trip_pct, features_json."""
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE candidates (
        contract_key TEXT, symbol TEXT, ts TEXT, expiration TEXT,
        strategy_name TEXT, round_trip_pct REAL, features_json TEXT)""")
    for r in rows:
        con.execute(
            "INSERT INTO candidates (contract_key, symbol, ts, expiration, "
            "strategy_name, round_trip_pct, features_json) VALUES "
            "(:contract_key,:symbol,:ts,:expiration,:strategy_name,"
            ":round_trip_pct,:features_json)", r)
    con.execute("CREATE TABLE candidate_marks (contract_key TEXT, "
               "mark_date TEXT, bid REAL, ask REAL, mid REAL, source TEXT)")
    con.commit()
    con.close()


def _bull_put(contract_key, symbol, ts, expiration, short_bid, short_ask,
              long_bid, long_ask, entry_delta=-0.20, round_trip_pct=None):
    import json as _json
    blob = {"short_bid": short_bid, "short_ask": short_ask,
           "long_bid": long_bid, "long_ask": long_ask,
           "entry_delta": entry_delta}
    return dict(contract_key=contract_key, symbol=symbol, ts=ts,
               expiration=expiration, strategy_name="Bull Put",
               round_trip_pct=round_trip_pct, features_json=_json.dumps(blob))


class FetchBandRowsTests(unittest.TestCase):
    def test_restricts_to_bandwidth_and_excludes_credit_gone(self):
        with tempfile.TemporaryDirectory() as d:
            db = f"{d}/candidates.db"
            # Verified against the real verdict_for: for a 2-leg spread the
            # cross-vs-mid slippage is always short_half + long_half, so
            # round_trip_pct = 2*(short_half+long_half)/credit regardless of
            # price level. near: credit=1.00, halves 0.05+0.05 -> 0.20 (in
            # the +/-0.10 band, x=-0.05). far: credit=1.00, halves
            # 0.50+0.50 -> 2.00 (nowhere near the band).
            near = _bull_put("k1", "AAPL", "2026-08-19T10:00:00Z",
                             "2026-09-05", 1.45, 1.55, 0.45, 0.55)
            far = _bull_put("k2", "MSFT", "2026-08-19T10:00:00Z",
                            "2026-09-05", 1.00, 2.00, 0.00, 1.00)
            # credit_gone: mid credit 1.00-0.90=0.10 (is_credit), but the
            # limit fill concedes 0.35*(0.50+0.40)=0.315 > 0.10 of it.
            gone = _bull_put("k3", "NVDA", "2026-08-19T10:00:00Z",
                             "2026-09-05", 0.50, 1.50, 0.50, 1.30)
            _make_candidates_db(db, [near, far, gone])
            rows = fetch_band_rows(db, bandwidth=0.10)
            keys = {r["contract_key"] for r in rows}
            self.assertEqual(keys, {"k1"})  # far is out of band, gone is excluded
            self.assertAlmostEqual(rows[0]["x"], -0.05, places=6)

    def test_row_carries_symbol_day_delta_dte_entry_signed(self):
        with tempfile.TemporaryDirectory() as d:
            db = f"{d}/candidates.db"
            # ts at midnight so dte is a clean whole number (SQLite's
            # julianday() keeps the time-of-day component: a 10:00:00 ts
            # against a bare-date expiration gives 9.58d, not 10.0d).
            row = _bull_put("k1", "AAPL", "2026-08-19T00:00:00Z",
                            "2026-08-29", 1.00, 1.10, 0.40, 0.50,
                            entry_delta=-0.22)
            _make_candidates_db(db, [row])
            rows = fetch_band_rows(db, bandwidth=1.0)
            self.assertEqual(len(rows), 1)
            r = rows[0]
            self.assertEqual(r["symbol"], "AAPL")
            self.assertEqual(r["day"], "2026-08-19")
            self.assertAlmostEqual(r["abs_delta"], 0.22)
            self.assertAlmostEqual(r["dte"], 10.0)
            self.assertIsNotNone(r["entry_signed"])
            self.assertAlmostEqual(r["rel_spread"], 0.10 / 1.05, places=6)


class AttachOutcomeTests(unittest.TestCase):
    def test_uses_first_mark_at_or_after_horizon(self):
        with tempfile.TemporaryDirectory() as d:
            db = f"{d}/candidates.db"
            row = _bull_put("k1", "AAPL", "2026-08-19T10:00:00Z",
                            "2026-09-05", 1.00, 1.10, 0.40, 0.50)
            _make_candidates_db(db, [row])
            con = sqlite3.connect(db)
            # marks at day+3 (too early for 5d horizon) and day+6 (qualifies)
            con.execute("INSERT INTO candidate_marks VALUES "
                       "('k1','2026-08-22',0.5,0.6,0.55,'chain')")
            con.execute("INSERT INTO candidate_marks VALUES "
                       "('k1','2026-08-25',0.3,0.4,0.35,'chain')")
            con.commit(); con.close()

            rows = fetch_band_rows(db, bandwidth=1.0)
            out = attach_outcome(rows, db, horizon_days=5)
            self.assertEqual(len(out), 1)
            self.assertIsNotNone(out[0]["outcome"])

    def test_no_qualifying_mark_gives_none_outcome(self):
        with tempfile.TemporaryDirectory() as d:
            db = f"{d}/candidates.db"
            row = _bull_put("k1", "AAPL", "2026-08-19T10:00:00Z",
                            "2026-09-05", 1.00, 1.10, 0.40, 0.50)
            _make_candidates_db(db, [row])
            con = sqlite3.connect(db)
            con.execute("INSERT INTO candidate_marks VALUES "
                       "('k1','2026-08-21',0.5,0.6,0.55,'chain')")  # only 2d out
            con.commit(); con.close()

            rows = fetch_band_rows(db, bandwidth=1.0)
            out = attach_outcome(rows, db, horizon_days=5)
            self.assertIsNone(out[0]["outcome"])


import random

from scripts.gate_rd_test import (
    cluster_bootstrap_rd,
    collapse_to_clusters,
    local_linear_intercept,
    rd_estimate,
)


class CollapseToClustersTests(unittest.TestCase):
    def test_one_point_per_symbol_day_per_side(self):
        rows = [
            {"symbol": "AAPL", "day": "2026-08-19", "x": -0.02, "outcome": 0.10},
            {"symbol": "AAPL", "day": "2026-08-19", "x": -0.03, "outcome": 0.20},
            {"symbol": "AAPL", "day": "2026-08-19", "x": 0.01, "outcome": -0.05},
            {"symbol": "MSFT", "day": "2026-08-19", "x": -0.01, "outcome": 0.30},
        ]
        below, above = collapse_to_clusters(rows, "outcome")
        self.assertEqual(len(below), 2)  # (AAPL, 2026-08-19) and (MSFT, 2026-08-19)
        self.assertEqual(len(above), 1)  # (AAPL, 2026-08-19) above side
        aapl_below = [p for p in below if abs(p[0] - (-0.025)) < 1e-9]
        self.assertEqual(len(aapl_below), 1)
        self.assertAlmostEqual(aapl_below[0][1], 0.15)  # mean of 0.10, 0.20

    def test_none_outcomes_are_dropped(self):
        rows = [
            {"symbol": "AAPL", "day": "2026-08-19", "x": -0.02, "outcome": None},
            {"symbol": "AAPL", "day": "2026-08-19", "x": -0.02, "outcome": 0.10},
        ]
        below, above = collapse_to_clusters(rows, "outcome")
        self.assertEqual(len(below), 1)
        self.assertAlmostEqual(below[0][1], 0.10)


class LocalLinearInterceptTests(unittest.TestCase):
    def test_recovers_known_line(self):
        points = [(x, 2.0 + 3.0 * x) for x in (-0.05, -0.03, -0.01, 0.0, 0.02)]
        self.assertAlmostEqual(local_linear_intercept(points), 2.0, places=6)

    def test_single_point_returns_its_own_value(self):
        self.assertAlmostEqual(local_linear_intercept([(-0.02, 0.5)]), 0.5)


class RdEstimateTests(unittest.TestCase):
    def test_recovers_a_known_jump(self):
        below = [(x, 0.10 + 2.0 * x) for x in (-0.05, -0.03, -0.01)]
        above = [(x, -0.05 + 2.0 * x) for x in (0.01, 0.03, 0.05)]  # jump = -0.15
        self.assertAlmostEqual(rd_estimate(below, above), -0.15, places=6)

    def test_no_jump_gives_zero(self):
        below = [(x, 0.10 + 2.0 * x) for x in (-0.05, -0.03, -0.01)]
        above = [(x, 0.10 + 2.0 * x) for x in (0.01, 0.03, 0.05)]
        self.assertAlmostEqual(rd_estimate(below, above), 0.0, places=6)


class ClusterBootstrapRdTests(unittest.TestCase):
    def test_confident_jump_clears_the_hurdle(self):
        # Many clusters, a real jump, WITH jitter (constant/noiseless y would
        # make every bootstrap resample fit the same exact line, giving zero
        # resampling variance and therefore se=0 -> t=None instead of a
        # clearly-significant t; verified empirically before writing this).
        gen = random.Random(7)
        below = [(-0.05 + 0.001 * i, 0.20 + gen.uniform(-0.03, 0.03))
                for i in range(60)]
        above = [(0.001 * i, -0.10 + gen.uniform(-0.03, 0.03))
                for i in range(60)]
        point, lo, hi, t = cluster_bootstrap_rd(below, above, n_boot=500, seed=1)
        self.assertAlmostEqual(point, -0.30, places=1)
        self.assertIsNotNone(t)
        self.assertLess(t, -3.0)
        self.assertLess(hi, 0.0)  # CI excludes zero, same direction as point

    def test_too_few_clusters_returns_no_tstat(self):
        below = [(-0.02, 0.1)]
        above = [(0.02, 0.1)]
        point, lo, hi, t = cluster_bootstrap_rd(below, above, n_boot=100, seed=1)
        self.assertIsNone(t)


if __name__ == "__main__":
    unittest.main()
