"""Tests for src/pick_context.py — per-pick decision context (offline)."""
from __future__ import annotations

import json
import os
import sqlite3
import tempfile
import unittest

from src import pick_context as pc


def _db(path, closed=(), open_=()):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (ticker TEXT, strategy_name TEXT, status TEXT, "
                 "date TEXT, expiration TEXT, entry_delta REAL, pnl_pct REAL)")
    for (strategy, date, exp, delta, pnl) in closed:
        conn.execute("INSERT INTO trades VALUES ('XYZ',?,?,?,?,?,?)"
                     .replace("VALUES ('XYZ',?,?,?,?,?,?)",
                              "VALUES ('XYZ',?, 'CLOSED', ?, ?, ?, ?)"),
                     (strategy, date, exp, delta, pnl))
    for (ticker, strategy) in open_:
        conn.execute("INSERT INTO trades VALUES (?,?, 'OPEN', '2026-06-01', "
                     "'2026-07-17', 0.4, NULL)", (ticker, strategy))
    conn.commit(); conn.close()


def _closed(n, strategy="Long Call", dte=35, delta=0.40, pnl=0.10):
    return [(strategy, "2026-04-01",
             f"2026-05-{1+dte-30:02d}" if dte >= 30 else f"2026-04-{1+dte:02d}",
             delta, pnl) for _ in range(n)]


class AnalogStatsTest(unittest.TestCase):
    def setUp(self):
        pc.reset_caches()
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")

    def tearDown(self):
        pc.reset_caches()

    def test_matches_strategy_dte_delta(self):
        closed = (_closed(4, pnl=0.20) + _closed(2, pnl=-0.10)
                  + _closed(9, strategy="Long Put"))
        _db(self.db, closed=closed)
        a = pc.analog_stats("Long Call", dte=35, delta=0.40, db_path=self.db)
        self.assertIsNotNone(a)
        self.assertEqual(a["n"], 6)
        self.assertAlmostEqual(a["win_rate"], 4 / 6)
        self.assertFalse(a["widened"])

    def test_widens_when_delta_match_thin(self):
        # 6 trades at delta 0.70 (outside ±0.10 of 0.40) -> widened match
        _db(self.db, closed=_closed(6, delta=0.70))
        a = pc.analog_stats("Long Call", dte=35, delta=0.40, db_path=self.db)
        self.assertIsNotNone(a)
        self.assertTrue(a["widened"])

    def test_none_when_too_few(self):
        _db(self.db, closed=_closed(3))
        self.assertIsNone(pc.analog_stats("Long Call", 35, 0.40, db_path=self.db))

    def test_dte_band_excludes_far(self):
        _db(self.db, closed=_closed(6, dte=7))   # 7 DTE vs pick 35 -> excluded
        self.assertIsNone(pc.analog_stats("Long Call", 35, 0.40, db_path=self.db))


class EventsInWindowTest(unittest.TestCase):
    def test_fomc_inside_window(self):
        evs = pc.events_in_window("2026-06-26", today="2026-06-11")
        names = {(e["name"], e["date"]) for e in evs}
        self.assertIn(("FOMC", "2026-06-17"), names)

    def test_nothing_after_expiry(self):
        evs = pc.events_in_window("2026-06-12", today="2026-06-12")
        self.assertEqual([e for e in evs if e["date"] > "2026-06-12"], [])


class CohortEligibleTest(unittest.TestCase):
    def test_long_call_30dte_eligible(self):
        self.assertTrue(pc.cohort_eligible("Long Call", 36))

    def test_short_dte_not(self):
        self.assertFalse(pc.cohort_eligible("Long Call", 18))

    def test_puts_never(self):
        self.assertFalse(pc.cohort_eligible("Long Put", 45))


class OpenBookTest(unittest.TestCase):
    def setUp(self):
        pc.reset_caches()
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")

    def tearDown(self):
        pc.reset_caches()

    def test_open_positions_listed(self):
        _db(self.db, open_=[("IWM", "Long Call"), ("IWM", "Long Call"),
                            ("SPY", "Iron Condor")])
        self.assertEqual(pc.open_book("IWM", db_path=self.db),
                         ["Long Call", "Long Call"])
        self.assertEqual(pc.open_book("TSLA", db_path=self.db), [])


class InsiderCacheTest(unittest.TestCase):
    def test_cache_hit_no_fetch(self):
        tmp = tempfile.mkdtemp()
        cache = os.path.join(tmp, "cache.json")
        from datetime import datetime
        today = datetime.now().strftime("%Y-%m-%d")
        with open(cache, "w") as f:
            json.dump({"AAPL": {"as_of": today,
                                "summary": {"label": "CLUSTER BUY",
                                            "buy_value": 500000.0}}}, f)
        s = pc.insider_summary("AAPL", cache_path=cache, fetch=False)
        self.assertEqual(s["label"], "CLUSTER BUY")

    def test_no_cache_no_fetch_returns_none(self):
        tmp = tempfile.mkdtemp()
        self.assertIsNone(pc.insider_summary(
            "AAPL", cache_path=os.path.join(tmp, "c.json"), fetch=False))


class ContextLinesTest(unittest.TestCase):
    def setUp(self):
        pc.reset_caches()
        self.tmp = tempfile.mkdtemp()
        self.db = os.path.join(self.tmp, "t.db")

    def tearDown(self):
        pc.reset_caches()

    def test_lines_for_cohort_call_with_history(self):
        _db(self.db, closed=_closed(8, pnl=0.15),
            open_=[("IWM", "Long Call")])
        row = {"type": "call", "T_years": 36 / 365, "delta": 0.40,
               "symbol": "IWM", "expiration": "2026-07-17"}
        lines = pc.context_lines(row, today="2026-06-11", db_path=self.db,
                                 with_insider=False)
        joined = "\n".join(lines)
        self.assertIn("History:", joined)
        self.assertIn("n=8", joined)
        self.assertIn("COHORT ✓", joined)
        self.assertIn("FOMC 06-17", joined)
        self.assertIn("already holding", joined)

    def test_never_raises_on_garbage(self):
        self.assertIsInstance(
            pc.context_lines({}, db_path="/no/such.db", with_insider=False), list)


class QuantReadTest(unittest.TestCase):
    def test_positive_ev_cheap_surface(self):
        line = pc.quant_read_line({
            "ev_per_contract": 42.0, "ev_gross_per_contract": 60.0,
            "ev_cost_per_contract": 18.0, "iv_surface_residual": -0.05,
            "vrp_regime": "RICH"})
        self.assertIn("POSITIVE EV after cost", line)
        self.assertIn("CHEAP vs surface", line)
        self.assertIn("VRP: RICH", line)

    def test_negative_ev_flagged(self):
        line = pc.quant_read_line({"ev_per_contract": -15.0,
                                   "ev_gross_per_contract": -3.0,
                                   "ev_cost_per_contract": 12.0,
                                   "iv_surface_residual": 0.04})
        self.assertIn("NEGATIVE EV after cost", line)
        self.assertIn("RICH vs surface", line)

    def test_marginal_ev_when_cost_eats_edge(self):
        # gross +38, cost 36 -> net +2: surviving edge << toll paid -> MARGINAL
        line = pc.quant_read_line({"ev_per_contract": 2.0, "ev_gross_per_contract": 38.0,
                                   "ev_cost_per_contract": 36.0})
        self.assertIn("MARGINAL EV", line)
        self.assertNotIn("POSITIVE EV after cost", line)

    def test_solid_positive_when_edge_beats_cost(self):
        line = pc.quant_read_line({"ev_per_contract": 216.0, "ev_gross_per_contract": 220.0,
                                   "ev_cost_per_contract": 4.0})
        self.assertIn("POSITIVE EV after cost", line)

    def test_none_without_ev(self):
        self.assertIsNone(pc.quant_read_line({}))
        self.assertIsNone(pc.quant_read_line({"ev_per_contract": float("nan")}))

    def test_unknown_vrp_omitted(self):
        line = pc.quant_read_line({"ev_per_contract": 10.0, "vrp_regime": "UNKNOWN"})
        self.assertNotIn("VRP", line)


class NetEvTest(unittest.TestCase):
    def test_round_trip_deducts_full_spread_and_two_commissions(self):
        from src.trade_analysis import net_ev_per_contract
        # gross edge 0.50/share -> $50; spread 10% of $2 prem over 2 sides = $20; comm 2*0.65=1.30
        ev = net_ev_per_contract(0.50, 2.0, 0.10, commission_per_contract=0.65)
        self.assertAlmostEqual(ev, 50.0 - 20.0 - 1.30, places=6)

    def test_round_trip_costs_more_than_one_way(self):
        from src.trade_analysis import net_ev_per_contract
        rt = net_ev_per_contract(0.50, 2.0, 0.10, round_trip=True)
        ow = net_ev_per_contract(0.50, 2.0, 0.10, round_trip=False)
        self.assertLess(rt, ow)


if __name__ == "__main__":
    unittest.main()
