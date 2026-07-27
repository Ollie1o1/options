import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import sqlite3
import tempfile
import unittest
from datetime import datetime

from src.structure import margins as M
from src.structure.types import StructureMargin


def _make_db(rows):
    """rows: list of (strategy_name, pnl_usd, date_str). Returns a temp db path."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (strategy_name TEXT, pnl_usd REAL, "
                 "status TEXT, date TEXT)")
    conn.executemany("INSERT INTO trades VALUES (?,?, 'CLOSED', ?)", rows)
    conn.commit()
    conn.close()
    return path


class TestBreakeven(unittest.TestCase):
    def test_symmetric_payoff_needs_half(self):
        self.assertAlmostEqual(M.breakeven_hit(100.0, 100.0), 0.5)

    def test_big_wins_small_losses_need_little(self):
        # avg win 815, avg loss 242 -> B/E 22.9% (the real Long Put numbers)
        self.assertAlmostEqual(M.breakeven_hit(815.0, 242.0), 0.229, places=3)

    def test_small_wins_big_losses_need_a_lot(self):
        self.assertAlmostEqual(M.breakeven_hit(23.0, 27.0), 0.540, places=3)

    def test_degenerate_returns_none(self):
        self.assertIsNone(M.breakeven_hit(0.0, 0.0))


class TestLeagueTable(unittest.TestCase):
    def test_sufficient_evidence_is_active(self):
        rows = ([("Bull Put", 116.0, "2026-07-01")] * 12 +
                [("Bull Put", -70.0, "2026-07-01")] * 10)
        t = M.compute_league_table(_make_db(rows), now=datetime(2026, 7, 27))
        m = t["Bull Put"]
        self.assertEqual(m.n, 22)
        self.assertEqual(m.wins, 12)
        self.assertEqual(m.losses, 10)
        self.assertNotEqual(m.state, "UNPROVEN")

    def test_too_few_wins_is_unproven_even_with_big_n(self):
        # n=30 but only 5 wins -> cannot estimate avg_win
        rows = ([("Long Call", 700.0, "2026-07-01")] * 5 +
                [("Long Call", -600.0, "2026-07-01")] * 25)
        t = M.compute_league_table(_make_db(rows), now=datetime(2026, 7, 27))
        self.assertEqual(t["Long Call"].state, "UNPROVEN")
        self.assertEqual(t["Long Call"].n, 30)

    def test_too_few_losses_is_unproven(self):
        rows = ([("Bear Call", 23.0, "2026-07-01")] * 25 +
                [("Bear Call", -27.0, "2026-07-01")] * 3)
        t = M.compute_league_table(_make_db(rows), now=datetime(2026, 7, 27))
        self.assertEqual(t["Bear Call"].state, "UNPROVEN")

    def test_window_excludes_old_trades(self):
        rows = ([("Bull Put", 116.0, "2020-01-01")] * 12 +
                [("Bull Put", -70.0, "2020-01-01")] * 10)
        t = M.compute_league_table(_make_db(rows), window_days=90,
                                   now=datetime(2026, 7, 27))
        self.assertNotIn("Bull Put", t)

    def test_missing_db_returns_empty_table(self):
        self.assertEqual(M.compute_league_table("/nonexistent/x.db"), {})


class TestBenching(unittest.TestCase):
    def _margin(self, name, margin, state="ACTIVE"):
        return StructureMargin(strategy=name, n=40, wins=20, losses=20,
                               avg_win=100.0, avg_loss=100.0,
                               breakeven_hit=0.5, realized_hit=0.5 + margin,
                               margin=margin, state=state, ci_lo=margin - 0.1,
                               ci_hi=margin + 0.1)

    def test_one_bad_week_does_not_bench(self):
        table = {"Long Call": self._margin("Long Call", -0.13)}
        history = [{"date": "2026-07-20", "strategy": "Long Call",
                    "margin": 0.02}]
        out = M.apply_states(table, history, "2026-07-27")
        self.assertEqual(out["Long Call"].state, "ACTIVE")

    def test_two_consecutive_bad_weeks_bench(self):
        table = {"Long Call": self._margin("Long Call", -0.13)}
        history = [{"date": "2026-07-20", "strategy": "Long Call",
                    "margin": -0.05}]
        out = M.apply_states(table, history, "2026-07-27")
        self.assertEqual(out["Long Call"].state, "BENCHED")

    def test_one_good_week_does_not_unbench(self):
        table = {"Long Call": self._margin("Long Call", 0.04)}
        history = [{"date": "2026-07-13", "strategy": "Long Call",
                    "margin": -0.20, "state": "BENCHED"},
                   {"date": "2026-07-20", "strategy": "Long Call",
                    "margin": -0.18, "state": "BENCHED"}]
        out = M.apply_states(table, history, "2026-07-27")
        self.assertEqual(out["Long Call"].state, "BENCHED")

    def test_two_good_weeks_unbench(self):
        table = {"Long Call": self._margin("Long Call", 0.04)}
        history = [{"date": "2026-07-13", "strategy": "Long Call",
                    "margin": -0.18, "state": "BENCHED"},
                   {"date": "2026-07-20", "strategy": "Long Call",
                    "margin": 0.01, "state": "BENCHED"}]
        out = M.apply_states(table, history, "2026-07-27")
        self.assertEqual(out["Long Call"].state, "ACTIVE")

    def test_unproven_is_never_benched(self):
        table = {"Iron Condor": self._margin("Iron Condor", -0.30,
                                             state="UNPROVEN")}
        history = [{"date": "2026-07-20", "strategy": "Iron Condor",
                    "margin": -0.40}]
        out = M.apply_states(table, history, "2026-07-27")
        self.assertEqual(out["Iron Condor"].state, "UNPROVEN")

    def test_snapshot_roundtrip(self):
        fd, path = tempfile.mkstemp(suffix=".tsv")
        os.close(fd)
        os.remove(path)
        table = {"Bull Put": self._margin("Bull Put", 0.287)}
        M.append_snapshot(path, table, "2026-07-27")
        hist = M.load_history(path)
        self.assertEqual(len(hist), 1)
        self.assertEqual(hist[0]["strategy"], "Bull Put")
        self.assertAlmostEqual(hist[0]["margin"], 0.287, places=3)
