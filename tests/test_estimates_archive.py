"""The analyst-estimates archive. No network: the fetcher is injected.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_estimates_archive -v
"""
import os
import sqlite3
import tempfile
import unittest

import pandas as pd

from src import estimates_archive as ea


def _data(eps_now=1.97, rev_up=1):
    trend = pd.DataFrame({"current": [eps_now, 8.8], "7daysAgo": [1.95, 8.8]},
                         index=pd.Index(["0q", "0y"], name="period"))
    revisions = pd.DataFrame({"upLast7days": [rev_up], "downLast7Days": [0]},
                             index=pd.Index(["0q"], name="period"))
    recs = pd.DataFrame({"period": ["0m", "-1m"], "strongBuy": [6, 6],
                         "buy": [19, 18]})
    actions = pd.DataFrame(
        {"Firm": ["JP Morgan", "Needham"], "ToGrade": ["Overweight", "Hold"],
         "FromGrade": ["Overweight", "Hold"], "Action": ["main", "reit"],
         "priceTargetAction": ["Raises", ""],
         "currentPriceTarget": [360.0, 0.0], "priorPriceTarget": [340.0, 0.0]},
        index=pd.DatetimeIndex(["2026-10-12 18:47:46", "2026-10-01 11:00:30"],
                               name="GradeDate"))
    return {"eps_trend": trend, "eps_revisions": revisions,
            "earnings_estimate": None, "revenue_estimate": pd.DataFrame(),
            "recommendations": recs, "upgrades_downgrades": actions}


class TestArchive(unittest.TestCase):
    def _db(self, d):
        return os.path.join(d, "e.db")

    def test_snapshot_is_stored_long_and_keeps_raw_field_names(self):
        with tempfile.TemporaryDirectory() as d:
            out = ea.archive_symbols(["AAPL"], db_path=self._db(d),
                                     today="2026-10-14", pause=0,
                                     fetch=lambda s: _data())
            self.assertTrue(out["complete"])
            self.assertEqual(out["with_estimates"], 1)
            with sqlite3.connect(self._db(d)) as conn:
                v, = conn.execute(
                    "select value from estimate_snapshots where dataset="
                    "'eps_trend' and period='0q' and field='current'").fetchone()
                fields = {r[0] for r in conn.execute(
                    "select field from estimate_snapshots")}
                periods = {r[0] for r in conn.execute(
                    "select period from estimate_snapshots "
                    "where dataset='recommendations'")}
            self.assertEqual(v, 1.97)
            self.assertIn("downLast7Days", fields)
            self.assertEqual(periods, {"0m", "-1m"})
            self.assertNotIn("period", fields)

    def test_one_snapshot_per_day_and_a_new_day_adds_rows(self):
        with tempfile.TemporaryDirectory() as d:
            for day, eps in (("2026-10-14", 1.97), ("2026-10-14", 1.98),
                             ("2026-10-15", 2.01)):
                ea.archive_symbols(["AAPL"], db_path=self._db(d), today=day,
                                   pause=0, fetch=lambda s, e=eps: _data(e))
            with sqlite3.connect(self._db(d)) as conn:
                rows = conn.execute(
                    "select snap_date, value from estimate_snapshots where "
                    "dataset='eps_trend' and period='0q' and field='current' "
                    "order by snap_date").fetchall()
            self.assertEqual(rows, [("2026-10-14", 1.98), ("2026-10-15", 2.01)])

    def test_analyst_actions_keep_the_first_time_they_were_seen(self):
        from datetime import datetime, timezone
        with tempfile.TemporaryDirectory() as d:
            t1 = datetime(2026, 10, 14, 21, tzinfo=timezone.utc)
            t2 = datetime(2026, 10, 15, 21, tzinfo=timezone.utc)
            a = ea.archive_symbols(["AAPL"], db_path=self._db(d), pause=0,
                                   now=t1, fetch=lambda s: _data())
            b = ea.archive_symbols(["AAPL"], db_path=self._db(d), pause=0,
                                   now=t2, fetch=lambda s: _data())
            self.assertEqual(a["new_actions"], 2)
            self.assertEqual(b["new_actions"], 0)
            with sqlite3.connect(self._db(d)) as conn:
                seen = {r[0] for r in conn.execute(
                    "select first_seen from analyst_actions")}
            self.assertEqual(seen, {t1.isoformat()})

    def test_an_etf_with_no_estimates_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as d:
            out = ea.archive_symbols(["SPY"] * 30, db_path=self._db(d), pause=0,
                                     fetch=lambda s: {})
            self.assertTrue(out["complete"])
            self.assertEqual(out["with_estimates"], 0)

    def test_a_dead_feed_stops_and_reports_incomplete(self):
        calls = []

        def dead(s):
            calls.append(s)
            raise RuntimeError("Too Many Requests")

        with tempfile.TemporaryDirectory() as d:
            out = ea.archive_symbols([f"T{i}" for i in range(50)],
                                     db_path=self._db(d), pause=0, fetch=dead)
        self.assertFalse(out["complete"])
        self.assertLess(len(calls), 50)

    def test_one_bad_symbol_does_not_cost_the_rest(self):
        def flaky(s):
            if s == "BAD":
                raise RuntimeError("boom")
            return _data()

        with tempfile.TemporaryDirectory() as d:
            out = ea.archive_symbols(["AAPL", "BAD", "MSFT"],
                                     db_path=self._db(d), pause=0, fetch=flaky)
        self.assertTrue(out["complete"])
        self.assertEqual((out["failed"], out["with_estimates"]), (1, 2))


class TestDue(unittest.TestCase):
    def test_after_the_close_on_a_trading_day_once(self):
        self.assertFalse(ea.due_estimates_archive({}, "2026-10-14", 1559))
        self.assertTrue(ea.due_estimates_archive({}, "2026-10-14", 1600))
        self.assertFalse(ea.due_estimates_archive(
            {"last_estimates_archive": "2026-10-14"}, "2026-10-14", 1700))

    def test_never_on_a_weekend_or_holiday(self):
        self.assertFalse(ea.due_estimates_archive({}, "2026-10-17", 1700))
        self.assertFalse(ea.due_estimates_archive({}, "2026-09-07", 1700))


class TestUniverse(unittest.TestCase):
    def test_watchlists_plus_recent_board_symbols(self):
        import json
        from datetime import datetime, timezone
        from src import candidate_record as cr
        with tempfile.TemporaryDirectory() as d:
            cfg = os.path.join(d, "cfg.json")
            with open(cfg, "w") as fh:
                json.dump({"watchlists": {"a": ["aapl", "MSFT"], "b": ["MSFT"]}}, fh)
            cdb = os.path.join(d, "c.db")
            with cr.connect(cdb) as conn:
                conn.execute(
                    "insert into candidates (scan_id, ts, board, contract_key,"
                    " symbol) values ('S', ?, 'b', 'k', 'NVDA')",
                    (datetime.now(timezone.utc).isoformat(),))
                conn.execute(
                    "insert into candidates (scan_id, ts, board, contract_key,"
                    " symbol) values ('S0', '2020-01-01', 'b', 'k', 'OLD')")
                conn.commit()
            self.assertEqual(ea.universe(cfg, candidates_db=cdb),
                             ["AAPL", "MSFT", "NVDA"])


if __name__ == "__main__":
    unittest.main()
