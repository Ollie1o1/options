"""The daily marking run, end to end.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_candidate_marks_run -v

No network: the quote fetcher is injected. No real ledger, database or config.
"""
import os
import sqlite3
import tempfile
import unittest

from src import candidate_marks as cm
from tests.test_candidate_marks import _insert_candidate, _write_config


class TestDue(unittest.TestCase):
    def test_due_once_per_day(self):
        self.assertTrue(cm.due_candidate_marks({}, "2026-08-19"))
        self.assertFalse(cm.due_candidate_marks(
            {"last_candidate_marks": "2026-08-19"}, "2026-08-19"))
        self.assertTrue(cm.due_candidate_marks(
            {"last_candidate_marks": "2026-08-18"}, "2026-08-19"))

    def test_a_weekend_is_never_due(self):
        # 2026-08-29/30: the scheduler marked on a Saturday and a Sunday and
        # resolved 33,286 simulated exits against weekend quotes.
        self.assertFalse(cm.due_candidate_marks({}, "2026-08-29"))
        self.assertFalse(cm.due_candidate_marks(
            {"last_candidate_marks": "2026-08-28"}, "2026-08-30"))

    def test_a_market_holiday_is_never_due(self):
        self.assertFalse(cm.due_candidate_marks({}, "2026-09-07"))

    def test_the_session_after_a_weekend_is_due(self):
        self.assertTrue(cm.due_candidate_marks(
            {"last_candidate_marks": "2026-08-28"}, "2026-08-31"))

    def test_a_missing_state_is_due(self):
        self.assertTrue(cm.due_candidate_marks(None, "2026-08-19"))


class TestMarkCandidates(unittest.TestCase):
    def test_the_run_opens_marks_and_reports_counts(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)
            out = cm.mark_candidates(
                db_path=path, today="2026-08-19", cfg_path=cfg,
                fetch=lambda t, e: {(190.0, "call"): (11.0, 11.4)})
            self.assertEqual(out["opened"], 1)
            self.assertEqual(out["marked"], 1)
            self.assertEqual(out["closed"], 0)

    def test_a_position_opened_today_is_marked_today(self):
        # Opening must happen before marking, or a candidate waits a day for
        # its first mark and every entry-to-first-mark gap is silently wrong.
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)
            cm.mark_candidates(db_path=path, today="2026-08-19", cfg_path=cfg,
                               fetch=lambda t, e: {(190.0, "call"): (11.0, 11.4)})
            with sqlite3.connect(path) as conn:
                n, = conn.execute(
                    "select count(*) from candidate_marks").fetchone()
            self.assertEqual(n, 1)

    def test_the_run_closes_a_position_that_hit_its_target(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)
            cm.mark_candidates(db_path=path, today="2026-08-01", cfg_path=cfg,
                               fetch=lambda t, e: {(190.0, "call"): (9.9, 10.1)})
            out = cm.mark_candidates(
                db_path=path, today="2026-08-10", cfg_path=cfg,
                fetch=lambda t, e: {(190.0, "call"): (24.9, 25.1)})
            self.assertEqual(out["closed"], 1)
            with sqlite3.connect(path) as conn:
                reason, = conn.execute(
                    "select exit_reason from candidate_positions").fetchone()
            self.assertEqual(reason, "take_profit")

    def test_a_structure_priced_at_exactly_zero_net_credit_still_gets_marked(self):
        # Found 2026-09-09: a live CMCSA Bear Call whose short and long legs
        # had converged to the same mid (both 0.075) priced at net credit
        # exactly $0.00 — a real, meaningful value — and `if not mid: continue`
        # silently and permanently dropped it, identical to an unpriceable
        # structure. Reproduced with both legs quoting bid=0.05/ask=0.10.
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(
                path, contract_key="AAPL|2026-09-18|Bear Call|190/195",
                strategy_name="Bear Call", opt_type=None, bid=None, ask=None,
                features_json='{"short_strike": 190.0, "long_strike": 195.0, '
                              '"short_bid": 0.05, "short_ask": 0.10, '
                              '"long_bid": 0.05, "long_ask": 0.10}')
            out = cm.mark_candidates(
                db_path=path, today="2026-08-19", cfg_path=cfg,
                fetch=lambda t, e: {(190.0, "call"): (0.05, 0.10),
                                    (195.0, "call"): (0.05, 0.10)})
            self.assertEqual(out["marked"], 1)
            with sqlite3.connect(path) as conn:
                mid, source = conn.execute(
                    "select mid, source from candidate_marks "
                    "where contract_key = ?",
                    ("AAPL|2026-09-18|Bear Call|190/195",)).fetchone()
            self.assertEqual(mid, 0.0)
            self.assertEqual(source, "live_quote_structure")

    def test_a_broken_fetch_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_candidate(path)

            def boom(t, e):
                raise RuntimeError("boom")

            out = cm.mark_candidates(db_path=path, today="2026-08-19",
                                     cfg_path=cfg, fetch=boom)
            self.assertEqual(out["marked"], 0)
            self.assertEqual(out["opened"], 1)   # the rest of the run survives


def _insert_pairs(path, n):
    """`n` candidates, each on its own (symbol, expiration) pair."""
    for i in range(n):
        _insert_candidate(path, scan_id=f"S{i}", symbol=f"T{i}",
                          contract_key=f"T{i}|2026-09-18|call|190")


class TestARateLimitedRun(unittest.TestCase):
    """2026-10-14: Yahoo rate-limited the 10:39 run, ~480 of ~800 pairs came
    back empty, and the day was stamped done — 3,700 contracts lost their mark
    for good, while later runs that day could have priced every one."""

    def test_a_run_that_lost_most_pairs_is_not_complete(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_pairs(path, 40)
            calls = []

            def flaky(t, e):
                calls.append(t)
                return {} if len(calls) > 10 else {(190.0, "call"): (9.9, 10.1)}

            out = cm.mark_candidates(db_path=path, today="2026-08-19",
                                     cfg_path=cfg, fetch=flaky)
            self.assertFalse(out["complete"])
            self.assertGreater(out["failed_pairs"], 5)

    def test_a_few_dead_pairs_still_count_as_complete(self):
        # The normal day: 1-4 pairs out of ~800 have no quotes at all.
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_pairs(path, 40)
            out = cm.mark_candidates(
                db_path=path, today="2026-08-19", cfg_path=cfg,
                fetch=lambda t, e: {} if t in ("T3", "T7") else
                {(190.0, "call"): (9.9, 10.1)})
            self.assertTrue(out["complete"])
            self.assertEqual(out["failed_pairs"], 2)
            self.assertEqual(out["marked"], 38)

    def test_a_rerun_fetches_only_what_is_still_unmarked(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_pairs(path, 10)
            good = {(190.0, "call"): (9.9, 10.1)}
            cm.mark_candidates(
                db_path=path, today="2026-08-19", cfg_path=cfg,
                fetch=lambda t, e: {} if t in ("T1", "T2") else good)
            fetched = []

            def record(t, e):
                fetched.append(t)
                return good

            out = cm.mark_candidates(db_path=path, today="2026-08-19",
                                     cfg_path=cfg, fetch=record)
            self.assertEqual(sorted(set(fetched)), ["T1", "T2"])
            self.assertEqual(out["marked"], 2)
            with sqlite3.connect(path) as conn:
                n, = conn.execute("select count(*) from candidate_marks "
                                  "where mark_date='2026-08-19'").fetchone()
            self.assertEqual(n, 10)

    def test_a_dead_feed_stops_after_a_run_of_failures(self):
        # Hammering a rate-limited feed for every remaining pair extends the
        # limit and starves the scan that runs next.
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_pairs(path, 100)
            calls = []

            def dead(t, e):
                calls.append(t)
                return {}

            out = cm.mark_candidates(db_path=path, today="2026-08-19",
                                     cfg_path=cfg, fetch=dead)
            self.assertFalse(out["complete"])
            self.assertTrue(out["aborted"])
            self.assertLess(len(set(calls)), 100)

    def test_an_incomplete_run_does_not_resolve(self):
        # `resolve` falls back to the newest mark on or before today. On a
        # partial day that is YESTERDAY's price for every unpriced contract,
        # booked as an exit a later run could have priced properly.
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            _insert_pairs(path, 40)
            cm.mark_candidates(db_path=path, today="2026-08-01", cfg_path=cfg,
                               fetch=lambda t, e: {(190.0, "call"): (9.9, 10.1)})
            up = {(190.0, "call"): (24.9, 25.1)}
            calls = []

            def flaky(t, e):
                calls.append(t)
                return up if len(calls) <= 10 else {}

            out = cm.mark_candidates(db_path=path, today="2026-08-10",
                                     cfg_path=cfg, fetch=flaky)
            self.assertFalse(out["complete"])
            self.assertEqual(out["closed"], 0)
            out = cm.mark_candidates(db_path=path, today="2026-08-10",
                                     cfg_path=cfg, fetch=lambda t, e: up)
            self.assertTrue(out["complete"])
            self.assertEqual(out["closed"], 40)


class TestAgainstRealRecordedRows(unittest.TestCase):
    """Drives the whole run over rows written by the real recorder, not by a
    test fixture that happens to look like them."""

    def test_a_full_lifecycle_from_record_to_close(self):
        import pandas as pd
        from src import candidate_record as cr
        from src import pick_ranking as pr

        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            leg = {"symbol": "AAPL", "type": "call",
                   "expiration": "2026-09-18", "strike": 190.0,
                   "bid": 9.90, "ask": 10.10, "premium": 10.0,
                   "theta": -0.05, "quality_score": 0.5,
                   "ev_per_contract": 25.0}
            with cr.scan("Discovery scan"):
                cr.record_board(
                    pr.BoardResult(
                        kept=pd.DataFrame([leg]),
                        refused=pd.DataFrame([dict(leg, strike=200.0,
                                                   bid=5.9, ask=6.1,
                                                   refused_by="negative_ev")]),
                        scanned=2),
                    board="DISCOVERY SCAN", db_path=path)

            quotes = {(190.0, "call"): (9.9, 10.1), (200.0, "call"): (5.9, 6.1)}
            day1 = cm.mark_candidates(db_path=path, today="2026-08-01",
                                      cfg_path=cfg, fetch=lambda t, e: quotes)
            self.assertEqual(day1["opened"], 2)   # kept AND refused
            self.assertEqual(day1["marked"], 2)

            up = {(190.0, "call"): (24.9, 25.1), (200.0, "call"): (14.9, 15.1)}
            day2 = cm.mark_candidates(db_path=path, today="2026-08-10",
                                      cfg_path=cfg, fetch=lambda t, e: up)
            self.assertEqual(day2["closed"], 2)

            with sqlite3.connect(path) as conn:
                rows = conn.execute(
                    "select p.exit_reason, p.pnl_pct, c.refused_by "
                    "from candidate_positions p join candidates c "
                    "  on c.scan_id=p.scan_id and c.board=p.board "
                    " and c.contract_key=p.contract_key "
                    "order by c.strike").fetchall()

            self.assertEqual([r[0] for r in rows], ["take_profit", "take_profit"])
            self.assertTrue(all(r[1] > 1.0 for r in rows))
            # The refused candidate has an outcome. That is the whole point.
            self.assertEqual(rows[1][2], "negative_ev")
            self.assertIsNotNone(rows[1][1])

    def test_a_recorded_row_carries_the_mode_the_marker_needs(self):
        # Ties sub-project 1's amendment to sub-project 2's requirement: if the
        # mode stops reaching the row, positions silently stop being created.
        import pandas as pd
        from src import candidate_record as cr
        from src import pick_ranking as pr

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            leg = {"symbol": "AAPL", "type": "call", "expiration": "2026-09-18",
                   "strike": 190.0, "bid": 9.9, "ask": 10.1, "premium": 10.0}
            with cr.scan("Discovery scan"):
                cr.record_board(pr.BoardResult(kept=pd.DataFrame([leg]),
                                               refused=pd.DataFrame(), scanned=1),
                                board="DISCOVERY SCAN", db_path=path)
            with sqlite3.connect(path) as conn:
                mode, = conn.execute("select mode from candidates").fetchone()
            self.assertEqual(mode, "Discovery scan")
            self.assertEqual(cm.family_for(mode, "call"), "long_option")


if __name__ == "__main__":
    unittest.main()
