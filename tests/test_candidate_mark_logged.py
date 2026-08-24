"""The candidate record has to know which candidates became trades.

`mark_logged`, and the `record_autolog_logged` wrapper over it, were written,
handled their own miss case, and were **called from nowhere**. So `auto_logged`
was 0 across all 43,526 rows and nothing could answer the one question the
table exists to make answerable: of the candidates the board offered, which
did the book actually take?

That gap blocks measuring the strategy allocation shipped 2026-08-24. A share
per structure cannot be checked against realised entries if realised entries
are not recorded against their candidates.

The insert-on-miss path is the reason this needed care rather than speed. It
writes a NEW candidates row when nothing matches, and `candidates` is the
table the frozen pre-registration reads. The safety is structural rather than
promised: `row_payload` does not set `gate_passed`, so an inserted row carries
NULL there, and `prereg_ranker.load_cohort` filters `gate_passed = 1`. A miss
therefore cannot reach the cohort. That property is asserted below, because if
it ever stops holding, the November test starts eating phantom rows.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import candidate_record as cr


BOARD = "autolog_structures"


def _row(symbol="NVDA", strike=140.0, expiration="2026-12-18"):
    return {"symbol": symbol, "expiration": expiration, "type": "put",
            "strike": strike, "strategy_name": "Bull Put",
            "short_put_strike": strike, "long_put_strike": strike - 5,
            "premium": 1.2, "bid": 1.15, "ask": 1.25}


class TestMarkLogged(unittest.TestCase):

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._dir.name, "candidates.db")
        self._scan = cr.scan("Credit Spreads")
        self.scan_id = self._scan.__enter__()

    def tearDown(self):
        self._scan.__exit__(None, None, None)
        self._dir.cleanup()

    def _rows(self):
        with sqlite3.connect(self.db) as conn:
            conn.row_factory = sqlite3.Row
            return [dict(r) for r in
                    conn.execute("SELECT * FROM candidates").fetchall()]

    def test_a_recorded_candidate_is_flagged_when_it_is_entered(self):
        cr.mark_ranked([_row()], board=BOARD, db_path=self.db)
        before = self._rows()
        self.assertEqual(len(before), 1)
        self.assertEqual(before[0]["auto_logged"], 0)

        cr.mark_logged(_row(), board=BOARD, entry_id=4321, db_path=self.db)
        after = self._rows()
        self.assertEqual(len(after), 1, "flagging created a duplicate row")
        self.assertEqual(after[0]["auto_logged"], 1)
        self.assertEqual(after[0]["entry_id"], 4321)

    def test_only_the_entered_candidate_is_flagged(self):
        cr.mark_ranked([_row(strike=140.0), _row(strike=145.0)],
                       board=BOARD, db_path=self.db)
        cr.mark_logged(_row(strike=145.0), board=BOARD, entry_id=7,
                       db_path=self.db)
        flags = {r["contract_key"]: r["auto_logged"] for r in self._rows()}
        self.assertEqual(sum(flags.values()), 1)
        self.assertEqual(
            [k for k, v in flags.items() if v][0],
            cr.contract_key(_row(strike=145.0)))

    def test_an_entry_with_no_recorded_candidate_is_still_kept(self):
        """Losing the taken row would be the worst gap this table can have."""
        cr.mark_logged(_row(), board=BOARD, entry_id=99, db_path=self.db)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["auto_logged"], 1)
        self.assertEqual(rows[0]["entry_id"], 99)

    def test_an_inserted_miss_can_never_reach_the_prereg_cohort(self):
        """THE guard. `candidates` backs the frozen pre-registration, and its
        cohort is `gate_passed = 1`. A row invented by the miss path must not
        satisfy that, or the November test starts counting phantoms."""
        cr.mark_logged(_row(), board=BOARD, entry_id=99, db_path=self.db)
        rows = self._rows()
        self.assertIsNone(rows[0]["gate_passed"],
                          "a miss-inserted row claims to have passed a gate "
                          "it was never shown to")

    def test_the_wrapper_flags_through_to_the_table(self):
        from src.options_screener import record_autolog_logged
        cr.mark_ranked([_row()], board=BOARD, db_path=self.db)
        record_autolog_logged(_row(), board=BOARD, entry_id=11,
                              db_path=self.db)
        self.assertEqual(self._rows()[0]["auto_logged"], 1)

    def test_flagging_twice_is_idempotent(self):
        """Re-running a scan on the same day must not double-count entries."""
        cr.mark_ranked([_row()], board=BOARD, db_path=self.db)
        cr.mark_logged(_row(), board=BOARD, entry_id=1, db_path=self.db)
        cr.mark_logged(_row(), board=BOARD, entry_id=1, db_path=self.db)
        rows = self._rows()
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["auto_logged"], 1)

    def test_it_works_on_the_pandas_row_the_call_sites_actually_pass(self):
        """The call sites sit inside `for _, row in _candidates.iterrows()`,
        so `row` is a Series, not the dict the tests above use. A key derived
        differently from one would flag nothing and silently insert instead."""
        import pandas as pd
        frame = pd.DataFrame([_row(strike=140.0), _row(strike=145.0)])
        cr.mark_ranked(frame.to_dict("records"), board=BOARD, db_path=self.db)

        for _, series in frame.iterrows():
            if float(series["strike"]) == 145.0:
                cr.mark_logged(dict(series), board=BOARD, entry_id=55,
                               db_path=self.db)

        rows = self._rows()
        self.assertEqual(len(rows), 2, "a Series produced a different key and "
                                       "inserted a phantom row")
        flagged = [r for r in rows if r["auto_logged"] == 1]
        self.assertEqual(len(flagged), 1)
        self.assertEqual(flagged[0]["contract_key"],
                         cr.contract_key(_row(strike=145.0)))
        self.assertEqual(flagged[0]["entry_id"], 55)

    def test_a_broken_call_cannot_stop_a_scan(self):
        """Failure-safe, like the rest of the recorder: a scan must not die
        because the bookkeeping did."""
        self.assertIsNone(
            cr.mark_logged(_row(), board=BOARD, entry_id=1,
                           db_path="/nonexistent/dir/candidates.db"))


class TestNeverMarkedIsNotADailyFalseAlarm(unittest.TestCase):
    """A position recorded TODAY has not missed a mark run; it is waiting for
    its first one.

    Counting those made the check go CRITICAL every single day: on 2026-08-24
    it reported "3905 OPEN POSITIONS HAVE NEVER BEEN MARKED", and every one of
    them had been entered that morning — every prior entry date showed zero.
    An alarm that is red every day carries the same information as one that is
    never red, and this repo has already paid for a health check that did not
    catch what it claimed.
    """

    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self._dir.name, "c.db")

    def tearDown(self):
        self._dir.cleanup()

    def _seed(self, entry_dates):
        """Seeds one unrelated mark too, so the separate and correct
        'NO MARKS while positions are open' alarm does not fire and this test
        isolates the branch it is actually about."""
        from src import candidate_marks as cm
        import datetime
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        with cm.connect(self.db) as conn:
            for i, d in enumerate(entry_dates):
                conn.execute(
                    "INSERT INTO candidate_positions (scan_id, board, "
                    "contract_key, family, entry_date, entry_price, status) "
                    "VALUES (?,?,?,?,?,?,?)",
                    (f"s{i}", "b", f"K{i}", "short_premium", d, 1.0, "OPEN"))
            conn.execute(
                "INSERT INTO candidate_marks (contract_key, mark_date, bid, "
                "ask, mid, source) VALUES (?,?,?,?,?,?)",
                ("UNRELATED|k", today, 1.0, 1.1, 1.05, "test"))
            conn.commit()

    def _lines(self):
        from src import candidate_marks as cm
        return "\n".join(cm.health_lines(self.db))

    def test_positions_entered_today_do_not_raise_the_alarm(self):
        import datetime
        today = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
        self._seed([today] * 5)
        out = self._lines()
        self.assertNotIn("NEVER BEEN MARKED", out)
        self.assertNotIn("CRITICAL", out)

    def test_an_older_unmarked_position_still_raises_it(self):
        """The alarm must keep working for what it was written to catch."""
        self._seed(["2026-01-05"] * 3)
        out = self._lines()
        self.assertIn("NEVER BEEN MARKED", out)
        self.assertIn("CRITICAL", out)


if __name__ == "__main__":
    unittest.main()
