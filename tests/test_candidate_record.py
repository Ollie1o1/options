"""Tests for src/candidate_record.py — the pre-gate candidate dataset.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_candidate_record -v

Never names the real ledger or the real data/candidates.db: every test passes
an explicit temp path.
"""
import json
import os
import sqlite3
import tempfile
import unittest
import unittest.mock

import pandas as pd

from src import candidate_record as cr
from src import pick_ranking as pr


def _leg(**over):
    """A single-leg candidate row as a scan frame carries it."""
    row = {"symbol": "AAPL", "strategy_name": "Long Call", "type": "call",
           "expiration": "2026-09-18", "strike": 190.0,
           "bid": 9.90, "ask": 10.10, "premium": 10.0, "theta": -0.05,
           "delta": 0.55, "quality_score": 0.50, "ev_per_contract": 25.0}
    row.update(over)
    return row


def _condor(**over):
    row = {"symbol": "SPY", "strategy_name": "Iron Condor",
           "expiration": "2026-09-18",
           "short_put_strike": 540.0, "long_put_strike": 535.0,
           "short_call_strike": 580.0, "long_call_strike": 585.0,
           "premium": 2.0, "theta": -0.02, "quality_score": 0.50}
    row.update(over)
    return row


class TestSchema(unittest.TestCase):
    def test_connect_creates_both_tables(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "candidates.db")
            with cr.connect(path) as conn:
                names = {r[0] for r in conn.execute(
                    "select name from sqlite_master where type='table'")}
            self.assertIn("candidates", names)
            self.assertIn("recorder_errors", names)

    def test_connect_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "candidates.db")
            cr.connect(path).close()
            with cr.connect(path) as conn:   # must not raise on second open
                cols = {r[1] for r in conn.execute("PRAGMA table_info(candidates)")}
            self.assertIn("round_trip_pct", cols)
            # rank_by_verdict writes round-trip cost into a column named
            # `friction_pct`. Persisting that name would store a number under a
            # label describing something else.
            self.assertNotIn("friction_pct", cols)


class TestContractKey(unittest.TestCase):
    def test_single_leg_key_is_stable(self):
        self.assertEqual(cr.contract_key(_leg()), cr.contract_key(_leg()))
        self.assertEqual(cr.contract_key(_leg()), "AAPL|2026-09-18|call|190")

    def test_strike_difference_changes_the_key(self):
        self.assertNotEqual(cr.contract_key(_leg()),
                            cr.contract_key(_leg(strike=195.0)))

    def test_condors_differing_in_one_leg_differ(self):
        a = cr.contract_key(_condor())
        b = cr.contract_key(_condor(long_call_strike=590.0))
        self.assertNotEqual(a, b)

    def test_condor_key_names_the_strategy(self):
        self.assertTrue(cr.contract_key(_condor()).startswith(
            "SPY|2026-09-18|Iron Condor|"))

    def test_strategy_in_type_is_not_read_as_an_option_type(self):
        # candidate_verdict._legs_of reads `strategy_name or type`, so `type`
        # sometimes carries a STRATEGY. It must never land in the opt_type
        # COLUMN, which is what this guards — the key's discriminator slot
        # legitimately falls back to the strategy.
        row = {"symbol": "X", "expiration": "2026-09-18",
               "type": "Bull Put", "strike": 10.0}
        self.assertEqual(cr._opt_type_of(row), "")
        self.assertEqual(cr._strategy_of(row), "Bull Put")

    def test_option_type_is_read_when_type_really_is_one(self):
        self.assertEqual(cr._opt_type_of(_leg()), "call")
        self.assertEqual(cr._opt_type_of(_leg(type="P")), "put")

    def test_a_structure_without_leg_strikes_does_not_collide(self):
        # Two Bull Puts on the same symbol and expiry, no leg strike columns.
        # Keying both as "X|exp|Bull Put|/" would let the primary key
        # silently overwrite one candidate with the other.
        a = {"symbol": "X", "expiration": "2026-09-18",
             "strategy_name": "Bull Put", "strike": 10.0}
        b = dict(a, strike=12.0)
        self.assertNotEqual(cr.contract_key(a), cr.contract_key(b))

    def test_a_structure_with_leg_strikes_still_uses_them(self):
        key = cr.contract_key({"symbol": "X", "expiration": "2026-09-18",
                               "strategy_name": "Bull Put",
                               "short_strike": 100.0, "long_strike": 95.0})
        self.assertEqual(key, "X|2026-09-18|Bull Put|100/95")


class TestDbPathResolution(unittest.TestCase):
    """The path is decided at CALL time. A default argument would be bound at
    import and be invisible to anything trying to redirect it — the same shape
    as the `sort_by` default that ranked the board by quality_score for
    months."""

    def test_explicit_path_wins(self):
        self.assertEqual(cr._resolve_db_path("/tmp/x.db"), "/tmp/x.db")

    def test_env_var_redirects_when_no_explicit_path(self):
        with unittest.mock.patch.dict(
                os.environ, {cr.DB_PATH_ENV: "/tmp/from_env.db"}):
            self.assertEqual(cr._resolve_db_path(), "/tmp/from_env.db")

    def test_default_is_used_when_nothing_overrides(self):
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(cr._resolve_db_path(), cr.DEFAULT_DB_PATH)

    def test_the_test_suite_is_not_pointed_at_the_real_database(self):
        # If this fails, the suite is writing fixture tickers into the dataset
        # the ranker will be judged from.
        self.assertNotEqual(cr._resolve_db_path(), cr.DEFAULT_DB_PATH)


class TestFailureCapture(unittest.TestCase):
    """A recorder that returns cleanly and writes nothing is how four months
    of shadow-mark data went missing. Failures must be counted and persisted."""

    def setUp(self):
        cr.reset_stats()

    def test_unwritable_path_does_not_raise_and_counts(self):
        with tempfile.TemporaryDirectory() as d:
            # A directory is not a database file: sqlite cannot open it.
            rows = [{"board": "discover", "contract_key": "K", "scan_id": "S"}]
            self.assertEqual(cr.record_board_rows(rows, db_path=d), 0)
            self.assertEqual(cr.STATS["errors"], 1)

    def test_broken_table_writes_an_error_row(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "candidates.db")
            cr.connect(path).close()
            conn = sqlite3.connect(path)
            conn.execute("DROP TABLE candidates")
            conn.execute("CREATE TABLE candidates (nonsense TEXT)")
            conn.commit()
            conn.close()

            rows = [{"board": "discover", "contract_key": "K", "scan_id": "S"}]
            self.assertEqual(cr.record_board_rows(rows, db_path=path), 0)
            self.assertEqual(cr.STATS["errors"], 1)

            with sqlite3.connect(path) as conn:
                errs = conn.execute("select where_, traceback "
                                    "from recorder_errors").fetchall()
            self.assertEqual(len(errs), 1)
            self.assertIn("record_board_rows", errs[0][0])
            self.assertTrue(errs[0][1].strip())

    def test_a_clean_write_leaves_the_error_counter_alone(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "candidates.db")
            rows = [{"scan_id": "S", "ts": "2026-08-18T00:00:00+00:00",
                     "board": "discover", "contract_key": "K"}]
            self.assertEqual(cr.record_board_rows(rows, db_path=path), 1)
            self.assertEqual(cr.STATS["errors"], 0)
            self.assertEqual(cr.STATS["recorded"], 1)


class TestRecordBoard(unittest.TestCase):
    def setUp(self):
        cr.reset_stats()

    def _result(self):
        kept = pd.DataFrame([_leg()])
        refused = pd.DataFrame([_leg(strike=200.0, refused_by="negative_ev"),
                                _leg(strike=205.0, refused_by="friction")])
        return pr.BoardResult(kept=kept, refused=refused, scanned=3)

    def test_kept_and_refused_are_both_written(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cr.scan("discover"):
                n = cr.record_board(self._result(), board="discover", db_path=path)
            self.assertEqual(n, 3)
            with sqlite3.connect(path) as conn:
                rows = conn.execute(
                    "select refused_by, gate_passed from candidates "
                    "order by strike").fetchall()
            self.assertEqual(rows[0], (None, 1))
            self.assertEqual(sorted(r[0] for r in rows[1:]),
                             ["friction", "negative_ev"])
            self.assertEqual([r[1] for r in rows[1:]], [0, 0])

    def test_scan_id_is_shared_across_boards(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cr.scan("discover"):
                cr.record_board(self._result(), board="discover", db_path=path)
                cr.record_board(self._result(), board="top", db_path=path)
            with sqlite3.connect(path) as conn:
                ids = conn.execute(
                    "select count(distinct scan_id), count(distinct board) "
                    "from candidates").fetchone()
            self.assertEqual(ids, (1, 2))

    def test_absent_columns_stay_null_not_zero(self):
        # NULL means *not recorded*. A row with no EV must not read as EV 0.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            bare = pd.DataFrame([{"symbol": "AAPL", "strategy_name": "Long Call",
                                  "expiration": "2026-09-18", "strike": 190.0}])
            with cr.scan("discover"):
                cr.record_board(pr.BoardResult(kept=bare, refused=pd.DataFrame(),
                                               scanned=1),
                                board="discover", db_path=path)
            with sqlite3.connect(path) as conn:
                ev, qs = conn.execute(
                    "select ev_net, quality_score from candidates").fetchone()
            self.assertIsNone(ev)
            self.assertIsNone(qs)

    def test_features_json_carries_the_tail(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            df = pd.DataFrame([_leg(gex_score=0.42, pcr_score=0.19)])
            with cr.scan("discover"):
                cr.record_board(pr.BoardResult(kept=df, refused=pd.DataFrame(),
                                               scanned=1),
                                board="discover", db_path=path)
            with sqlite3.connect(path) as conn:
                blob, = conn.execute(
                    "select features_json from candidates").fetchone()
            tail = json.loads(blob)
            self.assertEqual(tail["gex_score"], 0.42)
            self.assertEqual(tail["pcr_score"], 0.19)
            # Fixed columns are not duplicated into the blob.
            self.assertNotIn("quality_score", tail)

    def test_round_trip_pct_is_read_from_the_misnamed_column(self):
        # rank_by_verdict writes Verdict.round_trip_pct into `friction_pct`.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            df = pd.DataFrame([_leg(friction_pct=0.31)])
            with cr.scan("discover"):
                cr.record_board(pr.BoardResult(kept=df, refused=pd.DataFrame(),
                                               scanned=1),
                                board="discover", db_path=path)
            with sqlite3.connect(path) as conn:
                rt, blob = conn.execute(
                    "select round_trip_pct, features_json from candidates").fetchone()
            self.assertAlmostEqual(rt, 0.31)
            # And it is not ALSO left in the blob under its misleading name.
            self.assertNotIn("friction_pct", json.loads(blob or "{}"))

    def test_an_empty_board_writes_nothing_and_does_not_error(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cr.scan("discover"):
                n = cr.record_board(
                    pr.BoardResult(kept=pd.DataFrame(), refused=pd.DataFrame(),
                                   scanned=0), board="discover", db_path=path)
            self.assertEqual(n, 0)
            self.assertEqual(cr.STATS["errors"], 0)


class TestGatingFailed(unittest.TestCase):
    """gate_board's failure-safe branch returns a result byte-identical to a
    board where every candidate cleared every gate. Recording that as
    'all passed' would write false data into the one table meant to settle a
    scientific question."""

    def setUp(self):
        cr.reset_stats()

    def test_default_is_false(self):
        self.assertFalse(pr.BoardResult(kept=pd.DataFrame(),
                                        refused=pd.DataFrame()).gating_failed)

    def test_gate_board_failure_sets_the_flag(self):
        with unittest.mock.patch.object(
                pr, "top_quintile_cutoff", side_effect=RuntimeError("boom")):
            result = pr.gate_board(pd.DataFrame([_leg()]))
        self.assertTrue(result.gating_failed)
        self.assertEqual(len(result.kept), 1)      # still failure-safe

    def test_a_healthy_gating_leaves_the_flag_false(self):
        result = pr.gate_board(pd.DataFrame([_leg()]))
        self.assertFalse(result.gating_failed)

    def test_failed_gating_is_recorded_as_failed_not_as_all_passed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            result = pr.BoardResult(kept=pd.DataFrame([_leg()]),
                                    refused=pd.DataFrame(), scanned=1,
                                    gating_failed=True)
            with cr.scan("discover"):
                cr.record_board(result, board="discover", db_path=path)
            with sqlite3.connect(path) as conn:
                flag, passed = conn.execute(
                    "select gating_failed, gate_passed from candidates").fetchone()
            self.assertEqual(flag, 1)
            self.assertEqual(passed, 1)   # kept, but the keeping is not evidence


class TestMarkRanked(unittest.TestCase):
    def setUp(self):
        cr.reset_stats()

    def test_rank_updates_an_existing_gate_row_without_duplicating(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            row = _leg()
            with cr.scan("discover"):
                cr.record_board(pr.BoardResult(kept=pd.DataFrame([row]),
                                               refused=pd.DataFrame(), scanned=1),
                                board="discover", db_path=path)
                cr.mark_ranked([row], board="discover", db_path=path)
            with sqlite3.connect(path) as conn:
                rows = conn.execute(
                    "select rank_pos, gate_passed from candidates").fetchall()
            self.assertEqual(len(rows), 1)          # upsert, not insert
            self.assertEqual(rows[0], (1, 1))       # rank set, gate kept

    def test_rank_is_one_based_and_in_frame_order(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            rows = [_leg(strike=190.0), _leg(strike=195.0), _leg(strike=200.0)]
            with cr.scan("discover"):
                cr.mark_ranked(rows, board="discover", db_path=path)
            with sqlite3.connect(path) as conn:
                got = conn.execute("select strike, rank_pos from candidates "
                                   "order by rank_pos").fetchall()
            self.assertEqual(got, [(190.0, 1), (195.0, 2), (200.0, 3)])

    def test_autolog_only_rows_are_inserted_and_counted(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cr.scan("discover"):
                cr.record_board(pr.BoardResult(kept=pd.DataFrame([_leg()]),
                                               refused=pd.DataFrame(), scanned=1),
                                board="discover", db_path=path)
                # A row the board never saw — the divergence this counts.
                cr.mark_ranked([_leg(), _leg(strike=250.0)],
                               board="discover", db_path=path)
            self.assertEqual(cr.STATS["autolog_only"], 1)
            with sqlite3.connect(path) as conn:
                n, = conn.execute("select count(*) from candidates").fetchone()
            self.assertEqual(n, 2)

    def test_mark_logged_sets_the_flag_and_entry_id(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            row = _leg()
            with cr.scan("discover"):
                cr.mark_ranked([row], board="discover", db_path=path)
                cr.mark_logged(row, board="discover", entry_id=4242, db_path=path)
            with sqlite3.connect(path) as conn:
                got = conn.execute("select auto_logged, entry_id, rank_pos "
                                   "from candidates").fetchone()
            self.assertEqual(got, (1, 4242, 1))   # rank survives the update

    def test_mark_logged_inserts_a_row_it_has_never_seen(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cr.scan("discover"):
                cr.mark_logged(_leg(), board="discover", entry_id=7,
                               db_path=path)
            with sqlite3.connect(path) as conn:
                got = conn.execute(
                    "select auto_logged, entry_id from candidates").fetchone()
            self.assertEqual(got, (1, 7))

    def test_refusal_reason_survives_a_rank_update(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            row = _leg()
            with cr.scan("discover"):
                cr.record_board(
                    pr.BoardResult(kept=pd.DataFrame(),
                                   refused=pd.DataFrame([dict(row,
                                       refused_by="negative_ev")]), scanned=1),
                    board="discover", db_path=path)
                cr.mark_ranked([row], board="discover", db_path=path)
            with sqlite3.connect(path) as conn:
                got = conn.execute("select refused_by, gate_passed, rank_pos "
                                   "from candidates").fetchone()
            self.assertEqual(got, ("negative_ev", 0, 1))

    def test_mark_refused_records_the_pre_cut_reason(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            rows = [_leg(strike=190.0), _leg(strike=195.0)]
            with cr.scan("discover"):
                cr.mark_ranked(rows, board="autolog", db_path=path)
                cr.mark_refused([rows[1]], "budget_displaced",
                                board="autolog", db_path=path)
            with sqlite3.connect(path) as conn:
                got = dict(conn.execute(
                    "select strike, refused_by from candidates").fetchall())
                rank = conn.execute("select rank_pos from candidates "
                                    "where strike=195.0").fetchone()[0]
            self.assertIsNone(got[190.0])
            self.assertEqual(got[195.0], "budget_displaced")
            self.assertEqual(rank, 2)     # rank survives the refusal mark


if __name__ == "__main__":
    unittest.main()
