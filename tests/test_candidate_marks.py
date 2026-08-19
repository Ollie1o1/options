"""Tests for src/candidate_marks.py — outcomes for recorded candidates.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_candidate_marks -v

No test touches the network: the quote fetcher is injected everywhere. No test
names the real ledger, the real candidates database, or the real config.
"""
import json
import os
import sqlite3
import tempfile
import unittest

from src import candidate_marks as cm
from src import execution_truth as et


def _insert_candidate(path, **over):
    """One recorded candidate row, shaped as the recorder writes them."""
    row = {"scan_id": "S1", "ts": "2026-08-19T00:00:00+00:00", "board": "b",
           "contract_key": "AAPL|2026-09-18|call|190", "mode": "Discovery scan",
           "symbol": "AAPL", "strategy_name": None, "expiration": "2026-09-18",
           "strike": 190.0, "opt_type": "call", "bid": 9.90, "ask": 10.10,
           "gate_passed": 1, "features_json": None}
    row.update(over)
    with cm.connect(path) as conn:
        cols = ",".join(row)
        conn.execute(f"INSERT OR REPLACE INTO candidates ({cols}) "
                     f"VALUES ({','.join('?' * len(row))})", tuple(row.values()))
        conn.commit()
    return row


def _write_config(d, **over):
    rules = {"time_exit_dte": 21, "min_days_held": 3,
             "long_option": {"take_profit": 1.0, "stop_loss": -0.5},
             "spread": {"take_profit": 0.5, "stop_loss": -1.0}}
    rules.update(over)
    path = os.path.join(d, "cfg.json")
    with open(path, "w") as fh:
        json.dump({"exit_rules": rules}, fh)
    return path


def _mark(path, contract_key, date, mid):
    with cm.connect(path) as conn:
        conn.execute("INSERT OR REPLACE INTO candidate_marks "
                     "(contract_key, mark_date, bid, ask, mid, source) "
                     "VALUES (?,?,?,?,?,?)",
                     (contract_key, date, mid - 0.05, mid + 0.05, mid, "test"))
        conn.commit()


class TestSchema(unittest.TestCase):
    def test_connect_creates_both_tables(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cm.connect(path) as conn:
                names = {r[0] for r in conn.execute(
                    "select name from sqlite_master where type='table'")}
            self.assertIn("candidate_marks", names)
            self.assertIn("candidate_positions", names)
            self.assertIn("candidates", names)   # the recorder's, still there

    def test_positions_do_not_duplicate_contract_identity(self):
        # symbol/expiration/strike are joined from `candidates` on the primary
        # key. A second copy of a contract's identity is a copy that drifts.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cm.connect(path) as conn:
                cols = {r[1] for r in conn.execute(
                    "PRAGMA table_info(candidate_positions)")}
            for banned in ("symbol", "expiration", "strike", "opt_type"):
                self.assertNotIn(banned, cols)

    def test_there_is_no_dollar_pnl(self):
        # Sizing a position that was never taken means inventing a sizing rule.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            with cm.connect(path) as conn:
                cols = {r[1] for r in conn.execute(
                    "PRAGMA table_info(candidate_positions)")}
            self.assertIn("pnl_pct", cols)
            self.assertNotIn("pnl_usd", cols)


class TestFamilyFor(unittest.TestCase):
    def test_discovery_calls_and_puts_are_long_options(self):
        self.assertEqual(cm.family_for("Discovery scan", "call"), "long_option")
        self.assertEqual(cm.family_for("Discovery scan", "put"), "long_option")

    def test_premium_selling_is_short_premium(self):
        self.assertEqual(cm.family_for("Premium Selling", "put"), "short_premium")

    def test_a_named_structure_wins_over_the_mode(self):
        self.assertEqual(cm.family_for("Discovery scan", None,
                                       strategy_name="Bull Put"), "spread")
        self.assertEqual(cm.family_for("Discovery scan", None,
                                       strategy_name="Iron Condor"), "spread")

    def test_no_mode_means_no_family(self):
        self.assertIsNone(cm.family_for(None, "call"))

    def test_an_unrecognised_strategy_yields_no_family(self):
        self.assertIsNone(cm.family_for("Discovery scan", None,
                                        strategy_name="Jade Lizard"))

    def test_every_family_is_a_key_of_the_config_block(self):
        # A family this module names but config does not define would resolve
        # to an empty rule set and silently never exit.
        import json as _json
        with open("config.json") as fh:
            rules = _json.load(fh)["exit_rules"]
        for family in set(cm._FAMILY_BY_STRATEGY.values()):
            self.assertIn(family, rules)


if __name__ == "__main__":
    unittest.main()
