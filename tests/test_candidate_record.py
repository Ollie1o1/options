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


if __name__ == "__main__":
    unittest.main()
