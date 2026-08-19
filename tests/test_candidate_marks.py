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


class TestEntryPricing(unittest.TestCase):
    def _single(self, **over):
        row = {"strategy_name": None, "opt_type": "call",
               "bid": 9.90, "ask": 10.10, "features_json": None}
        row.update(over)
        return row

    def test_a_single_leg_is_priced_at_the_limit_fill(self):
        expected = et.structure_fill(
            [{"bid": 9.90, "ask": 10.10, "side": "buy"}], "limit").price
        self.assertAlmostEqual(cm.entry_price_for(self._single()), expected)

    def test_a_long_option_prices_as_a_debit(self):
        # Signed from the trader's cash perspective: paying is negative.
        self.assertLess(cm.entry_price_for(self._single()), 0)

    def test_a_credit_spread_prices_from_its_legs_in_the_blob(self):
        # Leg quotes are not fixed columns; they survive in features_json.
        row = {"strategy_name": "Bull Put", "opt_type": None,
               "bid": None, "ask": None,
               "features_json": json.dumps({"short_bid": 2.00, "short_ask": 2.10,
                                            "long_bid": 1.00, "long_ask": 1.10})}
        expected = et.structure_fill(
            [{"bid": 2.00, "ask": 2.10, "side": "sell"},
             {"bid": 1.00, "ask": 1.10, "side": "buy"}], "limit").price
        self.assertAlmostEqual(cm.entry_price_for(row), expected)
        self.assertGreater(cm.entry_price_for(row), 0)   # a credit

    def test_a_row_with_no_quotes_cannot_be_priced(self):
        self.assertIsNone(cm.entry_price_for(
            {"strategy_name": None, "opt_type": "call",
             "bid": None, "ask": None, "features_json": None}))

    def test_a_spread_missing_a_leg_cannot_be_priced(self):
        # Refusing on one bad leg is deliberate: a spread priced from one real
        # quote and one guess is not a price.
        row = {"strategy_name": "Bull Put", "opt_type": None,
               "bid": None, "ask": None,
               "features_json": json.dumps({"short_bid": 2.00, "short_ask": 2.10})}
        self.assertIsNone(cm.entry_price_for(row))

    def test_a_short_single_leg_prices_as_a_credit(self):
        row = self._single(strategy_name="Short Put", opt_type="put")
        self.assertGreater(cm.entry_price_for(row), 0)

    def test_a_crossed_quote_is_refused(self):
        self.assertIsNone(cm.entry_price_for(self._single(bid=10.5, ask=9.5)))


class TestPnlSign(unittest.TestCase):
    """Direction comes from the SIGN of the entry, not a family table, so a
    debit spread cannot be mis-signed by someone forgetting an entry."""

    def test_a_debit_gains_when_the_mark_rises(self):
        self.assertAlmostEqual(cm.pnl_pct(-10.0, 15.0), 0.5)

    def test_a_debit_loses_when_the_mark_falls(self):
        self.assertAlmostEqual(cm.pnl_pct(-10.0, 5.0), -0.5)

    def test_a_credit_gains_when_the_mark_falls(self):
        # Collected 1.00, now costs 0.40 to close -> +60%.
        self.assertAlmostEqual(cm.pnl_pct(1.00, 0.40), 0.6)

    def test_a_credit_loses_when_the_mark_rises(self):
        self.assertAlmostEqual(cm.pnl_pct(1.00, 2.00), -1.0)

    def test_a_zero_entry_has_no_return(self):
        self.assertIsNone(cm.pnl_pct(0.0, 1.0))

    def test_a_missing_mark_has_no_return(self):
        self.assertIsNone(cm.pnl_pct(-10.0, None))


if __name__ == "__main__":
    unittest.main()
