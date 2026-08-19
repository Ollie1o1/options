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


class TestOpenPositions(unittest.TestCase):
    def test_a_candidate_becomes_exactly_one_position(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            self.assertEqual(cm.open_positions(db_path=path, today="2026-08-19"), 1)
            with sqlite3.connect(path) as conn:
                rows = conn.execute(
                    "select family, status, entry_date from candidate_positions"
                ).fetchall()
            self.assertEqual(rows, [("long_option", "OPEN", "2026-08-19")])

    def test_running_twice_does_not_duplicate(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            cm.open_positions(db_path=path, today="2026-08-19")
            self.assertEqual(cm.open_positions(db_path=path, today="2026-08-20"), 0)
            with sqlite3.connect(path) as conn:
                n, = conn.execute(
                    "select count(*) from candidate_positions").fetchone()
            self.assertEqual(n, 1)

    def test_refused_candidates_get_positions_too(self):
        # The refused population is the entire point of the sub-project.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path, gate_passed=0, refused_by="negative_ev")
            cm.open_positions(db_path=path, today="2026-08-19")
            with sqlite3.connect(path) as conn:
                n, = conn.execute(
                    "select count(*) from candidate_positions").fetchone()
            self.assertEqual(n, 1)

    def test_the_same_contract_on_three_scans_makes_three_positions(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            for i, px in enumerate([9.9, 10.9, 11.9]):
                _insert_candidate(path, scan_id=f"S{i}", bid=px, ask=px + 0.2)
            cm.open_positions(db_path=path, today="2026-08-19")
            with sqlite3.connect(path) as conn:
                prices = [r[0] for r in conn.execute(
                    "select entry_price from candidate_positions order by scan_id")]
                keys = {r[0] for r in conn.execute(
                    "select contract_key from candidate_positions")}
            self.assertEqual(len(prices), 3)
            self.assertEqual(len(set(prices)), 3)   # three different entries
            self.assertEqual(len(keys), 1)          # one contract, one mark stream

    def test_a_null_mode_produces_no_position_at_all(self):
        # Not an UNMARKABLE row: a candidate with no derivable family is not a
        # decision this can simulate, and thousands of inert placeholders would
        # bury the rows that mean something.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path, mode=None)
            self.assertEqual(cm.open_positions(db_path=path, today="2026-08-19"), 0)
            with sqlite3.connect(path) as conn:
                n, = conn.execute(
                    "select count(*) from candidate_positions").fetchone()
            self.assertEqual(n, 0)

    def test_an_unquotable_candidate_opens_unmarkable(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path, bid=None, ask=None)
            cm.open_positions(db_path=path, today="2026-08-19")
            with sqlite3.connect(path) as conn:
                status, price = conn.execute(
                    "select status, entry_price from candidate_positions").fetchone()
            self.assertEqual(status, "UNMARKABLE")
            self.assertIsNone(price)

    def test_short_premium_opens_unsupported(self):
        # Its stops need spot and delta, which a bid/ask mark does not carry.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path, mode="Premium Selling", opt_type="put")
            cm.open_positions(db_path=path, today="2026-08-19")
            with sqlite3.connect(path) as conn:
                status, reason = conn.execute(
                    "select status, exit_reason from candidate_positions").fetchone()
            self.assertEqual(status, "UNSUPPORTED")
            self.assertEqual(reason, "needs_spot_and_delta")

    def test_the_paper_ledger_is_never_opened(self):
        # This module must not be able to touch the book.
        import builtins
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            real_connect = sqlite3.connect
            seen = []

            def spy(target, *a, **kw):
                seen.append(str(target))
                return real_connect(target, *a, **kw)

            sqlite3.connect = spy
            try:
                cm.open_positions(db_path=path, today="2026-08-19")
            finally:
                sqlite3.connect = real_connect
            self.assertTrue(seen)
            self.assertFalse([s for s in seen if "paper_trades" in s])


if __name__ == "__main__":
    unittest.main()
