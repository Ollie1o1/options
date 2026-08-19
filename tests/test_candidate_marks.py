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


class TestLegSpec(unittest.TestCase):
    """The leg spec is the single description of a structure's legs."""

    def test_it_describes_the_same_legs_as_the_recorder_in_the_same_order(self):
        # Two copies of a contract's identity drifting apart is the defect shape
        # this project keeps finding. _LEG_STRIKES is load-bearing for
        # contract_key and cannot change, so this pins the pair instead.
        from src import candidate_record as cr
        self.assertEqual(set(cm._LEG_SPEC), set(cr._LEG_STRIKES))
        for strategy, spec in cm._LEG_SPEC.items():
            derived = tuple(f"{prefix}_strike" for prefix, _t, _s in spec)
            self.assertEqual(derived, cr._LEG_STRIKES[strategy], strategy)

    def test_every_leg_names_a_real_option_type_and_side(self):
        for strategy, spec in cm._LEG_SPEC.items():
            for prefix, opt_type, side in spec:
                self.assertIn(opt_type, ("put", "call"), f"{strategy}/{prefix}")
                self.assertIn(side, ("buy", "sell"), f"{strategy}/{prefix}")

    def test_a_condor_is_two_puts_and_two_calls(self):
        types = [t for _p, t, _s in cm._LEG_SPEC["Iron Condor"]]
        self.assertEqual(sorted(types), ["call", "call", "put", "put"])

    def test_a_bull_put_is_two_puts_and_a_bear_call_is_two_calls(self):
        self.assertEqual([t for _p, t, _s in cm._LEG_SPEC["Bull Put"]],
                         ["put", "put"])
        self.assertEqual([t for _p, t, _s in cm._LEG_SPEC["Bear Call"]],
                         ["call", "call"])


class TestEntryPricingSurvivesTheRefactor(unittest.TestCase):
    """Entry prices are what every open position was booked at. If the leg-spec
    refactor moves one of them, every position already open is corrupted."""

    def test_a_bull_put_prices_exactly_as_before(self):
        row = {"strategy_name": "Bull Put", "opt_type": None,
               "bid": None, "ask": None,
               "features_json": json.dumps({"short_bid": 2.00, "short_ask": 2.10,
                                            "long_bid": 1.00, "long_ask": 1.10})}
        expected = et.structure_fill(
            [{"bid": 2.00, "ask": 2.10, "side": "sell"},
             {"bid": 1.00, "ask": 1.10, "side": "buy"}], "limit").price
        self.assertAlmostEqual(cm.entry_price_for(row), expected)

    def test_a_bear_call_prices_exactly_as_before(self):
        row = {"strategy_name": "Bear Call", "opt_type": None,
               "bid": None, "ask": None,
               "features_json": json.dumps({"short_bid": 3.00, "short_ask": 3.20,
                                            "long_bid": 1.40, "long_ask": 1.60})}
        expected = et.structure_fill(
            [{"bid": 3.00, "ask": 3.20, "side": "sell"},
             {"bid": 1.40, "ask": 1.60, "side": "buy"}], "limit").price
        self.assertAlmostEqual(cm.entry_price_for(row), expected)

    def test_an_iron_condor_prices_from_all_four_legs_in_order(self):
        blob = {"short_put_bid": 2.00, "short_put_ask": 2.10,
                "long_put_bid": 1.00, "long_put_ask": 1.10,
                "short_call_bid": 2.40, "short_call_ask": 2.50,
                "long_call_bid": 1.20, "long_call_ask": 1.30}
        row = {"strategy_name": "Iron Condor", "opt_type": None,
               "bid": None, "ask": None, "features_json": json.dumps(blob)}
        expected = et.structure_fill(
            [{"bid": 2.00, "ask": 2.10, "side": "sell"},
             {"bid": 1.00, "ask": 1.10, "side": "buy"},
             {"bid": 2.40, "ask": 2.50, "side": "sell"},
             {"bid": 1.20, "ask": 1.30, "side": "buy"}], "limit").price
        self.assertAlmostEqual(cm.entry_price_for(row), expected)
        self.assertGreater(cm.entry_price_for(row), 0)   # a condor is a credit

    def test_legs_for_still_refuses_a_structure_missing_one_leg(self):
        row = {"strategy_name": "Iron Condor", "opt_type": None,
               "bid": None, "ask": None,
               "features_json": json.dumps({"short_put_bid": 2.00,
                                            "short_put_ask": 2.10})}
        self.assertIsNone(cm.legs_for(row))
        self.assertIsNone(cm.entry_price_for(row))

    def test_legs_for_carries_the_side_of_each_leg(self):
        row = {"strategy_name": "Bull Put", "opt_type": None,
               "bid": None, "ask": None,
               "features_json": json.dumps({"short_bid": 2.00, "short_ask": 2.10,
                                            "long_bid": 1.00, "long_ask": 1.10})}
        self.assertEqual([leg["side"] for leg in cm.legs_for(row)],
                         ["sell", "buy"])


class TestMarkingLegs(unittest.TestCase):
    """What must be looked up in the chain to price a position today."""

    def test_a_bull_put_yields_two_puts_from_the_blob(self):
        row = {"strategy_name": "Bull Put", "strike": None, "opt_type": None,
               "features_json": json.dumps({"short_strike": 185.0,
                                            "long_strike": 180.0})}
        self.assertEqual(cm.marking_legs(row), [
            {"strike": 185.0, "opt_type": "put", "side": "sell"},
            {"strike": 180.0, "opt_type": "put", "side": "buy"}])

    def test_a_bear_call_yields_two_calls(self):
        row = {"strategy_name": "Bear Call", "strike": None, "opt_type": None,
               "features_json": json.dumps({"short_strike": 200.0,
                                            "long_strike": 205.0})}
        self.assertEqual(cm.marking_legs(row), [
            {"strike": 200.0, "opt_type": "call", "side": "sell"},
            {"strike": 205.0, "opt_type": "call", "side": "buy"}])

    def test_an_iron_condor_yields_four_legs_in_spec_order(self):
        row = {"strategy_name": "Iron Condor", "strike": None, "opt_type": None,
               "features_json": json.dumps({"short_put_strike": 180.0,
                                            "long_put_strike": 175.0,
                                            "short_call_strike": 210.0,
                                            "long_call_strike": 215.0})}
        self.assertEqual(cm.marking_legs(row), [
            {"strike": 180.0, "opt_type": "put", "side": "sell"},
            {"strike": 175.0, "opt_type": "put", "side": "buy"},
            {"strike": 210.0, "opt_type": "call", "side": "sell"},
            {"strike": 215.0, "opt_type": "call", "side": "buy"}])

    def test_a_structure_missing_one_strike_is_unmarkable(self):
        # Not three legs and a guess. The same refusal legs_for applies at entry.
        row = {"strategy_name": "Iron Condor", "strike": None, "opt_type": None,
               "features_json": json.dumps({"short_put_strike": 180.0,
                                            "long_put_strike": 175.0,
                                            "short_call_strike": 210.0})}
        self.assertIsNone(cm.marking_legs(row))

    def test_an_unparseable_strike_is_unmarkable(self):
        row = {"strategy_name": "Bull Put", "strike": None, "opt_type": None,
               "features_json": json.dumps({"short_strike": "n/a",
                                            "long_strike": 180.0})}
        self.assertIsNone(cm.marking_legs(row))

    def test_a_structure_with_no_blob_at_all_is_unmarkable(self):
        row = {"strategy_name": "Bull Put", "strike": None, "opt_type": None,
               "features_json": None}
        self.assertIsNone(cm.marking_legs(row))

    def test_a_single_leg_uses_the_fixed_columns(self):
        row = {"strategy_name": None, "strike": 190.0, "opt_type": "call",
               "features_json": None}
        self.assertEqual(cm.marking_legs(row), [
            {"strike": 190.0, "opt_type": "call", "side": "buy"}])

    def test_a_short_single_leg_is_sold(self):
        # Matches legs_for: a strategy named Short* is a leg the trader sold.
        row = {"strategy_name": "Short Put", "strike": 180.0, "opt_type": "put",
               "features_json": None}
        self.assertEqual(cm.marking_legs(row), [
            {"strike": 180.0, "opt_type": "put", "side": "sell"}])

    def test_an_unknown_strategy_with_null_fixed_columns_is_unmarkable(self):
        # Degrades to unmarkable, never to a half-priced guess. Today's
        # behaviour for such a row, preserved deliberately.
        row = {"strategy_name": "Butterfly", "strike": None, "opt_type": None,
               "features_json": json.dumps({"short_strike": 185.0})}
        self.assertIsNone(cm.marking_legs(row))

    def test_a_missing_option_type_on_a_single_leg_is_unmarkable(self):
        row = {"strategy_name": None, "strike": 190.0, "opt_type": None,
               "features_json": None}
        self.assertIsNone(cm.marking_legs(row))

    def test_the_option_type_is_lowercased(self):
        row = {"strategy_name": None, "strike": 190.0, "opt_type": "CALL",
               "features_json": None}
        self.assertEqual(cm.marking_legs(row)[0]["opt_type"], "call")


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


class TestMarkOpen(unittest.TestCase):
    def _stub(self, quotes, calls=None):
        def fetch(ticker, expiration):
            if calls is not None:
                calls.append((ticker, expiration))
            return quotes
        return fetch

    def test_an_open_position_gets_a_mark_at_mid(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            cm.open_positions(db_path=path, today="2026-08-19")
            n = cm.mark_open(db_path=path, today="2026-08-20",
                             fetch=self._stub({(190.0, "call"): (11.0, 11.4)}))
            self.assertEqual(n, 1)
            with sqlite3.connect(path) as conn:
                bid, ask, mid, src = conn.execute(
                    "select bid, ask, mid, source from candidate_marks").fetchone()
            self.assertAlmostEqual(bid, 11.0)
            self.assertAlmostEqual(ask, 11.4)
            self.assertAlmostEqual(mid, 11.2)
            self.assertEqual(src, "live_quote")

    def test_one_chain_call_serves_every_position_on_a_pair(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            for i in range(3):
                _insert_candidate(path, scan_id=f"S{i}")
            cm.open_positions(db_path=path, today="2026-08-19")
            calls = []
            cm.mark_open(db_path=path, today="2026-08-20",
                         fetch=self._stub({(190.0, "call"): (11.0, 11.4)}, calls))
            self.assertEqual(calls, [("AAPL", "2026-09-18")])

    def test_unmarkable_and_unsupported_are_never_marked(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path, scan_id="A", bid=None, ask=None)
            _insert_candidate(path, scan_id="B", mode="Premium Selling",
                              opt_type="put")
            cm.open_positions(db_path=path, today="2026-08-19")
            calls = []
            n = cm.mark_open(db_path=path, today="2026-08-20",
                             fetch=self._stub({}, calls))
            self.assertEqual(n, 0)
            self.assertEqual(calls, [])

    def test_a_failing_fetch_does_not_stop_the_others(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path, scan_id="A", symbol="AAPL",
                              contract_key="AAPL|2026-09-18|call|190")
            _insert_candidate(path, scan_id="B", symbol="MSFT",
                              contract_key="MSFT|2026-09-18|call|500",
                              strike=500.0)
            cm.open_positions(db_path=path, today="2026-08-19")

            def fetch(ticker, expiration):
                if ticker == "AAPL":
                    raise RuntimeError("boom")
                return {(500.0, "call"): (4.0, 4.2)}

            n = cm.mark_open(db_path=path, today="2026-08-20", fetch=fetch)
            self.assertEqual(n, 1)
            with sqlite3.connect(path) as conn:
                keys = [r[0] for r in conn.execute(
                    "select contract_key from candidate_marks")]
            self.assertEqual(keys, ["MSFT|2026-09-18|call|500"])

    def test_a_missing_quote_writes_no_mark(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            cm.open_positions(db_path=path, today="2026-08-19")
            n = cm.mark_open(db_path=path, today="2026-08-20",
                             fetch=self._stub({}))
            self.assertEqual(n, 0)

    def test_marking_twice_in_a_day_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            cm.open_positions(db_path=path, today="2026-08-19")
            stub = self._stub({(190.0, "call"): (11.0, 11.4)})
            cm.mark_open(db_path=path, today="2026-08-20", fetch=stub)
            cm.mark_open(db_path=path, today="2026-08-20", fetch=stub)
            with sqlite3.connect(path) as conn:
                n, = conn.execute(
                    "select count(*) from candidate_marks").fetchone()
            self.assertEqual(n, 1)

    def test_a_closed_position_is_no_longer_marked(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            cm.open_positions(db_path=path, today="2026-08-19")
            with cm.connect(path) as conn:
                conn.execute("update candidate_positions set status='CLOSED'")
                conn.commit()
            calls = []
            cm.mark_open(db_path=path, today="2026-08-20",
                         fetch=self._stub({}, calls))
            self.assertEqual(calls, [])


class TestResolve(unittest.TestCase):
    KEY = "AAPL|2026-09-18|call|190"

    def _open(self, path, today="2026-08-01"):
        _insert_candidate(path)
        cm.open_positions(db_path=path, today=today)

    def test_take_profit_fires_and_records_the_reason(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            self._open(path)
            # Entry is a ~10.00 debit; +100% means a mark of ~20.
            _mark(path, self.KEY, "2026-08-10", 25.0)
            self.assertEqual(cm.resolve(db_path=path, today="2026-08-10",
                                        cfg_path=cfg), 1)
            with sqlite3.connect(path) as conn:
                status, reason, pnl = conn.execute(
                    "select status, exit_reason, pnl_pct "
                    "from candidate_positions").fetchone()
            self.assertEqual(status, "CLOSED")
            self.assertEqual(reason, "take_profit")
            self.assertGreater(pnl, 1.0)

    def test_stop_loss_fires(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            self._open(path)
            _mark(path, self.KEY, "2026-08-10", 2.0)     # about -80%
            cm.resolve(db_path=path, today="2026-08-10", cfg_path=cfg)
            with sqlite3.connect(path) as conn:
                reason, = conn.execute(
                    "select exit_reason from candidate_positions").fetchone()
            self.assertEqual(reason, "stop_loss")

    def test_min_days_held_suppresses_a_same_day_exit(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            self._open(path, today="2026-08-01")
            _mark(path, self.KEY, "2026-08-02", 25.0)    # would take profit
            self.assertEqual(cm.resolve(db_path=path, today="2026-08-02",
                                        cfg_path=cfg), 0)
            with sqlite3.connect(path) as conn:
                status, = conn.execute(
                    "select status from candidate_positions").fetchone()
            self.assertEqual(status, "OPEN")

    def test_time_exit_fires_at_the_dte_floor(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            self._open(path)
            # Expiration 2026-09-18; 21 DTE lands on 2026-08-28.
            _mark(path, self.KEY, "2026-08-28", 10.5)
            cm.resolve(db_path=path, today="2026-08-28", cfg_path=cfg)
            with sqlite3.connect(path) as conn:
                reason, = conn.execute(
                    "select exit_reason from candidate_positions").fetchone()
            self.assertEqual(reason, "time_exit")

    def test_expiry_closes_at_the_final_mark(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            cfg = _write_config(d, time_exit_dte=0)
            self._open(path)
            _mark(path, self.KEY, "2026-09-18", 3.0)
            cm.resolve(db_path=path, today="2026-09-19", cfg_path=cfg)
            with sqlite3.connect(path) as conn:
                reason, price = conn.execute(
                    "select exit_reason, exit_price from candidate_positions"
                ).fetchone()
            self.assertEqual(reason, "expired")
            self.assertAlmostEqual(price, 3.0)

    def test_the_thresholds_come_from_config_not_from_source(self):
        # An allowlist entry is a claim about behaviour; test it by running.
        # With take_profit raised to 5.0, a +150% mark must NOT close.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            cfg = _write_config(d, long_option={"take_profit": 5.0,
                                                "stop_loss": -0.9})
            self._open(path)
            _mark(path, self.KEY, "2026-08-10", 25.0)
            self.assertEqual(cm.resolve(db_path=path, today="2026-08-10",
                                        cfg_path=cfg), 0)

    def test_a_position_with_no_mark_stays_open(self):
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            self._open(path)
            self.assertEqual(cm.resolve(db_path=path, today="2026-08-10",
                                        cfg_path=cfg), 0)

    def test_a_future_mark_is_not_used(self):
        # Resolving on day N must not see a mark from day N+1.
        with tempfile.TemporaryDirectory() as d:
            path, cfg = os.path.join(d, "c.db"), _write_config(d)
            self._open(path)
            _mark(path, self.KEY, "2026-08-20", 25.0)
            self.assertEqual(cm.resolve(db_path=path, today="2026-08-10",
                                        cfg_path=cfg), 0)

    def test_the_real_config_is_readable_and_supplies_every_family(self):
        rules = cm.exit_rules("config.json")
        self.assertIn("time_exit_dte", rules)
        for family in ("long_option", "spread"):
            self.assertIsNotNone(rules.get(family, {}).get("take_profit"))


class TestHealthLines(unittest.TestCase):
    def test_zero_marks_with_open_positions_is_loud(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            cm.open_positions(db_path=path, today="2026-08-19")
            text = " ".join(cm.health_lines(db_path=path)).upper()
            self.assertIn("NO MARKS", text)

    def test_marks_present_is_not_shouted_about(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            _insert_candidate(path)
            cm.open_positions(db_path=path, today="2026-08-19")
            cm.mark_open(db_path=path, today="2026-08-19",
                         fetch=lambda t, e: {(190.0, "call"): (11.0, 11.4)})
            text = " ".join(cm.health_lines(db_path=path)).upper()
            self.assertNotIn("NO MARKS", text)

    def test_no_open_positions_is_not_an_alarm(self):
        # Nothing to mark is not the same as failing to mark.
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "c.db")
            cm.connect(path).close()
            text = " ".join(cm.health_lines(db_path=path)).upper()
            self.assertNotIn("NO MARKS", text)

    def test_a_missing_database_does_not_raise(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertTrue(cm.health_lines(db_path=os.path.join(d, "absent.db")))


if __name__ == "__main__":
    unittest.main()
