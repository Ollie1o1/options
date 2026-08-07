"""What a setup costs to trade, measured from the ledger's own quotes.

The friction column exists because the cost wall, not the signal, is the binding
constraint on every short-premium setup in the library. A number that cannot
change when reality does is a defect here, so the table is measured first and
falls back to the recorded 2026-08-06 derivation only when the ledger is absent.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src.strategies import friction as fr
from src.strategies.seed import LIBRARY


def _ledger(rows, path):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (strategy_name TEXT, entry_price_mid REAL, "
                 "entry_price_cross REAL)")
    conn.executemany("INSERT INTO trades VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()


def _rec(setup_id):
    return [r for r in LIBRARY if r.spec.id == setup_id][0]


def _profile(setup_id, table=None):
    """Pin the table: the live ledger is not present on every machine, and a
    display test that reads it would assert on whatever the operator traded."""
    return fr.profile_for(_rec(setup_id), table=table or fr.RECORDED)


class RecordedTableTest(unittest.TestCase):
    """The fallback must reproduce the published figures, or it is folklore."""

    def test_bull_put_costs_two_thirds_of_its_credit(self):
        """Unfiltered, as measured: 68%. The 53% once published dropped the
        four fills whose crossed price was not a tradeable credit."""
        p = _profile("put_spread_ivr50")
        self.assertAlmostEqual(p.pct_of_credit, 0.68, delta=0.03)

    def test_bear_call_is_the_cheapest_credit_structure(self):
        bull = _profile("put_spread_ivr50")
        bear = _profile("call_spread_extended")
        self.assertLess(bear.pct_of_credit, bull.pct_of_credit)
        self.assertAlmostEqual(bear.pct_of_credit, 0.23, delta=0.03)

    def test_every_recorded_entry_carries_its_sample_size_and_source(self):
        for key, cell in fr.RECORDED.items():
            self.assertGreater(cell["n"], 0, key)
            self.assertTrue(cell["source"], key)


class HoldToExpiryTest(unittest.TestCase):
    def test_holding_to_expiry_halves_the_toll(self):
        managed = _profile("put_spread_ivr50")
        held = _profile("put_spread_ivr50_hold")
        self.assertFalse(held.round_trip)
        self.assertAlmostEqual(held.pct_of_credit, managed.pct_of_credit / 2.0,
                               places=6)


class UnmeasuredTest(unittest.TestCase):
    def test_a_structure_with_no_quotes_reports_unmeasured_not_zero(self):
        p = _profile("covered_call_holdings")
        self.assertFalse(p.measured)
        self.assertIsNone(p.pct_of_credit)
        self.assertEqual(p.n, 0)

    def test_an_unmeasured_profile_renders_as_a_dash(self):
        self.assertEqual(fr.format_cell(_profile("covered_call_holdings")),
                         "—")

    def test_a_measured_profile_renders_as_a_percentage_of_credit(self):
        """Bear Call's published toll is 23% of credit; the cell says so."""
        p = _profile("call_spread_extended")
        self.assertEqual(fr.format_cell(p), "23%")
        self.assertEqual(fr.format_cell(_profile("put_spread_ivr50")),
                         f"{_profile('put_spread_ivr50').pct_of_credit:.0%}")


class LedgerMeasurementTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.dir.name, "ledger.db")
        self.addCleanup(self.dir.cleanup)

    def test_measures_median_crossing_cost_per_structure(self):
        _ledger([("Bull Put", 1.00, 0.60)] * 12, self.path)
        table = fr.measure_from_ledger(self.path)
        self.assertEqual(table["bull_put"]["n"], 12)
        self.assertAlmostEqual(table["bull_put"]["per_share"], 0.40)
        self.assertAlmostEqual(table["bull_put"]["credit"], 1.00)

    def test_a_thin_bucket_does_not_set_a_constant(self):
        """Three quotes cannot overrule the recorded figure."""
        _ledger([("Bull Put", 1.00, 0.10)] * 3, self.path)
        table = fr.load_table(self.path)
        self.assertAlmostEqual(table["bull_put"]["per_share"],
                               fr.RECORDED["bull_put"]["per_share"])

    def test_a_full_bucket_overrules_the_recorded_figure(self):
        _ledger([("Bull Put", 1.00, 0.60)] * 20, self.path)
        table = fr.load_table(self.path)
        self.assertAlmostEqual(table["bull_put"]["per_share"], 0.40)
        self.assertIn("ledger", table["bull_put"]["source"])

    def test_rows_without_both_prices_are_skipped(self):
        _ledger([("Bull Put", 1.00, None)] * 5 + [("Bull Put", 1.00, 0.60)] * 10,
                self.path)
        self.assertEqual(fr.measure_from_ledger(self.path)["bull_put"]["n"], 10)

    def test_a_missing_ledger_falls_back_to_the_recorded_table(self):
        table = fr.load_table(os.path.join(self.dir.name, "nope.db"))
        self.assertAlmostEqual(table["bull_put"]["per_share"],
                               fr.RECORDED["bull_put"]["per_share"])

    def test_load_table_default_argument_branch(self):
        """The no-argument call is the one the desk actually makes."""
        self.assertIn("bull_put", fr.load_table())


class CeilingTest(unittest.TestCase):
    """The config ceiling is what makes the column actionable rather than trivia."""

    def test_the_ceiling_comes_from_config(self):
        self.assertGreater(fr.ceiling(), 0.0)
        self.assertLessEqual(fr.ceiling(), 1.0)

    def test_a_setup_over_the_ceiling_is_styled_bad(self):
        self.assertEqual(fr.style_for(_profile("put_spread_ivr50")),
                         "bad")

    def test_a_setup_approaching_the_ceiling_is_styled_warn(self):
        """23% of credit is under the 25% ceiling, but only just."""
        self.assertEqual(fr.style_for(_profile("call_spread_extended")),
                         "warn")

    def test_a_genuinely_cheap_structure_is_styled_good(self):
        cheap = fr.FrictionProfile("iron_condor", per_share=0.175, credit=9.64,
                                   round_trip=True, n=59, source="test")
        self.assertEqual(fr.style_for(cheap), "good")

    def test_an_unmeasured_setup_is_styled_muted(self):
        self.assertEqual(
            fr.style_for(_profile("covered_call_holdings")), "muted")

    def test_over_the_ceiling_is_reported_as_such(self):
        self.assertTrue(_profile("put_spread_ivr50").over_ceiling())
        self.assertFalse(_profile("call_spread_extended").over_ceiling())


class RecordOverrideTest(unittest.TestCase):
    def test_a_setups_own_cost_profile_wins_over_the_table(self):
        r = _rec("put_spread_ivr50").amend(
            "cost_profile", {"per_share": 0.01, "credit": 1.00, "n": 40,
                             "source": "backtest 2026-09-01"},
            reason="landed from a backtest", date="2026-09-01")
        p = fr.profile_for(r, table=fr.RECORDED)
        self.assertAlmostEqual(p.pct_of_credit, 0.02)
        self.assertIn("backtest", p.source)


if __name__ == "__main__":
    unittest.main()
