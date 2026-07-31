"""Replaying the closed cohort through the sizing engine.

The load-bearing property is that sizing is a FILTER, not just a scale factor:
positions the caps round down to zero contracts drop out of the book entirely.
A replay that silently kept them would answer the wrong question.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.sizing_replay import format_report, load_long_stop, replay  # noqa: E402


def _row(score=70.0, pnl_pct=-0.5, pnl_usd=-150.0, entry=3.0, **kw):
    base = {"date": "2026-06-01", "ticker": "AAPL", "quality_score": score,
            "pnl_pct": pnl_pct, "pnl_usd": pnl_usd, "entry_price": entry,
            "capital_at_risk": entry * 100, "quantity": 1.0}
    base.update(kw)
    return base


class ReplayTest(unittest.TestCase):
    def test_expensive_contracts_size_to_zero_and_leave_the_book(self):
        # $1,000 account, 2% risk cap = $20 of risk; a $30 premium risks $1,500
        # at a -50% stop, so it cannot be opened at all.
        r = replay([_row(entry=30.0)], account_value=1_000.0, stop_fraction=-0.50)
        self.assertEqual(r["n_zeroed"], 1)
        self.assertEqual(r["n_sized"], 0)
        self.assertEqual(r["sized_net_pnl"], 0.0)
        # ...but the unsized book still counts it, which is the whole point
        self.assertEqual(r["unsized_net_pnl"], -150.0)

    def test_affordable_contracts_scale_the_pnl_by_contract_count(self):
        r = replay([_row(entry=1.0, pnl_usd=-50.0)],
                   account_value=100_000.0, stop_fraction=-0.50)
        self.assertEqual(r["n_sized"], 1)
        self.assertGreater(r["contracts_total"], 1)
        self.assertAlmostEqual(r["sized_net_pnl"],
                               -50.0 * r["contracts_total"])

    def test_the_ic_is_recomputed_over_survivors_only(self):
        # Two cheap trades survive; two expensive ones are dropped. The
        # survivor IC must be computed on the two, not on all four.
        rows = [_row(score=90.0, pnl_pct=1.0, entry=1.0),
                _row(score=10.0, pnl_pct=-1.0, entry=1.0),
                _row(score=90.0, pnl_pct=-1.0, entry=500.0),
                _row(score=10.0, pnl_pct=1.0, entry=500.0)]
        r = replay(rows, account_value=20_000.0, stop_fraction=-0.50)
        self.assertEqual(r["n_zeroed"], 2)
        self.assertEqual(r["ic_survivors"]["n"], 2)
        self.assertEqual(r["ic_full_cohort"]["n"], 4)

    def test_a_degenerate_survivor_set_reports_none_rather_than_zero(self):
        # Fewer than 3 survivors cannot support a correlation. Reporting 0.000
        # would read as "measured, no relationship" instead of "not measurable".
        r = replay([_row(entry=1.0)], account_value=50_000.0, stop_fraction=-0.50)
        self.assertIsNone(r["ic_survivors"]["pearson"])
        self.assertIn("degenerate", format_report(r))

    def test_report_flags_cumulative_capital_when_it_exceeds_the_account(self):
        # Each trade is sized against the full account, so summed cost basis
        # can exceed it. That must be labelled, not read as peak exposure.
        rows = [_row(entry=1.0, pnl_usd=-10.0) for _ in range(40)]
        r = replay(rows, account_value=5_000.0, stop_fraction=-0.50)
        if r["capital_deployed"] > r["account_value"]:
            self.assertIn("not peak", format_report(r))

    def test_stop_fraction_is_read_from_config_not_hardcoded(self):
        stop = load_long_stop("config.json")
        self.assertLess(stop, 0.0)      # a stop is a loss fraction
        self.assertGreaterEqual(stop, -1.0)

    def test_zero_contract_trades_do_not_contribute_deployed_capital(self):
        r = replay([_row(entry=30.0)], account_value=1_000.0, stop_fraction=-0.50)
        self.assertEqual(r["capital_deployed"], 0.0)
        self.assertIsNone(r["return_on_deployed"])


if __name__ == "__main__":
    unittest.main()
