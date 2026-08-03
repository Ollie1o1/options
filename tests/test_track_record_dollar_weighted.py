"""Dollar-weighted public track record (scripts/publish_track_record.py).

The published headline used to be an unweighted mean of per-trade percentage
returns. Risk per trade in this book spans three orders of magnitude — a $28
credit spread and a $27,000 cash-secured put counted the same — so that number
described a portfolio nobody could have held. These tests pin the replacements:

- headline is net dollars and return on total capital risked,
- per-strategy lines carry net $ and a MEDIAN return on risk, so one large
  contract cannot carry a line unnoticed,
- the win rate states its numerator and denominator and what it excluded,
- cash-secured shorts publish return on credit beside return on risk, because
  their collateral denominator flattens return on risk to nearly zero whatever
  the trade did.

Pure computation against seeded rows — no db file, no network.
"""

import os
import sqlite3
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.publish_track_record import (  # noqa: E402
    credit_collected,
    fetch_closed_trades,
    render_track_record,
    sign_divergences,
    summarize_book,
    summarize_credit_strategies,
    summarize_strategies,
)

_EVIDENCE = {
    "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94,
    "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-07",
}


def _row(**kw):
    base = {
        "date": "2026-05-10", "ticker": "AAPL", "strategy_name": "Long Call",
        "type": "call", "strike": 150.0, "expiration": "2026-06-19",
        "entry_price": 3.0, "exit_price": 4.0, "pnl_pct": 0.33, "pnl_usd": 100.0,
        "exit_reason": "target", "paper_only": 0, "status": "CLOSED",
        "capital_at_risk": 300.0, "net_credit": None, "quantity": 1.0,
    }
    base.update(kw)
    return base


class TestDollarWeightedHeadline(unittest.TestCase):
    def test_return_on_risk_is_dollar_weighted_not_trade_averaged(self):
        # A big loser and a small winner: the unweighted mean is positive,
        # the dollar-weighted return is negative. The headline must be the
        # dollar-weighted one.
        rows = [
            _row(pnl_usd=-500.0, pnl_pct=-0.10, capital_at_risk=5000.0),
            _row(pnl_usd=50.0, pnl_pct=0.50, capital_at_risk=100.0),
        ]
        book = summarize_book(rows)
        self.assertAlmostEqual(book["net_pnl"], -450.0)
        self.assertAlmostEqual(book["capital_at_risk"], 5100.0)
        self.assertAlmostEqual(book["return_on_risk"], -450.0 / 5100.0)
        self.assertGreater(book["mean_return_unweighted"], 0)  # the old headline
        self.assertLess(book["return_on_risk"], 0)

    def test_rows_without_capital_at_risk_are_excluded_from_the_denominator(self):
        rows = [
            _row(pnl_usd=100.0, capital_at_risk=1000.0),
            _row(pnl_usd=25.0, capital_at_risk=None),
            _row(pnl_usd=25.0, capital_at_risk=0.0),
        ]
        book = summarize_book(rows)
        self.assertEqual(book["n_risked"], 1)
        self.assertAlmostEqual(book["capital_at_risk"], 1000.0)
        self.assertAlmostEqual(book["return_on_risk"], 0.10)
        # net P&L still counts every dollar recorded
        self.assertAlmostEqual(book["net_pnl"], 150.0)

    def test_no_capital_at_risk_anywhere_degrades_to_none(self):
        book = summarize_book([_row(capital_at_risk=None)])
        self.assertIsNone(book["return_on_risk"])
        self.assertIsNone(book["capital_at_risk"])

    def test_affordable_subset_uses_the_configured_ceiling(self):
        rows = [
            _row(pnl_usd=-900.0, capital_at_risk=9000.0),
            _row(pnl_usd=100.0, capital_at_risk=500.0),
        ]
        book = summarize_book(rows, budget_cap=4000.0)
        self.assertLess(book["return_on_risk"], 0)
        self.assertEqual(book["affordable"]["n"], 1)
        self.assertAlmostEqual(book["affordable"]["return_on_risk"], 0.20)

    def test_headline_renders_dollars_and_basis(self):
        md = render_track_record(
            [_row(pnl_usd=-500.0, pnl_pct=-0.10, capital_at_risk=5000.0),
             _row(pnl_usd=50.0, pnl_pct=0.50, capital_at_risk=100.0)],
            _EVIDENCE, budget_cap=4000.0)
        self.assertIn("## Headline", md)
        self.assertIn("Net P&L", md)
        self.assertIn("-$450.00", md)
        self.assertIn("Return on capital risked", md)
        self.assertIn("of capital risked", md)
        self.assertIn("unweighted mean of per-trade returns **on entry premium**", md)
        # the headline precedes the size-blind secondary figures
        self.assertLess(md.index("Return on capital risked"),
                        md.index("Mean return per trade"))

    def test_the_ratio_quotes_the_numerator_it_actually_divides(self):
        # One trade carries a dollar result but no capital at risk. The book's
        # net is -$450 while the ratio's numerator is only the risked -$500,
        # so rendering net_pnl beside the risked denominator would publish
        # arithmetic that does not reconcile.
        rows = [
            _row(pnl_usd=-500.0, pnl_pct=-0.10, capital_at_risk=5000.0),
            _row(pnl_usd=50.0, pnl_pct=0.50, capital_at_risk=None),
        ]
        book = summarize_book(rows)
        self.assertAlmostEqual(book["net_pnl"], -450.0)
        self.assertAlmostEqual(book["net_pnl_risked"], -500.0)
        self.assertAlmostEqual(book["return_on_risk"], -500.0 / 5000.0)

        md = render_track_record(rows, _EVIDENCE)
        line = next(ln for ln in md.splitlines()
                    if "Return on capital risked" in ln)
        self.assertIn("-$500.00", line)
        self.assertNotIn("-$450.00", line)
        # and the gap between the two is stated rather than left to be noticed
        self.assertIn("no recorded capital at risk", md)

    def test_premium_basis_is_stated_for_the_closed_trades_table(self):
        md = render_track_record([_row()], _EVIDENCE)
        self.assertIn("| P&L % of premium |", md)
        self.assertIn("debit paid on Long Call", md)
        self.assertIn("credit received on Short Put", md)


class TestMedianReturnOnRisk(unittest.TestCase):
    def test_median_is_not_carried_by_one_large_winner(self):
        # Four small losers and one large winner: aggregate is positive,
        # the typical trade is not.
        rows = [_row(strategy_name="Long Put", pnl_usd=-100.0, pnl_pct=-0.20,
                     capital_at_risk=500.0) for _ in range(4)]
        rows.append(_row(strategy_name="Long Put", ticker="GS", pnl_usd=3000.0,
                         pnl_pct=1.5, capital_at_risk=2000.0))
        (s,) = summarize_strategies(rows)
        self.assertEqual(s["strategy"], "Long Put")
        self.assertAlmostEqual(s["return_on_risk"], 2600.0 / 4000.0)
        self.assertAlmostEqual(s["median_return_on_risk"], -0.20)
        self.assertAlmostEqual(s["net_pnl"], 2600.0)

    def test_sign_divergence_is_detected_and_explained_in_the_notes(self):
        rows = [_row(strategy_name="Long Put", pnl_usd=-100.0, pnl_pct=-0.20,
                     capital_at_risk=500.0) for _ in range(4)]
        rows.append(_row(strategy_name="Long Put", ticker="GS", pnl_usd=3000.0,
                         pnl_pct=1.5, capital_at_risk=2000.0))
        diverging = sign_divergences(summarize_strategies(rows))
        self.assertEqual([d["strategy"] for d in diverging], ["Long Put"])
        md = render_track_record(rows, _EVIDENCE)
        self.assertIn("Median vs aggregate", md)
        self.assertIn("GS", md)

    def test_breakdown_return_on_risk_is_reused_when_supplied(self):
        # get_strategy_breakdown() is the existing definition of return_on_risk;
        # when it is passed in, the published number is its number.
        rows = [_row(strategy_name="Long Call", pnl_usd=100.0,
                     capital_at_risk=1000.0)]
        breakdown = [{"strategy": "Long Call", "return_on_risk": 0.4242}]
        (s,) = summarize_strategies(rows, breakdown=breakdown)
        self.assertAlmostEqual(s["return_on_risk"], 0.4242)

    def test_the_divergence_explanation_names_the_big_loser_not_the_big_winner(self):
        # Aggregate negative, median positive: the trade that explains the line
        # is the large loser. Picking the largest signed P&L would name a small
        # winner and publish a confidently wrong sentence.
        rows = [_row(strategy_name="Iron Condor", pnl_usd=60.0, pnl_pct=0.20,
                     capital_at_risk=300.0) for _ in range(4)]
        rows.append(_row(strategy_name="Iron Condor", ticker="TLT",
                         pnl_usd=-4000.0, pnl_pct=-1.5, capital_at_risk=2000.0))
        (s,) = summarize_strategies(rows)
        self.assertLess(s["return_on_risk"], 0)
        self.assertGreater(s["median_return_on_risk"], 0)

        md = render_track_record(rows, _EVIDENCE)
        frag = next(ln for ln in md.splitlines()
                    if "Iron Condor" in ln and "one " in ln)
        self.assertIn("TLT", frag)
        self.assertIn("-$4,000.00", frag)

    def test_a_zero_return_on_risk_from_the_breakdown_is_not_recomputed(self):
        # 0.0 is a legitimate value; a truthiness check would discard it and
        # silently fall back to the private recomputation.
        rows = [_row(strategy_name="Long Call", pnl_usd=100.0,
                     capital_at_risk=1000.0)]
        (s,) = summarize_strategies(
            rows, breakdown=[{"strategy": "Long Call", "return_on_risk": 0.0}])
        self.assertEqual(s["return_on_risk"], 0.0)

    def test_strategy_table_has_net_dollars_and_median_columns(self):
        md = render_track_record([_row()], _EVIDENCE)
        self.assertIn("| Net $ |", md)
        self.assertIn("Median return on risk", md)
        self.assertIn("(aggregate, of capital risked)", md)


class TestWinRateDenominator(unittest.TestCase):
    def test_numerator_denominator_and_exclusions_are_explicit(self):
        rows = [_row(pnl_pct=0.5), _row(pnl_pct=-0.5), _row(pnl_pct=None)]
        book = summarize_book(rows)
        self.assertEqual(book["n_closed"], 3)
        self.assertEqual(book["n_scored"], 2)
        self.assertEqual(book["n_wins"], 1)
        self.assertEqual(book["n_unscored"], 1)
        self.assertAlmostEqual(book["win_rate"], 0.5)

        md = render_track_record(rows, _EVIDENCE)
        self.assertIn("**50.0%** = 1 win / 2 closed trades with a recorded return",
                      md)
        self.assertIn("1 closed trade excluded for missing returns", md)

    def test_zero_returns_are_not_wins(self):
        book = summarize_book([_row(pnl_pct=0.0), _row(pnl_pct=0.1)])
        self.assertEqual(book["n_wins"], 1)


class TestReturnOnCredit(unittest.TestCase):
    def test_single_leg_short_credit_comes_from_entry_price(self):
        # net_credit is NULL on cash-secured shorts; the entry price is the credit.
        self.assertAlmostEqual(
            credit_collected(_row(strategy_name="Short Put", entry_price=1.52,
                                  net_credit=None, quantity=1.0)),
            152.0)

    def test_spread_credit_comes_from_net_credit_and_scales_with_contracts(self):
        self.assertAlmostEqual(
            credit_collected(_row(strategy_name="Bull Put", entry_price=9.9,
                                  net_credit=0.50, quantity=3.0)),
            150.0)

    def test_cash_secured_short_reports_credit_return_beside_flat_risk_return(self):
        # Collateral is ~50x the credit, so return on risk looks flat while
        # return on credit is large. Both must be published.
        rows = [
            _row(strategy_name="Short Put", entry_price=1.52, net_credit=None,
                 capital_at_risk=7598.0, pnl_usd=64.70, pnl_pct=0.42),
            _row(strategy_name="Short Put", entry_price=2.14, net_credit=None,
                 capital_at_risk=12486.0, pnl_usd=-66.14, pnl_pct=-0.31),
        ]
        (c,) = summarize_credit_strategies(rows)
        self.assertEqual(c["strategy"], "Short Put")
        self.assertTrue(c["cash_secured"])
        self.assertAlmostEqual(c["credit_collected"], 366.0)
        self.assertAlmostEqual(c["return_on_credit"], (64.70 - 66.14) / 366.0)
        self.assertLess(abs(c["return_on_risk"]), abs(c["return_on_credit"]))

        md = render_track_record(rows, _EVIDENCE)
        self.assertIn("return on credit collected", md.lower())
        self.assertIn("of credit collected", md)
        self.assertIn("Cash-secured collateral denominator", md)

    def test_debit_strategies_are_not_in_the_credit_table(self):
        out = summarize_credit_strategies([_row(strategy_name="Long Call")])
        self.assertEqual(out, [])


class TestBreakdownReuse(unittest.TestCase):
    """The publisher reuses PaperManager.get_strategy_breakdown() so the
    published return-on-risk and the portfolio view's are one definition. A
    rename on either side must fail here, not silently fall back to a private
    recomputation that can then drift."""

    def test_manager_symbol_and_return_on_risk_key_exist(self):
        from src.paper_manager import PaperManager

        self.assertTrue(hasattr(PaperManager, "get_strategy_breakdown"))

    def test_load_breakdown_returns_rows_for_a_real_db(self):
        import tempfile

        from scripts.publish_track_record import _load_breakdown

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            from src.paper_manager import PaperManager

            PaperManager(db_path=path)  # creates the schema
            conn = sqlite3.connect(path)
            conn.execute(
                "INSERT INTO trades (date, ticker, strategy_name, status, "
                "pnl_pct, pnl_usd, capital_at_risk) VALUES "
                "('2026-05-10','AAPL','Long Call','CLOSED',0.3,150.0,500.0)")
            conn.commit()
            conn.close()

            breakdown = _load_breakdown(path)
            self.assertIsNotNone(breakdown)
            self.assertEqual(len(breakdown), 1)
            self.assertAlmostEqual(breakdown[0]["return_on_risk"], 0.30)


class TestFetchToleratesOlderSchema(unittest.TestCase):
    def _seed(self, with_capital: bool):
        conn = sqlite3.connect(":memory:")
        extra = ", capital_at_risk REAL, net_credit REAL, quantity REAL" if with_capital else ""
        conn.execute(
            "CREATE TABLE trades (entry_id INTEGER, date TEXT, ticker TEXT, "
            "expiration TEXT, strike REAL, type TEXT, entry_price REAL, "
            "strategy_name TEXT, status TEXT, exit_price REAL, exit_date TEXT, "
            f"pnl_pct REAL, pnl_usd REAL, exit_reason TEXT, paper_only INTEGER{extra})"
        )
        vals = (1, "2026-05-10", "AAPL", "2026-06-19", 150, "call", 3.5,
                "Long Call", "CLOSED", 5.0, "2026-05-20", 0.428, 150.0,
                "target", 0)
        if with_capital:
            vals = vals + (350.0, None, 1.0)
        conn.execute(
            f"INSERT INTO trades VALUES ({','.join('?' * len(vals))})", vals)
        conn.commit()
        return conn

    def test_schema_without_capital_at_risk_still_renders(self):
        rows = fetch_closed_trades(self._seed(with_capital=False))
        self.assertEqual(len(rows), 1)
        md = render_track_record(rows, _EVIDENCE)
        self.assertIn("Return on capital risked: **n/a**", md)
        self.assertIn("+$150.00", md)  # dollars still published

    def test_schema_with_capital_at_risk_computes_the_headline(self):
        rows = fetch_closed_trades(self._seed(with_capital=True))
        book = summarize_book(rows)
        self.assertAlmostEqual(book["return_on_risk"], 150.0 / 350.0)


class EmptyLedgerIsSilentTest(unittest.TestCase):
    """The schema-mismatch warning exists to stop two definitions of
    return-on-risk drifting apart unnoticed. An empty ledger has no strategies
    to break down, so nothing is being skipped and there is nothing to drift.

    On a fresh clone the first startup created an empty ledger at
    `user_version = 0` and the publisher greeted the user with
    `warning: paper_trades.db is at schema v0, not v17` before anything had
    happened. A warning that fires when nothing is wrong is noise, and noise
    is what makes the real one easy to miss.
    """

    def _empty_db(self, path, user_version=0):
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (entry_id INTEGER PRIMARY KEY, "
                     "date TEXT, ticker TEXT, strategy_name TEXT, status TEXT, "
                     "pnl_pct REAL, pnl_usd REAL, capital_at_risk REAL)")
        conn.execute(f"PRAGMA user_version = {user_version}")
        conn.commit(); conn.close()

    def _warnings_from(self, path):
        import contextlib
        import io

        from scripts.publish_track_record import _load_breakdown

        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            result = _load_breakdown(path)
        return result, err.getvalue()

    def test_an_empty_ledger_warns_about_nothing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            self._empty_db(path)
            _result, warnings = self._warnings_from(path)
        self.assertEqual(warnings, "", "an empty ledger has nothing to skip")

    def test_an_old_ledger_with_rows_still_warns(self):
        # The drift this guards against is real once there are rows whose
        # breakdown is being skipped. That warning must survive.
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "t.db")
            self._empty_db(path)
            conn = sqlite3.connect(path)
            conn.execute(
                "INSERT INTO trades (date, ticker, strategy_name, status, "
                "pnl_pct, pnl_usd, capital_at_risk) VALUES "
                "('2026-05-10','AAPL','Long Call','CLOSED',0.3,150.0,500.0)")
            conn.commit(); conn.close()
            _result, warnings = self._warnings_from(path)
        self.assertIn("schema", warnings)

    def test_a_missing_ledger_is_not_a_crash(self):
        import tempfile

        with tempfile.TemporaryDirectory() as d:
            result, _warnings = self._warnings_from(os.path.join(d, "nope.db"))
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
