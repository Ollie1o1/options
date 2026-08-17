"""The breakeven margin is judged against the MANAGED required win rate.

`Worth.breakeven_margin` was `historical_win_rate - verdict.breakeven`, and
`verdict.breakeven` is `1 - credit/width` — the HOLD-TO-EXPIRY rate, computed
only when a row carries a `spread_width`. Two consequences:

* single legs have no width, so the margin was None and the board's Breakeven
  column read `n/a` on every row of every single-leg mode;
* where it WAS populated it used a basis this repo has already ruled wrong:
  "Judge against the managed rate, never `1-credit/width`"
  ([[project_required_win_rate_managed_exits]], 2026-08-13).

Extending the hold-to-expiry formula to single legs is the obvious fix and is
badly wrong. A cash-secured put's risk is the collateral and its reward is the
credit, so p* comes out at 98.7-99.0% — arithmetically right, and the right
answer to the wrong question, because it assumes you ride the put to TOTAL
LOSS. Nobody does; the config manages at take_profit 0.5 / stop_loss -0.25.
Every short put would have been refused.

The managed rate is measured from how trades ACTUALLY closed:

    p* = mean_loss / (mean_loss + mean_win)      over CLOSED trades

which reproduces the recorded figures exactly — Bull Put 50.9% needed against
66.4% delivered, +15.6pp; Bear Call -7.4pp; Iron Condor -10.4pp.

NOT wired into the refusal gate, deliberately. This rate is per-STRATEGY, not
per-contract, so gating on it would refuse whole families rather than
individual candidates — a strategy ban wearing a breakeven's clothes. The
gate keeps its per-contract check.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest


def _ledger(rows):
    """A tempfile ledger. Never the real book."""
    path = os.path.join(tempfile.mkdtemp(), "t.db")
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE trades (strategy_name TEXT, status TEXT, pnl_pct REAL)")
    conn.executemany("INSERT INTO trades VALUES (?,?,?)", rows)
    conn.commit()
    conn.close()
    return path


def _rates(path):
    from src.candidate_verdict import required_win_rates_from_ledger
    return required_win_rates_from_ledger(path)


class TestTheManagedRate(unittest.TestCase):

    def test_symmetric_wins_and_losses_need_half(self):
        rows = [("Bull Put", "CLOSED", 0.50)] * 10 + [("Bull Put", "CLOSED", -0.50)] * 10
        self.assertAlmostEqual(_rates(_ledger(rows))["Bull Put"], 0.50, places=6)

    def test_a_bigger_average_loss_needs_a_higher_win_rate(self):
        rows = [("Bull Put", "CLOSED", 0.25)] * 10 + [("Bull Put", "CLOSED", -0.75)] * 10
        # p* = 0.75 / (0.75 + 0.25)
        self.assertAlmostEqual(_rates(_ledger(rows))["Bull Put"], 0.75, places=6)

    def test_a_bigger_average_win_needs_a_lower_win_rate(self):
        rows = [("Bull Put", "CLOSED", 0.80)] * 10 + [("Bull Put", "CLOSED", -0.20)] * 10
        self.assertAlmostEqual(_rates(_ledger(rows))["Bull Put"], 0.20, places=6)

    def test_it_is_measured_per_strategy(self):
        rows = ([("A", "CLOSED", 0.50)] * 10 + [("A", "CLOSED", -0.50)] * 10 +
                [("B", "CLOSED", 0.80)] * 10 + [("B", "CLOSED", -0.20)] * 10)
        got = _rates(_ledger(rows))
        self.assertAlmostEqual(got["A"], 0.50, places=6)
        self.assertAlmostEqual(got["B"], 0.20, places=6)

    def test_zero_pnl_counts_as_a_loss_not_a_win(self):
        """A scratch is not a win; treating it as one flatters the rate."""
        rows = [("X", "CLOSED", 0.50)] * 10 + [("X", "CLOSED", 0.0)] * 10
        self.assertIn("X", _rates(_ledger(rows)))


class TestItRefusesThinEvidence(unittest.TestCase):
    """Same discipline as `win_rates_from_ledger`: an absent number is safer
    than one driven by five trades."""

    def test_under_twenty_closed_trades_is_omitted(self):
        rows = [("Thin", "CLOSED", 0.5)] * 9 + [("Thin", "CLOSED", -0.5)] * 9
        self.assertNotIn("Thin", _rates(_ledger(rows)))

    def test_all_winners_has_no_defined_rate(self):
        rows = [("AllWin", "CLOSED", 0.5)] * 30
        self.assertNotIn("AllWin", _rates(_ledger(rows)))

    def test_all_losers_has_no_defined_rate(self):
        rows = [("AllLose", "CLOSED", -0.5)] * 30
        self.assertNotIn("AllLose", _rates(_ledger(rows)))

    def test_open_trades_are_not_counted(self):
        rows = ([("Op", "CLOSED", 0.5)] * 10 + [("Op", "CLOSED", -0.5)] * 10 +
                [("Op", "OPEN", 9.9)] * 50)
        self.assertAlmostEqual(_rates(_ledger(rows))["Op"], 0.50, places=6)

    def test_a_missing_ledger_returns_empty_rather_than_raising(self):
        self.assertEqual(_rates("/nonexistent/nope.db"), {})


class TestTheMarginReachesTheCard(unittest.TestCase):
    """`Worth.breakeven_margin` is the number the board renders."""

    def test_a_single_leg_now_has_a_margin(self):
        """The n/a this whole change exists to remove.

        The rate is passed in rather than looked up: `paper_trades.db` is
        gitignored, so a test that reads it passes locally and fails on CI —
        which is exactly what this one did on its first run. The lookup path
        is covered above against tempfile ledgers.
        """
        from src.worth import assess
        row = {"ev_per_contract": 40.0, "vega_dollar": 20.0,
               "hv_252d": 0.25, "hv_30d": 0.25, "strategy_name": "Short Put",
               "expiration": "2026-09-18", "date": "2026-08-17"}
        w = assess(row, historical_win_rate=0.495, required_win_rate=0.519)
        self.assertIsNotNone(w.breakeven_margin,
                             "single-leg breakeven margin is still absent")
        self.assertAlmostEqual(w.breakeven_margin, 0.495 - 0.519, places=6)

    def test_the_margin_is_history_minus_the_managed_rate(self):
        from src.worth import assess
        row = {"ev_per_contract": 40.0, "vega_dollar": 20.0,
               "hv_252d": 0.25, "hv_30d": 0.25, "strategy_name": "Bull Put",
               "expiration": "2026-09-18", "date": "2026-08-17"}
        w = assess(row, historical_win_rate=0.664, required_win_rate=0.509)
        self.assertAlmostEqual(w.breakeven_margin, 0.664 - 0.509, places=6)

    def test_no_required_rate_leaves_it_absent_rather_than_zero(self):
        from src.worth import assess
        row = {"ev_per_contract": 40.0, "vega_dollar": 20.0,
               "hv_252d": 0.25, "hv_30d": 0.25, "strategy_name": "Unknown",
               "expiration": "2026-09-18", "date": "2026-08-17"}
        w = assess(row, historical_win_rate=0.5, required_win_rate=None)
        self.assertIsNone(w.breakeven_margin)


class TestTheGateIsUntouched(unittest.TestCase):
    """A per-STRATEGY rate in a per-contract gate is a family ban."""

    def test_verdict_still_uses_the_per_contract_rate(self):
        from src.paths import repo_path
        with open(repo_path("src/candidate_verdict.py")) as fh:
            src = fh.read()
        gate = src[src.index("def verdict_for("):src.index("def rank(")]
        self.assertIn("et.breakeven_win_rate(", gate,
                      "the per-contract breakeven left the gate")
        self.assertNotIn("required_win_rates_from_ledger", gate,
                         "the per-strategy managed rate reached the gate — "
                         "that refuses whole families, not candidates")


if __name__ == "__main__":
    unittest.main()


class TestEveryBoardCarriesItsStrategyName(unittest.TestCase):
    """Without it the managed rate cannot be looked up and Breakeven is n/a.

    Only Premium Selling routed through `rank_single_legs_by_verdict`, which
    labels rows. Discovery, Budget scan and Single-stock call `gate_and_report`
    directly, so their rows reached the board unlabelled and every one showed
    `n/a` — not for want of a rate (Long Call 38.8%, Long Put 37.6% both exist)
    but for want of a key to look it up with.
    """

    def test_the_budget_board_labels_an_unlabelled_frame(self):
        import pandas as pd
        from src import options_screener as osx
        df = pd.DataFrame([{"symbol": "SPY", "type": "call", "ask": 5.0},
                           {"symbol": "SPY", "type": "put", "ask": 4.0}])
        out = osx._budget_board(
            df, lambda r: osx._strategy_label_for_mode("Discovery scan",
                                                       r.get("type")),
            None, verbose=False)
        self.assertIn("strategy_name", out.columns)
        self.assertEqual(sorted(out["strategy_name"].unique()),
                         ["Long Call", "Long Put"])

    def test_an_existing_label_is_not_overwritten(self):
        import pandas as pd
        from src import options_screener as osx
        df = pd.DataFrame([{"symbol": "SPY", "type": "put", "ask": 4.0,
                            "strategy_name": "Short Put"}])
        out = osx._budget_board(
            df, lambda r: osx._strategy_label_for_mode("Discovery scan",
                                                       r.get("type")),
            None, verbose=False)
        self.assertEqual(out["strategy_name"].iloc[0], "Short Put")
