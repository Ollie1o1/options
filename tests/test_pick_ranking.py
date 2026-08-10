"""Tests for src/pick_ranking.py — what a board may show, and what it may claim.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_pick_ranking -v

The gates are scoped to the population each was measured on. Most of what is
asserted here is that a gate does NOT fire outside its evidence — a rule found
on condors must not quietly judge a long call.
"""
import sqlite3
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src import pick_ranking as pr


def _leg(**over):
    """A quotable long call that clears every gate."""
    row = {"strategy_name": "Long Call", "symbol": "AAPL",
           "bid": 9.90, "ask": 10.10, "quality_score": 0.50,
           "ev_per_contract": 25.0, "theta": -0.05, "premium": 10.0}
    row.update(over)
    return row


def _condor(**over):
    row = {"strategy_name": "Iron Condor", "symbol": "SPY",
           "short_put_bid": 2.00, "short_put_ask": 2.10,
           "long_put_bid": 1.00, "long_put_ask": 1.10,
           "short_call_bid": 2.00, "short_call_ask": 2.10,
           "long_call_bid": 1.00, "long_call_ask": 1.10,
           "spread_width": 5.0, "quality_score": 0.50,
           "ev_per_contract": 15.0, "theta": -0.02, "premium": 2.0}
    row.update(over)
    return row


def _board(rows):
    return pr.gate_board(pd.DataFrame(rows), db_path="/nonexistent.db")


class CondorUniverseGateTest(unittest.TestCase):
    """G5. Measured on n=139: +9.5% on broad index, -11.8% off it, p < 1e-5."""

    def test_a_condor_on_a_single_name_is_refused(self):
        r = _board([_condor(symbol="AAPL")])
        self.assertTrue(r.empty)
        self.assertEqual(r.reasons["condor_universe"], 1)

    def test_a_condor_on_spy_is_kept(self):
        r = _board([_condor(symbol="SPY")])
        self.assertEqual(len(r.kept), 1)

    def test_every_broad_index_underlying_is_allowed(self):
        rows = [_condor(symbol=s) for s in pr.BROAD_INDEX]
        self.assertEqual(len(_board(rows).kept), len(pr.BROAD_INDEX))

    def test_the_gate_does_not_touch_single_legs_on_the_same_ticker(self):
        """The rule was measured on condors. It says nothing about a long call
        on AAPL and must not be applied to one."""
        r = _board([_leg(symbol="AAPL")])
        self.assertEqual(len(r.kept), 1)


class BoardRefusesButTheLedgerKeepsLearningTest(unittest.TestCase):
    """The board/auto-log asymmetry is deliberate. Do not "fix" it.

    G5 refuses off-index condors on the board. The auto-logger keeps writing
    them as `paper_only=1` research rows, because gating there would freeze the
    off-index sample at the n=139 that produced the rule — and a rule that can
    only ever be confirmed by its own training set is not a finding.

    Ruled 2026-08-10. If this ever needs revisiting, the question is whether
    `scripts/validate_gates.py` still supports G5 on the grown sample.
    """

    def test_the_board_refuses_an_off_index_condor(self):
        r = _board([_condor(symbol="AAPL")])
        self.assertTrue(r.empty)

    def test_the_auto_log_ordering_never_drops_a_row(self):
        """`rank_structures_by_verdict` orders; it must not gate.

        A behavioural guard, not a comment: adding a filter here would silently
        stop the off-index condor sample from growing, which is exactly what
        makes G5 falsifiable.
        """
        from src.options_screener import rank_structures_by_verdict
        rows = pd.DataFrame([_condor(symbol=s) for s in
                             ("SPY", "AAPL", "TLT", "QQQ", "NVDA")])
        out = rank_structures_by_verdict(rows)
        self.assertEqual(len(out), len(rows),
                         "the auto-log path dropped a structure — that freezes "
                         "the sample G5 is meant to stay testable against")
        self.assertEqual(set(out["symbol"]), set(rows["symbol"]))


class TopQuintileGateTest(unittest.TestCase):
    """G6. The top quintile of the composite ran -15.7% and -$10,173."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = str(Path(self.dir.name) / "ledger.db")
        pr._top_quintile_cutoff.cache_clear()

    def tearDown(self):
        self.dir.cleanup()
        pr._top_quintile_cutoff.cache_clear()

    def _ledger(self, scores):
        conn = sqlite3.connect(self.db)
        conn.execute("create table trades (quality_score real, status text, "
                     "duplicate_of int, strategy_name text)")
        conn.executemany(
            "insert into trades values (?, 'CLOSED', null, 'Long Call')",
            [(s,) for s in scores])
        conn.commit()
        conn.close()

    def test_the_cutoff_is_the_eightieth_percentile_of_the_ledger(self):
        self._ledger([i / 100 for i in range(100)])
        self.assertAlmostEqual(pr.top_quintile_cutoff(self.db), 0.792, places=2)

    def test_a_thin_ledger_disables_the_gate_rather_than_guessing(self):
        self._ledger([0.5] * 5)
        self.assertEqual(pr.top_quintile_cutoff(self.db), float("inf"))

    def test_a_missing_ledger_disables_the_gate(self):
        self.assertEqual(pr.top_quintile_cutoff("/nonexistent.db"), float("inf"))

    def test_a_top_quintile_long_call_is_refused(self):
        self._ledger([i / 100 for i in range(100)])
        r = pr.gate_board(pd.DataFrame([_leg(quality_score=0.95)]), db_path=self.db)
        self.assertTrue(r.empty)
        self.assertEqual(r.reasons["top_quintile"], 1)

    def test_a_mid_score_long_call_is_kept(self):
        self._ledger([i / 100 for i in range(100)])
        r = pr.gate_board(pd.DataFrame([_leg(quality_score=0.50)]), db_path=self.db)
        self.assertEqual(len(r.kept), 1)

    def test_the_gate_does_not_touch_condors(self):
        """Measured on long single legs only. A high-scoring SPY condor is not
        judged by a threshold fitted on long calls."""
        self._ledger([i / 100 for i in range(100)])
        r = pr.gate_board(pd.DataFrame([_condor(quality_score=0.99)]), db_path=self.db)
        self.assertEqual(len(r.kept), 1)


class NegativeEvGateTest(unittest.TestCase):
    """G4 is a consistency gate: the board may not print BUY over a number it
    computed as negative."""

    def test_a_negative_ev_candidate_is_refused(self):
        r = _board([_leg(ev_per_contract=-18.0)])
        self.assertTrue(r.empty)
        self.assertEqual(r.reasons["negative_ev"], 1)

    def test_a_missing_ev_is_not_treated_as_negative(self):
        r = _board([_leg(ev_per_contract=None)])
        self.assertEqual(len(r.kept), 1)

    def test_a_nan_ev_is_not_treated_as_negative(self):
        r = _board([_leg(ev_per_contract=float("nan"))])
        self.assertEqual(len(r.kept), 1)


class CostGateTest(unittest.TestCase):
    """G1-G3 delegate to the existing cost work."""

    def test_an_unquotable_candidate_is_refused(self):
        r = _board([_leg(bid=None, ask=None)])
        self.assertTrue(r.empty)
        self.assertEqual(r.reasons["unquotable"], 1)

    def test_a_candidate_whose_friction_eats_the_reward_is_refused(self):
        r = _board([_leg(bid=6.00, ask=14.00)])
        self.assertTrue(r.empty)
        self.assertEqual(r.reasons["friction"], 1)


class RefusalAttributionTest(unittest.TestCase):
    """A candidate usually fails several gates; the board reports one."""

    def test_the_most_fundamental_failure_is_reported(self):
        r = _board([_condor(symbol="AAPL", bid=None, ask=None,
                            short_put_bid=None, short_put_ask=None,
                            ev_per_contract=-5.0)])
        self.assertEqual(r.reasons["unquotable"], 1)
        self.assertEqual(sum(r.reasons.values()), 1)

    def test_every_refusal_carries_an_evidence_class(self):
        for gate in pr.GATES.values():
            self.assertIn(gate.evidence, {pr.MEASURED, pr.ARITHMETIC, pr.CONSISTENCY})

    def test_summary_lines_describe_every_refusal(self):
        r = _board([_condor(symbol="AAPL"), _leg(ev_per_contract=-1.0)])
        self.assertEqual(len(r.summary_lines()), 2)


class OrderingTest(unittest.TestCase):
    """The board orders survivors. It does not rank them."""

    def test_survivors_come_back_cheapest_carry_first(self):
        rows = [_leg(symbol="A", theta=-0.50, premium=10.0),
                _leg(symbol="B", theta=-0.05, premium=10.0)]
        self.assertEqual(list(_board(rows).kept["symbol"]), ["B", "A"])

    def test_a_frame_without_ordering_columns_is_still_gated(self):
        """Ordering may degrade to unordered. It must never degrade to ungated.

        `df.get("theta")` returns None for a missing column, which `to_numeric`
        turned into a bare float; `.abs()` then raised and the failure-safe
        path returned every row, bypassing all six gates on any frame that
        happened to lack the column.
        """
        rows = [{"strategy_name": "Iron Condor", "symbol": "AAPL",
                 "short_put_bid": 2.0, "short_put_ask": 2.1,
                 "long_put_bid": 1.0, "long_put_ask": 1.1,
                 "short_call_bid": 2.0, "short_call_ask": 2.1,
                 "long_call_bid": 1.0, "long_call_ask": 1.1,
                 "spread_width": 5.0, "quality_score": 0.5}]
        r = pr.gate_board(pd.DataFrame(rows), db_path="/nonexistent.db")
        self.assertTrue(r.empty)
        self.assertEqual(r.reasons["condor_universe"], 1)

    def test_a_frame_without_ordering_columns_still_returns_survivors(self):
        rows = [{"strategy_name": "Long Call", "symbol": "AAPL",
                 "bid": 9.90, "ask": 10.10, "quality_score": 0.50,
                 "ev_per_contract": 25.0}]
        r = pr.gate_board(pd.DataFrame(rows), db_path="/nonexistent.db")
        self.assertEqual(len(r.kept), 1)

    def test_the_ordering_basis_is_disclosed(self):
        self.assertIn("not a ranking", _board([_leg()]).order_basis)

    def test_quality_score_does_not_order_the_board(self):
        """The composite is -0.131 on long calls. It must not decide what sits
        at the top of a board."""
        rows = [_leg(symbol="LOW", quality_score=0.10, theta=-0.50, premium=10.0),
                _leg(symbol="HIGH", quality_score=0.70, theta=-0.05, premium=10.0)]
        kept = _board(rows).kept
        self.assertEqual(kept.iloc[0]["symbol"], "HIGH")   # by carry, not score
        rows[0]["theta"], rows[1]["theta"] = -0.05, -0.50
        self.assertEqual(_board(rows).kept.iloc[0]["symbol"], "LOW")


class FailureSafetyTest(unittest.TestCase):
    """A crash must never be able to hide candidates."""

    def test_an_empty_frame_is_handled(self):
        r = pr.gate_board(pd.DataFrame())
        self.assertTrue(r.empty)
        self.assertEqual(r.scanned, 0)

    def test_none_is_handled(self):
        self.assertTrue(pr.gate_board(None).empty)

    def test_a_raising_gate_keeps_every_row(self):
        original = pr._refusal_for
        pr._refusal_for = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom"))
        try:
            df = pd.DataFrame([_leg(), _leg()])
            with self.assertLogs("src.pick_ranking", level="WARNING"):
                r = pr.gate_board(df, db_path="/nonexistent.db")
            self.assertEqual(len(r.kept), 2)
        finally:
            pr._refusal_for = original


if __name__ == "__main__":
    unittest.main()
