"""The post-composite adjustment stack: one scale, a floor, and a record of it.

`quality_score` is a 27-component weighted average and then ~20 hand-set
additions and multipliers. Measured 2026-08-07 on a chain scored through the
real pipeline: those adjustments can subtract 1.28 and add 0.47, against a
composite whose whole documented range spans 0.54 and whose observed spread was
0.29. A single `decay_warning` at -0.20 outweighs any one component; two
penalties outweigh all 27 together.

The constants are not touched here — none of them has ever been measured, and
re-tuning by taste is exactly what the audit warned against. What is fixed is
the two defects around them, and what is added is the record that makes the
constants answerable later.
"""
from __future__ import annotations

import importlib.util
import unittest

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "_scorer_fx", "tests/test_scorer_signal_recovery.py")
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)


class TestDisplayScaleIsOneScale(unittest.TestCase):
    """The squeeze board showed the RAW composite while the main table showed
    the NORMALISED one, under the same "Score" header. Raw 0.55 and display
    0.64 are the same contract.
    """

    def test_normalisation_is_a_pure_function_of_the_raw_score(self):
        from src.options_screener import _cross_section_normalize
        one = _cross_section_normalize(pd.DataFrame({"quality_score": [0.55]}))
        many = _cross_section_normalize(
            pd.DataFrame({"quality_score": [0.31, 0.55, 0.78]}))
        self.assertAlmostEqual(float(one["quality_score"].iloc[0]),
                               float(many["quality_score"].iloc[1]), places=6)

    def test_a_single_row_scan_is_normalised_too(self):
        """The old `n <= 1: return df` guard left a one-contract scan on the
        raw scale — the same contract on two scales depending on how many
        others happened to be fetched beside it."""
        from src.options_screener import _cross_section_normalize
        out = _cross_section_normalize(pd.DataFrame({"quality_score": [0.55]}))
        self.assertNotAlmostEqual(float(out["quality_score"].iloc[0]), 0.55,
                                  places=3)

    def test_empty_and_missing_column_are_survivable(self):
        from src.options_screener import _cross_section_normalize
        _cross_section_normalize(pd.DataFrame({"quality_score": []}))
        _cross_section_normalize(pd.DataFrame({"other": [1, 2]}))

    def test_mapping_endpoints_hold(self):
        from src.options_screener import _cross_section_normalize
        out = _cross_section_normalize(
            pd.DataFrame({"quality_score": [0.28, 0.82]}))
        self.assertAlmostEqual(out["quality_score"].iloc[0], 0.0, places=3)
        self.assertAlmostEqual(out["quality_score"].iloc[1], 1.0, places=3)


class TestScoreHasAFloor(unittest.TestCase):
    """A clip(0,1) sits partway through the stack, but three mutations follow
    it and none restores a floor — the residual crush penalty SUBTRACTS and the
    risk clip is `upper=` only. Below zero the risk multiplier inverted: at
    -0.030, three flags gave -0.026 and five gave -0.015, so MORE structural
    risk scored HIGHER.
    """

    def test_scored_output_is_never_negative(self):
        out = _fx._run(_fx._make_chain(n=40), _fx._config())
        if out.empty:
            self.skipTest("scan filtered everything")
        q = pd.to_numeric(out["quality_score"], errors="coerce")
        self.assertTrue((q >= 0).all(), f"negative score: {q.min()}")
        self.assertTrue((q <= 1).all(), f"score above 1: {q.max()}")

    def test_the_inversion_this_prevents(self):
        """Pins the arithmetic, so the floor is not removed as redundant."""
        below_zero = -0.030
        self.assertGreater(below_zero * 0.50, below_zero * 0.85,
                           "the premise is wrong; re-check before deleting "
                           "the final clamp")


class TestAdjustmentFlagsAreRecorded(unittest.TestCase):
    """The ledger stored every component score and no record of which flags
    fired, so `flag -> outcome` had no data behind it. This is that data.
    """

    def _flags(self, **cols):
        from src.options_screener import _score_adjustment_flags
        n = len(next(iter(cols.values()))) if cols else 1
        base = {"type": ["call"] * n}
        base.update(cols)
        return list(_score_adjustment_flags(pd.DataFrame(base)))

    def test_nothing_firing_is_an_empty_string(self):
        self.assertEqual(self._flags(decay_warning=[False]), [""])

    def test_penalties_are_named(self):
        got = self._flags(decay_warning=[True], gamma_ramp=[True],
                          sr_warning=["NEAR RESISTANCE"])
        self.assertIn("decay_warning", got[0])
        self.assertIn("gamma_ramp", got[0])
        self.assertIn("sr_warning", got[0])

    def test_bonuses_are_named(self):
        got = self._flags(Trend_Aligned=[True], short_interest=[0.31])
        self.assertIn("trend_aligned", got[0])
        self.assertIn("si_heavy", got[0])

    def test_the_multiplier_stage_is_recorded_separately(self):
        """Every flag risk_flag_count counts ALSO fired as an additive penalty
        — the double-count. Recording the level keeps the two stages separable
        in analysis instead of silently compounded."""
        got = self._flags(decay_warning=[True], risk_flag_count=[4])
        self.assertIn("risk_mult_4", got[0])
        self.assertIn("decay_warning", got[0])

    def test_below_the_multiplier_threshold_nothing_is_recorded(self):
        self.assertNotIn("risk_mult", self._flags(risk_flag_count=[2])[0])

    def test_rows_are_independent(self):
        got = self._flags(decay_warning=[True, False, False],
                          Trend_Aligned=[False, True, False])
        self.assertEqual(got[0], "decay_warning")
        self.assertEqual(got[1], "trend_aligned")
        self.assertEqual(got[2], "")

    def test_missing_columns_do_not_raise(self):
        from src.options_screener import _score_adjustment_flags
        self.assertEqual(list(_score_adjustment_flags(
            pd.DataFrame({"type": ["call", "put"]}))), ["", ""])

    def test_the_scan_pipeline_populates_it(self):
        out = _fx._run(_fx._make_chain(n=40), _fx._config())
        if out.empty:
            self.skipTest("scan filtered everything")
        self.assertIn("score_adjustments", out.columns)
        self.assertTrue(out["score_adjustments"].map(
            lambda v: isinstance(v, str)).all())


class TestLedgerRecordsTheFlags(unittest.TestCase):

    def test_round_trip_through_the_ledger(self):
        import os
        import sqlite3
        import tempfile

        from src.paper_manager import PaperManager
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "t.db")
            pm = PaperManager(db_path=db)
            pm.log_trade({
                "ticker": "TEST", "expiration": "2026-12-18", "strike": 100.0,
                "type": "call", "entry_price": 2.5, "quality_score": 0.61,
                "strategy_name": "Long Call", "trader_pref_score": 0.77,
                "score_adjustments": "decay_warning,risk_mult_3",
                "weight_profile": "baseline",
            })
            row = sqlite3.connect(db).execute(
                "select score_adjustments, trader_pref_score, weight_profile "
                "from trades").fetchone()
        self.assertEqual(row[0], "decay_warning,risk_mult_3")
        # Neighbours on both sides, so a column-order slip cannot pass.
        self.assertEqual(row[1], 0.77)
        self.assertEqual(row[2], "baseline")

    def test_no_flags_stores_NULL_not_empty_string(self):
        """NULL must mean 'not recorded' so pre-migration rows stay
        distinguishable from rows that genuinely had no flag fire. An analysis
        that reads NULL as empty would give the whole pre-2026-08-07 book a
        clean bill of health it was never given."""
        import os
        import sqlite3
        import tempfile

        from src.paper_manager import PaperManager
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "t.db")
            pm = PaperManager(db_path=db)
            pm.log_trade({
                "ticker": "TEST", "expiration": "2026-12-18", "strike": 100.0,
                "type": "call", "entry_price": 2.5, "quality_score": 0.61,
                "strategy_name": "Long Call", "score_adjustments": "",
            })
            val = sqlite3.connect(db).execute(
                "select score_adjustments from trades").fetchone()[0]
        self.assertIsNone(val)


if __name__ == "__main__":
    unittest.main()
