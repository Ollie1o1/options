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

import datetime as _dt
import importlib.util
import unittest

import numpy as np
import pandas as pd

# 30 days out, relative to today. A hardcoded date drifts out of the
# cost model's calibrated DTE range (and eventually into the past),
# which has nothing to do with what these tests measure.
_NEAR_EXP = (_dt.date.today() + _dt.timedelta(days=30)).isoformat()


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
                "ticker": "TEST", "expiration": _NEAR_EXP, "strike": 100.0,
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
                "ticker": "TEST", "expiration": _NEAR_EXP, "strike": 100.0,
                "type": "call", "entry_price": 2.5, "quality_score": 0.61,
                "strategy_name": "Long Call", "score_adjustments": "",
            })
            val = sqlite3.connect(db).execute(
                "select score_adjustments from trades").fetchone()[0]
        self.assertIsNone(val)


if __name__ == "__main__":
    unittest.main()


class TestBonusSuppression(unittest.TestCase):
    """The stack's BONUSES rank backwards; its penalties do not.

    Measured 2026-08-08 with `scripts/measure_adjustment_stack.py` on closed
    ledger rows, rank IC against friction-adjusted return:

                              Long Call/Put   Short Put   negative windows
      as shipped                  -0.0995      -0.0970      5/5 , 4/5
      stack OFF                   +0.0038      +0.0330      4/5 , 2/5
      penalties only              -0.0291      +0.0429      5/5 , 2/5
      BONUSES only                -0.1029      -0.1546      5/5 , 5/5

    Bonuses are negative in five windows of five in BOTH families, and rows the
    stack net-bonuses underperform rows it net-penalises (-0.195 vs -0.153 on
    long premium, -0.192 vs -0.058 on short puts). Penalties are mixed and are
    the single best variant for Short Put, so they are kept.

    Unlike the IV-rank result that failed its holdout the same day, these
    constants were hand-set and never fitted to this ledger, so measuring them
    negative on it is already out-of-sample evidence.

    The gate is on the NET per-row adjustment, which is exactly how the
    measurement builds its "penalties only" column (`composite +
    stack.clip(upper=0)`). So the shipped default IS that column: -0.0291 on
    long premium and +0.0429 on short puts, against -0.0995 / -0.0970 as
    shipped.

    Note stack-OFF scores slightly better than penalties-only on long premium
    (+0.0038 vs -0.0291) and slightly worse on short puts (+0.0330 vs
    +0.0429). That gap is not what is being optimised here: choosing between
    them on ~0.03 of IC at n=335 and n=109 would be the same small-sample
    tuning this repo keeps getting burned by. The penalties are kept because
    they encode risk — decay, gamma ramp, stale quotes, macro — and dropping
    risk guards to chase a third of a point of IC is a bad trade.
    """

    def _scored(self, scales=None):
        cfg = {}
        if scales is not None:
            cfg["scoring"] = {"adjustment_scales": scales}
        return _fx._run(_fx._make_chain(n=40), _fx._config(**cfg))

    def test_default_suppresses_bonuses_and_keeps_penalties(self):
        from src import options_screener as osc
        d = osc.DEFAULT_ADJUSTMENT_SCALES
        self.assertEqual(d["bonus"], 0.0)
        self.assertEqual(d["penalty"], 1.0)

    def test_a_bonus_scale_of_zero_never_scores_above_the_composite(self):
        out = self._scored()
        pre = out["quality_score_pre_adjust"]
        self.assertTrue((out["quality_score"] <= pre + 1e-9).all(),
                        "a row scored ABOVE its composite with bonuses off")

    def test_penalties_still_bite_at_the_default(self):
        out = self._scored()
        delta = out["quality_score"] - out["quality_score_pre_adjust"]
        self.assertLess(delta.min(), 0.0,
                        "no row was penalised — the stack is fully inert")

    def test_bonus_scale_is_a_net_per_row_gate(self):
        """`bonus_scale` credits a row only when its NET adjustment is positive.

        Rewritten 2026-08-10. This was `test_scales_of_one_restore_upward_
        adjustments`, asserting `on["quality_score"].max() > off[...].max()`
        with the message "bonus_scale=1.0 must lift at least one row".

        It never verified that. The scale is applied as
        `bonus * delta.clip(lower=0)` where `delta` is the row's NET
        adjustment, and on this fixture only two adjustments fire —
        `trend_aligned` (bonus) and `oi_wall_warning` (penalty) — with the
        penalty larger, so every row is net negative, `clip(lower=0)` is zero
        everywhere, and `bonus_scale` provably changes nothing: the row-wise
        difference is exactly 0.0000 on every row, on every repeat.

        The two maxima it compared were therefore the same number plus float
        noise, agreeing to eleven significant figures. It passed on 3.12 and
        3.14 and failed on 3.11 (CI run 31413572815) because a comparison at
        the noise floor is a coin toss, not a test.

        What is asserted now is the designed behaviour: a row the penalties
        already put underwater gets no bonus credit, whatever the scale.
        """
        on = self._scored({"bonus": 1.0, "penalty": 1.0})
        off = self._scored({"bonus": 0.0, "penalty": 1.0})

        # Precondition, stated rather than assumed: if a future fixture ever
        # produces a net-positive row this test is measuring the wrong thing
        # and should fail loudly instead of passing vacuously.
        net = off["quality_score"] - off["quality_score_pre_adjust"]
        self.assertTrue((net <= 0).all(),
                        "fixture now has a net-positive row — this test no "
                        "longer covers the gate it was written for")

        stack_on = on["quality_score"] - on["quality_score_pre_adjust"]
        stack_off = off["quality_score"] - off["quality_score_pre_adjust"]
        self.assertTrue(
            ((stack_on - stack_off).abs() < 1e-9).all(),
            "bonus_scale changed a row whose net adjustment is negative")

    def test_both_scales_zero_removes_every_additive_adjustment(self):
        off = self._scored({"bonus": 0.0, "penalty": 0.0})
        pre = off["quality_score_pre_adjust"].clip(0, 1)
        # <= not ==: the post-stack risk MULTIPLIERS still apply, and they can
        # only reduce. Anything above the composite would mean an additive
        # adjustment survived the gate.
        self.assertTrue((off["quality_score"] <= pre + 1e-9).all())

    def test_the_pre_adjustment_composite_is_recorded_for_audit(self):
        # Without this column the stack's effect cannot be measured after the
        # fact, which is how it went unexamined for as long as it did.
        self.assertIn("quality_score_pre_adjust", self._scored().columns)

    def test_scores_stay_in_range(self):
        for scales in (None, {"bonus": 1.0, "penalty": 1.0}):
            out = self._scored(scales)
            self.assertGreaterEqual(out["quality_score"].min(), 0.0)
            self.assertLessEqual(out["quality_score"].max(), 1.0)
