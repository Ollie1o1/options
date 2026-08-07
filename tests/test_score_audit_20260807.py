"""Audit fixes 2026-08-07: numbers that could not mean what they were labelled.

Each test pins one defect found by auditing how `quality_score`, `ev_per_contract`
and the squeeze board's columns are computed against what they are presented as.
Full write-up and the measurements behind each: docs/SCORE_AUDIT_20260807.md.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_score_audit_20260807 -v
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import unittest

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# 1. The IC weight blend must transmit the evidence, not the survivor count.
# --------------------------------------------------------------------------
class TestICWeightBlend(unittest.TestCase):
    """`load_ic_adjusted_weights` divided each surviving IC by the sum over
    *survivors*. With one survivor that ratio is 1.0 by construction, so the
    component absorbed the entire 0.30 reallocation budget no matter how weak
    its IC was — and lost that weight again the moment an unrelated component
    crossed p=0.10. Measured live 2026-08-07: theta held 24.3% of the composite
    on IC=+0.082, against a base weight of 0.0197.
    """

    BASE = {"pop": 0.10, "ev": 0.10, "rr": 0.10, "liquidity": 0.10,
            "momentum": 0.10, "iv_rank": 0.10, "theta": 0.10}

    def _blend(self, ic, pvals):
        import src.options_screener as S
        S._invalidate_ic_weights_cache()
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            with open(path, "w") as fh:
                json.dump({"component_ic": ic, "component_pvalues": pvals}, fh)
            return dict(S.load_ic_adjusted_weights(
                {"composite_weights": dict(self.BASE)}, cache_path=path))
        finally:
            os.unlink(path)
            S._invalidate_ic_weights_cache()

    def test_weight_rises_with_the_size_of_the_evidence(self):
        """Doubling a component's measured IC must raise its weight."""
        ic = {"theta_score": 0.08, "pop_score": 0.04, "rr_score": 0.02,
              "ev_score": 0.01, "liquidity_score": 0.01,
              "momentum_score": 0.01, "iv_rank_score": 0.01}
        p_gate = {"theta_score": 0.02, "pop_score": 0.50, "rr_score": 0.50,
                  "ev_score": 0.50, "liquidity_score": 0.50,
                  "momentum_score": 0.50, "iv_rank_score": 0.50}
        weak = self._blend(ic, p_gate)["theta"]
        ic2 = dict(ic, theta_score=0.16)
        strong = self._blend(ic2, p_gate)["theta"]
        self.assertGreater(
            strong, weak,
            "theta's weight did not move when its measured IC doubled — the "
            "blend is reflecting the survivor count, not the evidence")

    def test_weight_survives_an_unrelated_component_becoming_eligible(self):
        """A second survivor must not strip weight from the first.

        Momentum crossing p=0.10 says nothing about theta's evidence, so it
        must not halve theta's contribution to the score.
        """
        ic = {"theta_score": 0.08, "pop_score": 0.04, "rr_score": 0.02,
              "ev_score": 0.01, "liquidity_score": 0.01,
              "momentum_score": 0.03, "iv_rank_score": 0.01}
        alone = self._blend(ic, {"theta_score": 0.02, "momentum_score": 0.50,
                                 "pop_score": 0.5, "rr_score": 0.5, "ev_score": 0.5,
                                 "liquidity_score": 0.5, "iv_rank_score": 0.5})["theta"]
        joined = self._blend(ic, {"theta_score": 0.02, "momentum_score": 0.09,
                                  "pop_score": 0.5, "rr_score": 0.5, "ev_score": 0.5,
                                  "liquidity_score": 0.5, "iv_rank_score": 0.5})["theta"]
        self.assertAlmostEqual(
            alone, joined, places=9,
            msg="theta lost weight because an unrelated component became "
                "eligible; its own evidence did not change")

    def test_a_lone_weak_survivor_cannot_take_the_whole_budget(self):
        """One survivor with a tiny IC must not be handed the full 0.30 share."""
        ic = {"theta_score": 0.01, "pop_score": 0.05, "rr_score": 0.05,
              "ev_score": 0.05, "liquidity_score": 0.05,
              "momentum_score": 0.05, "iv_rank_score": 0.05}
        w = self._blend(ic, {"theta_score": 0.01, "pop_score": 0.5, "rr_score": 0.5,
                             "ev_score": 0.5, "liquidity_score": 0.5,
                             "momentum_score": 0.5, "iv_rank_score": 0.5})
        # 0.7 * 0.10 = 0.07 is the untouched base share; the old rule added a
        # flat 0.30 on top of it regardless of how small the IC was.
        self.assertLess(w["theta"], 0.20,
                        "a component with IC=0.01 still absorbed the entire "
                        "reallocation budget")

    def test_falls_back_to_config_weights_when_nothing_is_eligible(self):
        w = self._blend({"theta_score": 0.08}, {"theta_score": 0.50})
        self.assertEqual(w, self.BASE)


# --------------------------------------------------------------------------
# 2. A refused EV must not read as a neutral one.
# --------------------------------------------------------------------------
class TestRefusedEVPresentation(unittest.TestCase):
    """`ev_per_contract` is set to NaN on purpose when the realized/implied vol
    gap is implausible or HV is missing — the basis is absent, not zero. The
    decision zone rendered that as "FLAT EV +nan/ct", which reads as "no edge
    either way" while `_verdict_for_row` on the same row returns INDETERMINATE.
    """

    def _zone(self, ev):
        from src.cli_display import format_decision_zone
        row = pd.Series({
            "ev_per_contract": ev, "ev_gross_per_contract": 100.0,
            "ev_cost_per_contract": 20.0, "strategy_name": "Long Call",
            "underlying": 100.0, "strike": 100.0, "premium": 5.0,
            "type": "call", "symbol": "TEST",
        })
        return "\n".join(str(x) for x in format_decision_zone(row))

    def test_refused_ev_does_not_render_as_flat(self):
        text = self._zone(float("nan"))
        self.assertNotIn("nan", text.lower(),
                         "a refused EV printed the literal 'nan'")
        self.assertNotIn("FLAT EV", text,
                         "a refused EV printed as FLAT — indistinguishable "
                         "from a genuinely zero edge")

    def test_refused_ev_says_so(self):
        self.assertIn("UNAVAILABLE", self._zone(float("nan")))

    def test_real_ev_values_still_render(self):
        self.assertIn("POSITIVE EV +40/ct", self._zone(40.0))
        self.assertIn("NEGATIVE EV -40/ct", self._zone(-40.0))
        self.assertIn("FLAT EV +0/ct", self._zone(0.0))


# --------------------------------------------------------------------------
# 3. The score explanation must name the components that carry the score.
# --------------------------------------------------------------------------
class TestScoreExplanationUsesLiveWeights(unittest.TestCase):
    """`explain_quality_score` ranked drivers by a hardcoded table
    (PoP 1.0, EV 1.0, RR 0.8 ...) bearing no relation to the weights actually
    in force. It named components carrying ~0.4% of the score as top drivers
    and had no entry at all for iv_velocity, the third-largest live weight.
    """

    def test_a_zero_weight_component_is_never_named_a_driver(self):
        from src.trade_analysis import explain_quality_score
        row = pd.Series({
            "catalyst_score": 1.0,      # weight 0.0 below — contributes nothing
            "prob_profit": 0.95,        # weight 0.5 — the real driver
            "theta_score": 0.05,
        })
        weights = {"catalyst": 0.0, "pop": 0.5, "theta": 0.5}
        out = explain_quality_score(row, weights=weights)
        self.assertNotIn("Catalyst", out,
                         "a component with weight 0.0 was named a top driver")
        self.assertIn("PoP", out)

    def test_ranks_by_contribution_not_by_raw_value(self):
        from src.trade_analysis import explain_quality_score
        row = pd.Series({"prob_profit": 0.62, "theta_score": 0.99})
        # theta has the higher raw value but a twentieth of the weight, so PoP
        # contributes more to the score and must be listed first.
        out = explain_quality_score(row, weights={"pop": 0.60, "theta": 0.03})
        self.assertLess(out.index("PoP"), out.index("Theta"))

    def test_still_works_with_no_weights_supplied(self):
        from src.trade_analysis import explain_quality_score
        row = pd.Series({"prob_profit": 0.90, "theta_score": 0.10})
        self.assertIsInstance(explain_quality_score(row), str)

    def test_a_spread_is_explained_with_the_weights_that_scored_it(self):
        """`spread_scoring` discards the composite and recomputes
        `quality_score` from `credit_spread_weights`, so a spread explained
        with composite weights describes a scorer that never touched it.
        `credit_to_width` is 20% of the spread weights and has no entry in the
        composite at all.
        """
        from src.trade_analysis import explain_quality_score
        row = pd.Series({
            "_is_spread": True, "type": "PUT SPREAD",
            "credit_to_width_score": 0.95, "pop_score": 0.90,
            "theta_score": 0.10,
        })
        out = explain_quality_score(row)
        self.assertIn("Credit/W", out,
                      "the spread's largest driver after PoP was invisible")

    def test_a_spread_label_is_not_printed_twice(self):
        from src.trade_analysis import explain_quality_score
        row = pd.Series({
            "_is_spread": True, "type": "PUT SPREAD",
            "pop_score": 0.90, "prob_profit": 0.90, "theta_score": 0.10,
        })
        self.assertEqual(explain_quality_score(row).count("PoP"), 1)

    def test_single_leg_rows_are_unaffected(self):
        from src.trade_analysis import explain_quality_score
        row = pd.Series({"prob_profit": 0.90, "theta_score": 0.10})
        self.assertNotIn("Credit/W", explain_quality_score(row))


# --------------------------------------------------------------------------
# 4. The short-interest bonus must follow the direction it was measured in.
# --------------------------------------------------------------------------
class TestShortInterestBonusIsDirectional(unittest.TestCase):
    """The screener added +0.05 to every contract on a name with SI > 20%,
    calls and puts alike. Measured on the 810,266-row squeeze panel
    (si_scale 1.25, 42 trading days): heavy short interest lifts the
    sigma-normalised up-tail by +3.39pp and *lowers* the down-tail by 1.28pp,
    95% CI [-2.07, -0.51]. A long put is paid by the down tail, so the bonus
    was rewarding puts for a tail measurably thinner than the base rate.
    """

    def test_calls_keep_the_bonus_and_puts_do_not(self):
        from src.options_screener import _short_interest_bonus
        df = pd.DataFrame({
            "type": ["call", "put", "call", "put"],
            "short_interest": [0.25, 0.25, 0.05, 0.05],
        })
        bonus = _short_interest_bonus(df, mode="Scan")
        self.assertAlmostEqual(bonus.iloc[0], 0.05)   # heavy SI call
        self.assertAlmostEqual(bonus.iloc[1], 0.0)    # heavy SI put
        self.assertAlmostEqual(bonus.iloc[2], 0.0)    # light SI call
        self.assertAlmostEqual(bonus.iloc[3], 0.0)

    def test_short_premium_modes_get_nothing(self):
        """A premium seller is short the tail the bonus is measuring."""
        from src.options_screener import _short_interest_bonus
        df = pd.DataFrame({"type": ["call", "put"], "short_interest": [0.25, 0.25]})
        bonus = _short_interest_bonus(df, mode="Premium Selling")
        self.assertTrue((bonus == 0.0).all())

    def test_missing_column_is_no_bonus_not_a_crash(self):
        from src.options_screener import _short_interest_bonus
        df = pd.DataFrame({"type": ["call", "put"]})
        self.assertTrue((_short_interest_bonus(df, mode="Scan") == 0.0).all())


# --------------------------------------------------------------------------
# 5. The squeeze board's "BE vol +N" must be the spread cost it claims to be.
# --------------------------------------------------------------------------
class TestBreakevenVolReference(unittest.TestCase):
    """The board prints `BE vol` and, beside it, the gap over the contract's IV,
    captioned "what crossing the spread costs in vol points". It subtracted the
    *vendor's* reported IV, which is solved from a different price than the mid
    the breakeven is built on. Measured over 29,769 archived CBOE call
    snapshots: 16.7% of contracts printed a NEGATIVE cost, and the mean absolute
    error was 0.92vp against a median true cost of 1.40vp.
    """

    ROW = {"underlying": 100.0, "strike": 100.0, "premium": 8.0,
           "dte": 90.0, "spread_pct": 0.06, "impliedVolatility": 0.55}

    def test_cost_in_vol_points_is_never_negative(self):
        from src.squeeze.board import breakeven_vol_premium_ref
        # Vendor IV sits well above the mid-implied IV — the case that used to
        # print the spread as a discount.
        row = dict(self.ROW, impliedVolatility=0.90)
        be, ref = breakeven_vol_premium_ref(row)
        self.assertIsNotNone(be)
        self.assertIsNotNone(ref)
        self.assertGreaterEqual(be - ref, 0.0,
                                "crossing the spread priced as a discount")

    def test_the_gap_is_the_spread_and_nothing_else(self):
        from src.squeeze.board import breakeven_vol_premium_ref
        zero_spread = dict(self.ROW, spread_pct=0.0)
        be, ref = breakeven_vol_premium_ref(zero_spread)
        self.assertAlmostEqual(be, ref, places=6,
                               msg="a zero-spread contract still showed a cost")

    def test_a_wider_spread_costs_more_vol_points(self):
        from src.squeeze.board import breakeven_vol_premium_ref
        narrow = breakeven_vol_premium_ref(dict(self.ROW, spread_pct=0.02))
        wide = breakeven_vol_premium_ref(dict(self.ROW, spread_pct=0.20))
        self.assertGreater(wide[0] - wide[1], narrow[0] - narrow[1])

    def test_unusable_row_returns_none_not_zero(self):
        from src.squeeze.board import breakeven_vol_premium_ref
        self.assertEqual(breakeven_vol_premium_ref({"premium": 0.0}), (None, None))


# --------------------------------------------------------------------------
# 6. The cross-section EV tiebreaker was reading a column that never exists.
# --------------------------------------------------------------------------
class TestCrossSectionNormalize(unittest.TestCase):
    """`_cross_section_normalize` documented an EV tiebreaker reading `df["ev"]`.
    No code path in the repo ever creates a bare `ev` column — the scan carries
    `ev_per_contract`, `ev_score`, `ev_gross_per_contract`. The branch was dead
    from the day it shipped, so the docstring described behaviour that never ran.
    """

    def test_display_score_is_a_pure_function_of_the_raw_score(self):
        from src.options_screener import _cross_section_normalize
        raw = [0.30, 0.45, 0.60, 0.75]
        a = _cross_section_normalize(
            pd.DataFrame({"quality_score": raw,
                          "ev_per_contract": [500.0, -500.0, 500.0, -500.0]}))
        b = _cross_section_normalize(
            pd.DataFrame({"quality_score": raw,
                          "ev_per_contract": [-500.0, 500.0, -500.0, 500.0]}))
        pd.testing.assert_series_equal(a["quality_score"], b["quality_score"])

    def test_the_documented_reference_maps_as_stated(self):
        from src.options_screener import _cross_section_normalize
        out = _cross_section_normalize(pd.DataFrame({"quality_score": [0.28, 0.82]}))
        self.assertAlmostEqual(out["quality_score"].iloc[0], 0.0, places=3)
        self.assertAlmostEqual(out["quality_score"].iloc[1], 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
