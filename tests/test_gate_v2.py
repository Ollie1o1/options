"""Gate v2: the signed redesign (docs/GATE_REDESIGN_SPEC.md, 2026-07-31).

The spec's core promise is that every state has an entry AND an exit — v1's
EXTEND had none, so an IC drifting between 0.03 and 0.08 extended forever.
These tests pin each band boundary and, above all, that no state is unbounded.
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_checkpoint import (  # noqa: E402
    GATE_V2_MAX_EXTENSIONS,
    decide_v1,
    decide_v2,
    design_effect,
    posterior_ic_above,
)


def _ic_for_posterior(target_post: float, n: int) -> float:
    """The rank IC whose posterior P(true IC >= 0.08) equals `target`, so the
    band tests probe the boundary itself rather than a value near it."""
    lo, hi = -0.95, 0.95
    for _ in range(200):
        mid = (lo + hi) / 2
        if posterior_ic_above(mid, n, 0.08) < target_post:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


class BandBoundaryTest(unittest.TestCase):
    N = 200

    def test_just_above_the_ready_bar_is_ready(self):
        ic = _ic_for_posterior(0.851, self.N)
        decision, _ = decide_v2(self.N, ic_rank=ic, ic_pearson=ic)
        self.assertEqual(decision, "READY")

    def test_just_below_the_ready_bar_is_not_ready(self):
        ic = _ic_for_posterior(0.849, self.N)
        decision, _ = decide_v2(self.N, ic_rank=ic, ic_pearson=ic)
        self.assertNotEqual(decision, "READY")

    def test_at_or_below_the_stop_floor_is_stop(self):
        ic = _ic_for_posterior(0.149, self.N)
        decision, why = decide_v2(self.N, ic_rank=ic, ic_pearson=ic)
        self.assertEqual(decision, "STOP")
        self.assertIn("%", why)

    def test_between_the_bands_extends(self):
        ic = _ic_for_posterior(0.50, self.N)
        decision, _ = decide_v2(self.N, ic_rank=ic, ic_pearson=ic)
        self.assertEqual(decision, "EXTEND")

    def test_a_rank_ic_below_the_floor_stops_even_mid_band(self):
        decision, why = decide_v2(self.N, ic_rank=0.02, ic_pearson=0.02)
        self.assertEqual(decision, "STOP")
        self.assertIn("floor", why)


class TerminalConditionTest(unittest.TestCase):
    """The bug v2 exists to fix: EXTEND must not be able to run forever."""

    def test_extend_is_granted_at_most_twice_then_stops(self):
        n = 200
        ic = _ic_for_posterior(0.50, n)  # squarely mid-band: v1 would loop here
        seen = []
        for used in range(GATE_V2_MAX_EXTENSIONS + 1):
            decision, _ = decide_v2(n, ic_rank=ic, ic_pearson=ic, extensions_used=used)
            seen.append(decision)
        self.assertEqual(seen[:GATE_V2_MAX_EXTENSIONS],
                         ["EXTEND"] * GATE_V2_MAX_EXTENSIONS)
        self.assertEqual(seen[GATE_V2_MAX_EXTENSIONS], "STOP")

    def test_the_same_input_extends_forever_under_v1(self):
        # Demonstrates the defect rather than asserting it away: a Pearson IC of
        # 0.05 with a hopeless p-value is EXTEND under v1 at any week count.
        for weeks in (6, 60, 600):
            self.assertEqual(decide_v1(n=200, ic_p=0.05, p_p=0.9, weeks=weeks),
                             "EXTEND")

    def test_every_state_is_reachable_and_leaves(self):
        n = 200
        states = {
            decide_v2(10, 0.5, 0.5)[0],                                    # GATHERING
            decide_v2(n, _ic_for_posterior(0.9, n), _ic_for_posterior(0.9, n))[0],
            decide_v2(n, _ic_for_posterior(0.5, n), _ic_for_posterior(0.5, n))[0],
            decide_v2(n, 0.0, 0.0)[0],                                     # STOP
        }
        self.assertEqual(states, {"GATHERING", "READY", "EXTEND", "STOP"})


class SignGuardTest(unittest.TestCase):
    def test_ready_is_withheld_when_the_statistics_disagree_in_sign(self):
        # Never authorise real money on a statistic its counterpart contradicts.
        ic = _ic_for_posterior(0.95, 200)
        decision, why = decide_v2(200, ic_rank=ic, ic_pearson=-0.05)
        self.assertEqual(decision, "EXTEND")
        self.assertIn("disagree", why)

    def test_agreement_lets_ready_through(self):
        ic = _ic_for_posterior(0.95, 200)
        decision, _ = decide_v2(200, ic_rank=ic, ic_pearson=0.10)
        self.assertEqual(decision, "READY")


class EffectiveNTest(unittest.TestCase):
    def test_clustered_entries_reduce_n(self):
        # Two entry days, tight within-day agreement: n_eff must fall below n.
        returns = np.array([1.0, 1.02, 0.98, -1.0, -1.02, -0.98])
        dates = ["2026-06-01"] * 3 + ["2026-06-02"] * 3
        icc, de, n_eff = design_effect(returns, dates)
        self.assertGreater(icc, 0.0)
        self.assertGreater(de, 1.0)
        self.assertLess(n_eff, len(returns))

    def test_one_trade_per_day_leaves_n_untouched(self):
        returns = np.array([0.4, -0.2, 0.1, -0.5, 0.3])
        dates = [f"2026-06-0{i}" for i in range(1, 6)]
        icc, de, n_eff = design_effect(returns, dates)
        self.assertEqual(de, 1.0)
        self.assertEqual(n_eff, float(len(returns)))

    def test_icc_is_floored_at_zero(self):
        # Anti-correlated within days would give a negative ICC; that means
        # "no positive clustering", never "better than independent".
        returns = np.array([1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
        dates = ["2026-06-01"] * 3 + ["2026-06-02"] * 3
        icc, de, n_eff = design_effect(returns, dates)
        self.assertGreaterEqual(icc, 0.0)
        self.assertGreaterEqual(de, 1.0)
        self.assertLessEqual(n_eff, len(returns))

    def test_effective_n_gates_the_trigger(self):
        # 60 nominal trades that cluster down below 50 effective must not fire.
        decision, why = decide_v2(49.9, ic_rank=0.5, ic_pearson=0.5)
        self.assertEqual(decision, "GATHERING")
        self.assertIn("effective n", why)


class V1PreservationTest(unittest.TestCase):
    """v1 must keep answering exactly as it always did, so the superseded rule
    stays auditable rather than being quietly rewritten."""

    def test_v1_thresholds_are_unchanged(self):
        self.assertEqual(decide_v1(49, 0.5, 0.001, 10), "GATHERING")
        self.assertEqual(decide_v1(50, 0.08, 0.049, 10), "READY")
        self.assertEqual(decide_v1(50, 0.05, 0.9, 10), "EXTEND")
        self.assertEqual(decide_v1(50, 0.02, 0.9, 6), "STOP")
        self.assertEqual(decide_v1(50, 0.02, 0.9, 5), "GATHERING")


if __name__ == "__main__":
    unittest.main()
