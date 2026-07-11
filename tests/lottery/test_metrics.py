"""Tests for the honest lottery metrics engine."""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.lottery import metrics as M
from src.utils import bs_price, norm_cdf


class TestStrikeSigma(unittest.TestCase):
    def test_atm_is_zero(self):
        self.assertAlmostEqual(M.strike_sigma(100, 100, 0.5, 0.1), 0.0, places=6)

    def test_otm_positive_and_scales(self):
        s1 = M.strike_sigma(100, 110, 0.5, 0.1)
        s2 = M.strike_sigma(100, 120, 0.5, 0.1)
        self.assertGreater(s1, 0)
        self.assertGreater(s2, s1)

    def test_invalid_returns_none(self):
        self.assertIsNone(M.strike_sigma(0, 100, 0.5, 0.1))
        self.assertIsNone(M.strike_sigma(100, 100, 0, 0.1))
        self.assertIsNone(M.strike_sigma(100, 100, 0.5, 0))


class TestBreakeven(unittest.TestCase):
    def test_call_breakeven(self):
        # spot 100, strike 110, premium 2 -> BE price 112 -> 12% move
        self.assertAlmostEqual(M.breakeven_move_pct(100, 110, "call", 2.0), 0.12, places=6)

    def test_put_breakeven(self):
        # spot 100, strike 90, premium 2 -> BE price 88 -> 12% down move
        self.assertAlmostEqual(M.breakeven_move_pct(100, 90, "put", 2.0), 0.12, places=6)


class TestHitProbability(unittest.TestCase):
    def test_matches_closed_form(self):
        # call, spot 100, strike 110, premium 1, hit 3x -> barrier = 110 + 3 = 113
        spot, K, prem, T, sig, hm = 100.0, 110.0, 1.0, 0.25, 0.6, 3.0
        p = M.hit_probability(spot, K, "call", prem, T, sig, hit_multiple=hm)
        barrier = K + hm * prem
        unit = sig * math.sqrt(T)
        d = (math.log(spot / barrier) + (-0.5 * sig * sig) * T) / unit
        self.assertAlmostEqual(p, float(norm_cdf(d)), places=9)

    def test_probability_in_unit_interval(self):
        p = M.hit_probability(100, 110, "call", 1.0, 0.25, 0.6)
        self.assertTrue(0.0 <= p <= 1.0)

    def test_further_otm_lower_prob(self):
        near = M.hit_probability(100, 105, "call", 1.0, 0.25, 0.6)
        far = M.hit_probability(100, 130, "call", 1.0, 0.25, 0.6)
        self.assertGreater(near, far)

    def test_put_unreachable_barrier_is_zero(self):
        # strike 5, premium 3, hit 3x -> barrier = 5 - 9 = -4 < 0 -> impossible
        self.assertEqual(M.hit_probability(100, 5, "put", 3.0, 0.25, 0.6), 0.0)

    def test_invalid_inputs_return_none(self):
        self.assertIsNone(M.hit_probability(0, 110, "call", 1.0, 0.25, 0.6))
        self.assertIsNone(M.hit_probability(100, 110, "call", 0, 0.25, 0.6))
        self.assertIsNone(M.hit_probability(100, 110, "call", 1.0, 0, 0.6))


class TestTailMultiple(unittest.TestCase):
    def test_reprices_above_premium_on_favorable_move(self):
        # A 2-EM favorable move should be worth a multiple of the tiny debit.
        mult = M.tail_multiple(100, 110, "call", 1.0, 0.25, 0.6, em_dollars=8.0, n_em=2.0)
        self.assertIsNotNone(mult)
        self.assertGreater(mult, 1.0)

    def test_two_em_beats_one_em(self):
        one = M.tail_multiple(100, 110, "call", 1.0, 0.25, 0.6, 8.0, 1.0)
        two = M.tail_multiple(100, 110, "call", 1.0, 0.25, 0.6, 8.0, 2.0)
        self.assertGreater(two, one)

    def test_matches_bs_reprice(self):
        spot, K, prem, T, sig, em = 100.0, 110.0, 1.0, 0.25, 0.6, 8.0
        mult = M.tail_multiple(spot, K, "call", prem, T, sig, em, 1.0, time_elapsed_frac=0.5)
        expected = bs_price("call", spot + em, K, T * 0.5, 0.0, sig, 0.0) / prem
        self.assertAlmostEqual(mult, expected, places=6)


class TestIvStateAndCrushTrap(unittest.TestCase):
    def test_iv_state(self):
        self.assertEqual(M.iv_state(0.20), "cheap")
        self.assertEqual(M.iv_state(0.50), "fair")
        self.assertEqual(M.iv_state(0.85), "rich")
        self.assertEqual(M.iv_state(None), "fair")

    def test_crush_trap_trips_on_rich_iv_into_event(self):
        self.assertTrue(M.crush_trap(0.88, 4))
        self.assertIn("4d", M.crush_trap(0.88, 4))

    def test_crush_trap_quiet_when_cheap_or_no_event(self):
        self.assertEqual(M.crush_trap(0.20, 4), "")   # cheap IV
        self.assertEqual(M.crush_trap(0.88, 200), "")  # event beyond window
        self.assertEqual(M.crush_trap(None, 4), "")


class TestEdgeFlag(unittest.TestCase):
    def _base(self, **kw):
        args = dict(
            spot=100, strike=108, opt_type="call", iv=0.5, t_years=0.1,
            iv_rank=0.30, realized_vol=None, has_catalyst=True,
            momentum_aligned=False, spread_pct=0.2,
        )
        args.update(kw)
        return M.edge_flag(**args)

    def test_passes_when_cheap_reachable_catalyst_liquid(self):
        self.assertTrue(self._base())

    def test_fails_when_iv_rich_and_no_realized_edge(self):
        self.assertFalse(self._base(iv_rank=0.90))

    def test_realized_gt_implied_counts_as_cheap(self):
        self.assertTrue(self._base(iv_rank=0.90, realized_vol=0.7))

    def test_fails_when_too_deep_otm(self):
        self.assertFalse(self._base(strike=200))  # far beyond max_strike_sigma

    def test_fails_without_catalyst_or_momentum(self):
        self.assertFalse(self._base(has_catalyst=False, momentum_aligned=False))

    def test_momentum_alignment_substitutes_for_catalyst(self):
        self.assertTrue(self._base(has_catalyst=False, momentum_aligned=True))

    def test_fails_when_illiquid(self):
        self.assertFalse(self._base(spread_pct=0.95))


class TestContractRead(unittest.TestCase):
    def test_full_bundle_on_good_row(self):
        row = {
            "underlying": 100.0, "strike": 108.0, "type": "call", "premium": 1.2,
            "T_years": 0.1, "iv": 0.5, "iv_rank_score": 0.30, "expected_move": 8.0,
            "earnings_dte": 6, "spread_pct": 0.2, "momentum": 0.03,
        }
        read = M.contract_read(row)
        self.assertEqual(read["iv_state"], "cheap")
        self.assertTrue(read["edge"])
        self.assertEqual(read["crush_trap"], "")
        self.assertTrue(0.0 <= read["hit_prob"] <= 1.0)
        self.assertGreater(read["tail_x_2em"], read["tail_x_1em"])
        self.assertIsNotNone(read["breakeven_vs_em"])

    def test_crush_trap_row(self):
        row = {
            "underlying": 100.0, "strike": 108.0, "type": "call", "premium": 1.2,
            "T_years": 0.1, "iv": 0.9, "iv_rank_score": 0.88, "expected_move": 8.0,
            "earnings_dte": 3,
        }
        read = M.contract_read(row)
        self.assertTrue(read["crush_trap"])
        self.assertEqual(read["iv_state"], "rich")
        self.assertFalse(read["edge"])

    def test_safe_on_sparse_row(self):
        read = M.contract_read({"strike": 100})  # almost everything missing
        self.assertIsNone(read["hit_prob"])
        self.assertFalse(read["edge"])
        self.assertEqual(read["crush_trap"], "")


if __name__ == "__main__":
    unittest.main()
