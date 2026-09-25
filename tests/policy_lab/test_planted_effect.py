"""End-to-end sanity: plant a known optimal take-profit and require the lab to
recover it, then remove the effect and require the lab to find nothing.

The second half is the one that matters. Every prior research effort in this
repo returned null; a harness that cannot return null is not measuring.

CORRECTION vs. the task brief's `_make_path`: the brief passed a varying
`entry_date` (2026-08-18 through 2026-08-22) while every path's points always
start at 2026-08-18. When `entry_date` lands after `points[0].date`,
`PricePath.interior_points` (points strictly between `entry_date` and
`actual_exit_date`) silently returns fewer points or none, so every policy
falls through to `hold_to_end` and the planted effect vanishes regardless of
what the test claims to measure. `_make_path` here always sets
`entry_date = points[0].date` and gets cluster variety from `symbol` alone
(30 distinct symbols, comfortably above `MIN_CLUSTERS = 20`).
"""
import unittest

import numpy as np

from src.policy_lab.costs import CostModel
from src.policy_lab.policies import ExitPolicy
from src.policy_lab.replay import replay
from src.policy_lab.stats import (
    cluster_bootstrap_mean_ci, family_wise_p, paired_frame,
)
from src.policy_lab.types import PathPoint, PricePath


def _make_path(pid, symbol, mids, car=400.0):
    points = []
    for i, m in enumerate(mids):
        d = f"2026-08-{18 + i:02d}"
        points.append(PathPoint(d, m - 0.05, m + 0.05, m, 100.0, 40 - i))
    return PricePath(pid, symbol, "Bull Put", points[0].date, 1.00, car, True,
                     tuple(points), points[-1].date, 0.0, "A")


class TestPlantedEffect(unittest.TestCase):
    def test_recovers_a_planted_take_profit_advantage(self):
        """Paths that dip to 0.50 then rebound to 0.95 reward a 50% TP."""
        rng = np.random.default_rng(0)
        paths = []
        for i in range(120):
            sym = f"S{i % 30}"
            mids = [1.00, 0.50 + rng.normal(0, 0.01), 0.95, 0.95]
            paths.append(_make_path(f"p{i}", sym, mids))

        tp = ExitPolicy("tp50", 0.50, None, None, None)
        null = ExitPolicy("null", None, None, None, None)
        costs = CostModel.mid()
        base = {p.position_id: replay(p, null, costs) for p in paths}
        alt = {p.position_id: replay(p, tp, costs) for p in paths}

        df = paired_frame(paths, base, alt)
        lo, hi = cluster_bootstrap_mean_ci(df, "d", n_boot=500, seed=1)
        self.assertGreater(lo, 0.0, "planted advantage should be detected")
        self.assertGreater(df["d"].mean(), 0.0)

    def test_finds_nothing_when_nothing_is_planted(self):
        """A true martingale must not produce a CI excluding zero.

        Prices are UNFLOORED here, so the walk is a real martingale and the
        optional stopping theorem gives E[d] = 0 for any bounded stopping rule.
        Negative prices are unphysical on purpose: they isolate the statistics
        from the geometry of a lower barrier, which is tested separately below.
        The noise is set so take-profits fire on roughly 18 of 120 paths —
        the previous version fired on 2, which tested almost nothing.
        """
        rng = np.random.default_rng(7)
        paths = []
        for i in range(120):
            walk = [1.00]
            for _ in range(3):
                walk.append(walk[-1] + rng.normal(0, 0.35))   # NO floor
            paths.append(_make_path(f"p{i}", f"S{i % 30}", walk))

        tp = ExitPolicy("tp50", 0.50, None, None, None)
        null = ExitPolicy("null", None, None, None, None)
        costs = CostModel.cross()
        base = {p.position_id: replay(p, null, costs) for p in paths}
        alt = {p.position_id: replay(p, tp, costs) for p in paths}

        df = paired_frame(paths, base, alt)
        self.assertGreater((df["d"] != 0).sum(), 5,
                           "take-profit must actually fire, or this tests nothing")
        lo, _ = cluster_bootstrap_mean_ci(df, "d", n_boot=600, seed=1)
        self.assertLessEqual(lo, 0.0,
                             "a CI excluding zero on a martingale means the lab lies")

    def test_a_lower_barrier_gives_early_exit_a_real_advantage(self):
        """Banking a dip beats holding when prices cannot go below zero.

        This is not a defect and not an edge. Near a lower barrier the walk can
        only go up, so an early exit genuinely outperforms holding — and real
        option prices are bounded below by zero, so the lab WILL see this on
        real data. It would appear identically in any short-premium book on any
        underlyings, which is why a raw take-profit advantage is not evidence of
        anything specific to this book. Measured: 11/20 runs produce a CI
        excluding zero at this noise level with a floor, 0/20 without one.
        """
        rng = np.random.default_rng(7)
        paths = []
        for i in range(120):
            walk = [1.00]
            for _ in range(3):
                walk.append(max(0.01, walk[-1] + rng.normal(0, 0.45)))   # FLOORED
            paths.append(_make_path(f"p{i}", f"S{i % 30}", walk))

        tp = ExitPolicy("tp50", 0.50, None, None, None)
        null = ExitPolicy("null", None, None, None, None)
        costs = CostModel.cross()
        base = {p.position_id: replay(p, null, costs) for p in paths}
        alt = {p.position_id: replay(p, tp, costs) for p in paths}

        df = paired_frame(paths, base, alt)
        self.assertGreater(df["d"].mean(), 0.0)

    def test_crossing_costs_never_improve_a_result(self):
        paths = [_make_path(f"p{i}", f"S{i%10}",
                            [1.00, 0.60, 0.40, 0.30]) for i in range(40)]
        pol = ExitPolicy("tp50", 0.50, None, None, None)
        at_mid = np.mean([replay(p, pol, CostModel.mid()).pnl_frac_car
                          for p in paths])
        at_cross = np.mean([replay(p, pol, CostModel.cross()).pnl_frac_car
                            for p in paths])
        self.assertLessEqual(at_cross, at_mid)

    def test_family_wise_p_finds_nothing_in_a_null_grid(self):
        """A grid of pure noise must not produce a family-wise rejection.

        This is the property that matters most: this repo's research history is
        ~25 null results, so a correction that fires on noise would turn the lab
        from a filter into a generator of false leads.
        """
        rng = np.random.default_rng(0)
        grid = rng.normal(0.0, 0.046, size=(750, 329))
        self.assertGreater(family_wise_p(grid, n_perm=500, seed=1), 0.05)


if __name__ == "__main__":
    unittest.main()
