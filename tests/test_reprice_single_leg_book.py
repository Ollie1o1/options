"""tests/test_reprice_single_leg_book.py

Pure computation only — no db file, no network. Mirrors the fixture style of
tests/test_spread_surface_report.py and tests/test_track_record_equal_weighted.py.
"""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.spread_surface import Cell, SpreadSurface  # noqa: E402

from scripts.reprice_single_leg_book import (  # noqa: E402
    cluster_bootstrap_pf,
    gross_pct,
    is_expired,
    new_friction_fraction,
)


class GrossPctTests(unittest.TestCase):
    def test_short_leg_profits_when_price_falls(self):
        # Sold at 2.00, bought back at 0.50: short profits as price falls.
        self.assertAlmostEqual(gross_pct(2.00, 0.50, short=True), 0.75)

    def test_long_leg_profits_when_price_rises(self):
        self.assertAlmostEqual(gross_pct(1.00, 2.50, short=False), 1.5)

    def test_matches_paper_manager_formula_sign(self):
        # paper_manager._evaluate_short_single_leg_exit:
        #   pnl_raw = (entry_price - current_price) / entry_price
        entry, exit_ = 3.20, 4.10
        self.assertAlmostEqual(
            gross_pct(entry, exit_, short=True),
            (entry - exit_) / entry,
        )


class IsExpiredTests(unittest.TestCase):
    def test_expired_reason_detected(self):
        self.assertTrue(is_expired("Expired (settled at intrinsic)"))

    def test_other_reasons_not_expired(self):
        self.assertFalse(is_expired("Take Profit (50% @ 12d)"))
        self.assertFalse(is_expired(None))
        self.assertFalse(is_expired(""))


class NewFrictionFractionTests(unittest.TestCase):
    def setUp(self):
        # Three OI buckets at the SAME (delta, dte) cell — bucket 0 is the
        # illiquid pin, higher buckets progressively cheaper. This mirrors
        # the real surface (a populated bucket 0 always exists), so
        # relative(..., open_interest=None) hits bucket 0's OWN exact cell
        # directly instead of falling through the collapse ladder — and
        # oi_collapsed_relative's median-across-buckets then genuinely
        # differs from it, which is the property being tested.
        self.surface = SpreadSurface(
            {
                (1, 1, 0): Cell(n=40, rel_half_spread=0.05, median_depth=5),
                (1, 1, 2): Cell(n=50, rel_half_spread=0.02, median_depth=20),
                (1, 1, 3): Cell(n=30, rel_half_spread=0.01, median_depth=50),
            },
            {"fit_date": "2026-09-01"},
        )

    def test_known_oi_hits_exact_cell(self):
        frac, prov = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=500.0, round_trip=True,
        )
        self.assertEqual(prov, "cell")
        # half=0.02*2.00=0.04; round trip both sides, no commission: 2*0.04=0.08
        # fraction of mid: 0.08/2.00 = 0.04
        self.assertAlmostEqual(frac, 0.04)

    def test_hold_to_expiry_charges_opening_side_only(self):
        frac, _ = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=500.0, round_trip=False,
        )
        self.assertAlmostEqual(frac, 0.02)  # one side: 0.04/2.00 = 0.02

    def test_unknown_oi_conservative_pins_bucket_zero(self):
        # open_interest=None resolves cell_key's oi dimension to bucket 0,
        # which is populated here — an exact "cell" hit at the illiquid pin,
        # not a fallback rung.
        frac, prov = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=None, round_trip=True, central=False,
        )
        self.assertEqual(prov, "cell")
        # half=0.05*2.00=0.10; round trip: 2*0.10=0.20; fraction 0.20/2.00=0.10
        self.assertAlmostEqual(frac, 0.10)

    def test_unknown_oi_central_uses_oi_collapsed_marginal(self):
        frac, prov = new_friction_fraction(
            self.surface, mid=2.00, abs_delta=0.15, dte=10.0,
            open_interest=None, round_trip=True, central=True,
        )
        self.assertEqual(prov, "oi_collapsed")
        # median of [0.05, 0.02, 0.01] = 0.02; half=0.04; round trip 0.08;
        # fraction 0.08/2.00 = 0.04 — strictly less than the conservative
        # bucket-0 pin (0.10) computed above.
        self.assertAlmostEqual(frac, 0.04)


class ClusterBootstrapPfTests(unittest.TestCase):
    def test_point_estimate_matches_plain_profit_factor(self):
        rows = [
            {"date": "2026-08-01", "r": 0.10},
            {"date": "2026-08-01", "r": -0.05},
            {"date": "2026-08-02", "r": 0.20},
            {"date": "2026-08-03", "r": -0.10},
        ]
        point, lo, hi = cluster_bootstrap_pf(rows, "r", n_boot=200, seed=1)
        # wins 0.30, losses 0.15 -> PF 2.0
        self.assertAlmostEqual(point, 2.0)
        self.assertIsNotNone(lo)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)

    def test_clustering_widens_the_interval_vs_row_level(self):
        # 10 entry days, 4 rows each. Every row's sign is fully determined by
        # its day (5 "good" days all positive, 5 "bad" days all negative) —
        # the day is the true independent unit; the 4 rows inside it carry
        # almost no extra information beyond magnitude noise. A row-level
        # bootstrap that resamples all 40 rows as if independent treats one
        # day's outcome as 4 pieces of evidence, understating uncertainty —
        # the same overcounting trap that inflated the ranker test and the
        # catalyst bootstrap. The correct cluster bootstrap must be wider.
        rows = []
        good_vals = [0.28, 0.30, 0.32, 0.34]
        bad_vals = [-0.23, -0.25, -0.27, -0.29]
        for d in (0, 2, 4, 6, 8):
            for v in good_vals:
                rows.append({"date": f"day{d}", "r": v})
        for d in (1, 3, 5, 7, 9):
            for v in bad_vals:
                rows.append({"date": f"day{d}", "r": v})
        _, lo_cluster, hi_cluster = cluster_bootstrap_pf(
            rows, "r", n_boot=4000, seed=1)

        def _row_level(rows, n_boot, seed):
            import random
            from scripts.publish_track_record import profit_factor
            rnd = random.Random(seed)
            vals = [r["r"] for r in rows]
            draws = [profit_factor([rnd.choice(vals) for _ in vals])
                     for _ in range(n_boot)]
            draws = sorted(d for d in draws if d is not None)
            return draws[int(0.025 * len(draws))], draws[int(0.975 * len(draws)) - 1]

        lo_row, hi_row = _row_level(rows, 4000, 1)
        self.assertGreater((hi_cluster - lo_cluster), (hi_row - lo_row))

    def test_single_cluster_returns_no_interval(self):
        rows = [{"date": "2026-08-01", "r": 0.1}, {"date": "2026-08-01", "r": -0.1}]
        point, lo, hi = cluster_bootstrap_pf(rows, "r", n_boot=100, seed=1)
        self.assertIsNone(lo)
        self.assertIsNone(hi)


if __name__ == "__main__":
    unittest.main()
