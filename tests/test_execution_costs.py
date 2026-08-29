"""Execution cost model: what a round trip actually costs, per structure.

Every "clears / fails the cost wall" verdict in this repo rests on one number,
and that number was a single flat $0.05 per share applied to every leg of every
structure. Measured against the archived quotes of the exact contracts the
ledger logged, the flat figure is 3.2x too low for Bull Put and 2x too high for
Bear Call — which is precisely the comparison prefer-bull-put-at-small-size
turns on.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.execution_costs import (
    CostModel,
    fx_cost,
    half_spread_for,
    is_measured,
    measure_half_spreads,
    reprice_pnl_pct,
    round_trip_friction,
)
from src.spread_surface import Cell, SpreadSurface, cell_key


class TestHalfSpreadLookup(unittest.TestCase):
    def setUp(self):
        self.table = {
            "Bull Put": {"n": 28, "median_half_spread": 0.162},
            "Bear Call": {"n": 39, "median_half_spread": 0.025},
            "Thin Thing": {"n": 3, "median_half_spread": 0.900},
        }

    def test_uses_the_measured_value_for_a_known_structure(self):
        self.assertAlmostEqual(half_spread_for("Bull Put", self.table, default=0.05), 0.162)

    def test_falls_back_for_an_unmeasured_structure(self):
        self.assertAlmostEqual(half_spread_for("Calendar", self.table, default=0.05), 0.05)

    def test_ignores_buckets_with_too_little_data(self):
        # 3 observations cannot set a cost constant; a thin bucket must not
        # quietly become a 0.90 haircut.
        self.assertAlmostEqual(half_spread_for("Thin Thing", self.table, default=0.05), 0.05)

    def test_matching_is_case_and_whitespace_tolerant(self):
        self.assertAlmostEqual(half_spread_for(" bull put ", self.table, default=0.05), 0.162)

    def test_is_measured_true_for_a_structure_clearing_the_floor(self):
        self.assertTrue(is_measured("Bull Put", self.table))

    def test_is_measured_false_below_min_observations(self):
        # Thin Thing has a table entry, but half_spread_for never uses its
        # value — n=3 is below MIN_OBSERVATIONS, so it falls back to the
        # caller's default. is_measured must say so, not just "found".
        self.assertFalse(is_measured("Thin Thing", self.table))

    def test_is_measured_false_for_a_structure_absent_from_the_table(self):
        self.assertFalse(is_measured("Calendar", self.table))


class TestRoundTripFriction(unittest.TestCase):
    def test_two_leg_credit_spread_pays_both_sides_of_both_legs(self):
        # 2 legs x 2 sides x $0.05 = $0.20/share = $20, plus 4 commissions.
        f = round_trip_friction(n_legs=2, half_spread=0.05, commission_per_contract=0.65)
        self.assertAlmostEqual(f, 0.20 + (4 * 0.65 / 100.0))

    def test_iron_condor_pays_for_four_legs(self):
        f = round_trip_friction(n_legs=4, half_spread=0.05, commission_per_contract=0.65)
        self.assertAlmostEqual(f, 0.40 + (8 * 0.65 / 100.0))

    def test_a_wider_measured_spread_costs_proportionally_more(self):
        flat = round_trip_friction(n_legs=2, half_spread=0.05, commission_per_contract=0.65)
        real = round_trip_friction(n_legs=2, half_spread=0.162, commission_per_contract=0.65)
        self.assertGreater(real, flat)

    def test_expiring_worthless_pays_only_the_opening_side(self):
        # The hold-to-expiry case: no closing trade, so no closing spread and no
        # closing commission.
        opened = round_trip_friction(n_legs=2, half_spread=0.05,
                                     commission_per_contract=0.65, round_trip=False)
        both = round_trip_friction(n_legs=2, half_spread=0.05, commission_per_contract=0.65)
        self.assertAlmostEqual(opened, both / 2)


class TestCostModelOnATrade(unittest.TestCase):
    def test_friction_as_a_fraction_of_the_credit_taken(self):
        # $0.50 credit, 2 legs, flat model: $0.226 friction on $0.50 = 45% of credit.
        model = CostModel(table={}, default_half_spread=0.05, commission_per_contract=0.65)
        self.assertAlmostEqual(
            model.friction_fraction("Bull Put", entry_credit=0.50, n_legs=2),
            0.226 / 0.50, places=6,
        )

    def test_measured_spread_changes_the_verdict_for_bull_put(self):
        # The same trade under the measured $0.162 half-spread costs far more.
        model = CostModel(
            table={"Bull Put": {"n": 28, "median_half_spread": 0.162}},
            default_half_spread=0.05, commission_per_contract=0.65,
        )
        self.assertGreater(model.friction_fraction("Bull Put", 0.50, 2), 1.0)

    def test_zero_credit_does_not_divide_by_zero(self):
        model = CostModel(table={}, default_half_spread=0.05, commission_per_contract=0.65)
        self.assertEqual(model.friction_fraction("Bull Put", 0.0, 2), 0.0)


class TestCurrencyConversion(unittest.TestCase):
    """A CAD account trading US-listed options converts on the way in and again
    on the way out. Wealthsimple's spread is 1.5% under $10k of conversions,
    which is a real cost the model never carried — and unlike commission it
    scales with the money moved, not the contract count."""

    def test_premium_pays_the_spread_in_both_directions(self):
        # $1.00/share premium, 100 shares: $1.50 in, $1.50 out.
        self.assertAlmostEqual(fx_cost(cash_usd=100.0, rate=0.015), 3.0)

    def test_one_way_conversion_for_a_position_left_to_expire(self):
        self.assertAlmostEqual(
            fx_cost(cash_usd=100.0, rate=0.015, round_trip=False), 1.5
        )

    def test_a_usd_account_removes_the_cost(self):
        # $10/month buys a USD account and a 0% conversion rate.
        self.assertEqual(fx_cost(cash_usd=100.0, rate=0.0), 0.0)

    def test_cost_scales_with_size_not_with_legs(self):
        self.assertAlmostEqual(
            fx_cost(1000.0, 0.015), 10 * fx_cost(100.0, 0.015)
        )


class TestWealthsimpleFeeShape(unittest.TestCase):
    """With $0 commission and $0 contract fees, friction is spread plus FX."""

    def test_zero_commission_leaves_only_the_spread(self):
        f = round_trip_friction(n_legs=2, half_spread=0.05, commission_per_contract=0.0)
        self.assertAlmostEqual(f, 0.20)

    def test_friction_fraction_includes_conversion_when_a_rate_is_set(self):
        no_fx = CostModel(table={}, default_half_spread=0.05,
                          commission_per_contract=0.0, fx_rate=0.0)
        with_fx = CostModel(table={}, default_half_spread=0.05,
                            commission_per_contract=0.0, fx_rate=0.015)
        self.assertGreater(
            with_fx.friction_fraction("Bull Put", 1.00, 2),
            no_fx.friction_fraction("Bull Put", 1.00, 2),
        )

    def test_conversion_is_charged_on_the_credit_actually_moved(self):
        # $1.00 credit x 100 = $100 converted, 1.5% each way = $3.00 = $0.03/share.
        model = CostModel(table={}, default_half_spread=0.0,
                          commission_per_contract=0.0, fx_rate=0.015)
        self.assertAlmostEqual(model.friction("Bull Put", n_legs=2, entry_credit=1.00),
                               0.03)


class TestRepricing(unittest.TestCase):
    """Re-pricing a closed trade under a different cost model. The ledger stores
    pnl_pct net of the friction it charged, so swapping models means adding back
    the old friction and subtracting the new one."""

    def setUp(self):
        self.flat = CostModel(table={}, default_half_spread=0.05,
                              commission_per_contract=0.65)
        self.measured = CostModel(
            table={"Bull Put": {"n": 28, "median_half_spread": 0.162}},
            default_half_spread=0.05, commission_per_contract=0.65,
        )

    def test_a_costlier_model_reduces_recorded_profit(self):
        repriced = reprice_pnl_pct(
            pnl_pct=0.40, strategy="Bull Put", entry_credit=1.00, n_legs=2,
            old_model=self.flat, new_model=self.measured,
        )
        self.assertLess(repriced, 0.40)

    def test_the_shift_is_exactly_the_difference_in_friction(self):
        old_f = self.flat.friction_fraction("Bull Put", 1.00, 2)
        new_f = self.measured.friction_fraction("Bull Put", 1.00, 2)
        repriced = reprice_pnl_pct(0.40, "Bull Put", 1.00, 2, self.flat, self.measured)
        self.assertAlmostEqual(repriced, 0.40 + old_f - new_f)

    def test_an_overcharged_structure_gains_when_repriced(self):
        # Bear Call is not in the measured table here, so it keeps the default —
        # use a table where it is measured cheaper than the flat charge.
        cheaper = CostModel(table={"Bear Call": {"n": 39, "median_half_spread": 0.025}},
                            default_half_spread=0.05, commission_per_contract=0.65)
        repriced = reprice_pnl_pct(0.10, "Bear Call", 0.50, 2, self.flat, cheaper)
        self.assertGreater(repriced, 0.10)

    def test_identical_models_leave_the_number_untouched(self):
        self.assertAlmostEqual(
            reprice_pnl_pct(0.40, "Bull Put", 1.00, 2, self.flat, self.flat), 0.40
        )


class TestMeasurementFromTheArchive(unittest.TestCase):
    """The table is measured by joining logged contracts to archived quotes on
    the trade's own entry date — no modelling, just what was quoted."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.archive = os.path.join(self.tmp.name, "archive.db")
        self.ledger = os.path.join(self.tmp.name, "ledger.db")

        led = sqlite3.connect(self.ledger)
        led.execute("CREATE TABLE trades (date TEXT, ticker TEXT, strike REAL,"
                    " expiration TEXT, type TEXT, strategy_name TEXT)")
        led.executemany("INSERT INTO trades VALUES (?,?,?,?,?,?)", [
            ("2026-07-01 10:00:00", "INTC", 80.0, "2026-08-21", "put", "Bull Put"),
            ("2026-07-01 10:00:00", "AMD", 200.0, "2026-08-21", "call", "Bear Call"),
        ])
        led.commit(); led.close()

        arc = sqlite3.connect(self.archive)
        arc.execute("CREATE TABLE chain_snapshots (symbol TEXT, snap_date TEXT,"
                    " type TEXT, strike REAL, expiration TEXT, bid REAL, ask REAL)")
        arc.executemany("INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?)", [
            ("INTC", "2026-07-01", "put", 80.0, "2026-08-21", 1.00, 1.40),   # half 0.20
            ("AMD", "2026-07-01", "call", 200.0, "2026-08-21", 2.00, 2.10),  # half 0.05
        ])
        arc.commit(); arc.close()

    def tearDown(self):
        self.tmp.cleanup()

    def test_measures_a_half_spread_per_structure(self):
        table = measure_half_spreads(self.archive, self.ledger)
        self.assertAlmostEqual(table["Bull Put"]["median_half_spread"], 0.20)
        self.assertAlmostEqual(table["Bear Call"]["median_half_spread"], 0.05)

    def test_records_the_sample_size_behind_each_number(self):
        table = measure_half_spreads(self.archive, self.ledger)
        self.assertEqual(table["Bull Put"]["n"], 1)

    def test_crossed_or_zero_quotes_are_not_measured(self):
        arc = sqlite3.connect(self.archive)
        arc.execute("UPDATE chain_snapshots SET bid=0, ask=0 WHERE symbol='INTC'")
        arc.commit(); arc.close()
        table = measure_half_spreads(self.archive, self.ledger)
        self.assertNotIn("Bull Put", table)


class SurfaceBackedCostModelTest(unittest.TestCase):
    def _surface(self):
        return SpreadSurface(
            {cell_key(0.50, 30.0, 500.0): Cell(40, 0.02, 42)}, {})

    def test_without_contract_context_the_strategy_table_still_wins(self):
        # Every existing caller passes no context and must be unaffected.
        m = CostModel(table={"Bull Put": {"n": 50, "median_half_spread": 0.162}},
                      default_half_spread=0.05, surface=self._surface())
        self.assertAlmostEqual(m.half_spread("Bull Put"), 0.162)

    def test_without_a_surface_behaviour_is_exactly_as_before(self):
        m = CostModel(table={"Bull Put": {"n": 50, "median_half_spread": 0.162}},
                      default_half_spread=0.05)
        self.assertAlmostEqual(m.half_spread("Bull Put"), 0.162)
        self.assertAlmostEqual(m.half_spread("Unknown"), 0.05)

    def test_with_contract_context_the_surface_prices_it(self):
        m = CostModel(table={"Bull Put": {"n": 50, "median_half_spread": 0.162}},
                      default_half_spread=0.05, surface=self._surface())
        # rel 0.02 * mid 2.50 = 0.05
        self.assertAlmostEqual(
            m.half_spread("Bull Put", mid=2.50, abs_delta=0.50, dte=30.0,
                          open_interest=500.0), 0.05)

    def test_partial_context_falls_back_to_the_strategy_table(self):
        # A mid without a delta is not enough to locate a cell; silently
        # guessing the missing dimension is how a wrong number gets used.
        m = CostModel(table={"Bull Put": {"n": 50, "median_half_spread": 0.162}},
                      default_half_spread=0.05, surface=self._surface())
        self.assertAlmostEqual(m.half_spread("Bull Put", mid=2.50), 0.162)


if __name__ == "__main__":
    unittest.main()
