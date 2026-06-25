"""Unit tests for the leverage execution-vehicle selector.

Risk-matched option-vs-leverage comparison: size leverage so the margin at
risk equals the option premium, then report the implied leverage, liquidation
price, funding drag, and which vehicle the IV regime favors.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_leverage_selector -v
"""
from __future__ import annotations

import math
import unittest

from src.leverage_selector import leverage_vehicle, leverage_vehicle_line


def _call_row(**over):
    row = {
        "type": "call",
        "underlying": 270.0,
        "strike": 270.0,
        "premium": 6.42,
        "delta": 0.45,
        "theta": -0.31,
        "T_years": 21 / 365.0,
        "iv_surface_residual": 0.0,
        "vrp_regime": "NORMAL",
        "symbol": "AAPL",
    }
    row.update(over)
    return row


# Venue maintenance-margin default the selector falls back to when none supplied.
DEFAULT_MAINT = 0.005


class RiskMatchedMathTest(unittest.TestCase):
    def test_implied_leverage_is_delta_notional_over_premium(self):
        r = _call_row(underlying=270.0, premium=6.42, delta=0.45)
        res = leverage_vehicle(r)
        # margin (risk-matched) = premium*100; notional = |delta|*100*spot
        expected_lev = (0.45 * 270.0) / 6.42
        self.assertAlmostEqual(res["leverage"], expected_lev, places=4)

    def test_margin_equals_option_premium_per_contract(self):
        r = _call_row(premium=6.42)
        res = leverage_vehicle(r)
        self.assertAlmostEqual(res["margin"], 642.0, places=6)

    def test_notional_is_delta_matched(self):
        r = _call_row(underlying=270.0, delta=0.45)
        res = leverage_vehicle(r)
        self.assertAlmostEqual(res["notional"], 0.45 * 100 * 270.0, places=6)


class LiquidationTest(unittest.TestCase):
    def test_call_long_liquidates_below_spot(self):
        r = _call_row()
        res = leverage_vehicle(r)
        self.assertEqual(res["direction"], "long")
        self.assertLess(res["liq_price"], r["underlying"])

    def test_put_short_liquidates_above_spot(self):
        r = _call_row(type="put", delta=-0.45)
        res = leverage_vehicle(r)
        self.assertEqual(res["direction"], "short")
        self.assertGreater(res["liq_price"], r["underlying"])

    def test_liq_move_uses_maintenance_haircut(self):
        r = _call_row(underlying=100.0, premium=10.0, delta=1.0)
        # leverage = (1.0*100)/10 = 10x ; liq move = (1/10)*(1-maint)
        res = leverage_vehicle(r)
        expected_move = (1.0 / 10.0) * (1.0 - DEFAULT_MAINT)
        self.assertAlmostEqual(res["liq_move_frac"], expected_move, places=6)
        self.assertAlmostEqual(res["liq_price"], 100.0 * (1 - expected_move), places=6)


class FundingTest(unittest.TestCase):
    def test_funding_derived_from_live_rate_when_no_override(self):
        # Funding for a leveraged delta-1 position = live short rate + venue spread.
        r = _call_row(underlying=270.0, delta=0.45, T_years=21 / 365.0)
        res = leverage_vehicle(
            r, config={"leverage_selector": {"carry_spread_annual": 0.05}},
            rate_fetcher=lambda: 0.04)
        notional = 0.45 * 100 * 270.0
        daily = (0.04 + 0.05) / 365.0
        self.assertAlmostEqual(res["funding_cost"], notional * daily * 21, places=4)

    def test_explicit_funding_rate_overrides_live(self):
        r = _call_row(underlying=270.0, delta=0.45, T_years=21 / 365.0)
        res = leverage_vehicle(
            r, config={"leverage_selector": {"funding_rate_daily": 0.001}},
            rate_fetcher=lambda: 0.99)  # live rate must be ignored when overridden
        notional = 0.45 * 100 * 270.0
        self.assertAlmostEqual(res["funding_cost"], notional * 0.001 * 21, places=4)

    def test_live_rate_failure_falls_back_safely(self):
        def boom():
            raise RuntimeError("no network")
        res = leverage_vehicle(_call_row(), rate_fetcher=boom)
        self.assertIsNotNone(res)
        self.assertGreaterEqual(res["funding_cost"], 0.0)


class VerdictSwitchTest(unittest.TestCase):
    def test_rich_surface_favors_leverage(self):
        r = _call_row(iv_surface_residual=0.05)
        res = leverage_vehicle(r)
        self.assertEqual(res["vehicle"], "LEVERAGE")

    def test_cheap_surface_favors_option(self):
        r = _call_row(iv_surface_residual=-0.05)
        res = leverage_vehicle(r)
        self.assertEqual(res["vehicle"], "OPTION")

    def test_flat_surface_is_toss_up(self):
        r = _call_row(iv_surface_residual=0.0)
        res = leverage_vehicle(r)
        self.assertEqual(res["vehicle"], "TOSS-UP")


class LeverageCapTest(unittest.TestCase):
    def test_implied_leverage_capped_at_venue_max(self):
        # tiny premium -> huge implied leverage, must clamp
        r = _call_row(underlying=100.0, premium=0.10, delta=0.50)
        res = leverage_vehicle(r, config={"leverage_selector": {"max_leverage": 50}})
        self.assertLessEqual(res["leverage"], 50.0)
        self.assertTrue(res["capped"])
        # when capped, margin to hold the notional exceeds the option premium
        self.assertGreater(res["margin"], r["premium"] * 100)


class GuardTest(unittest.TestCase):
    def test_none_when_premium_missing(self):
        r = _call_row()
        del r["premium"]
        self.assertIsNone(leverage_vehicle(r))

    def test_none_when_premium_nonpositive(self):
        self.assertIsNone(leverage_vehicle(_call_row(premium=0.0)))

    def test_none_when_delta_missing(self):
        r = _call_row()
        del r["delta"]
        self.assertIsNone(leverage_vehicle(r))

    def test_none_when_disabled(self):
        res = leverage_vehicle(_call_row(), config={"leverage_selector": {"enabled": False}})
        self.assertIsNone(res)

    def test_never_raises_on_garbage(self):
        for bad in [{}, {"type": "call"}, {"premium": "x", "delta": None}]:
            self.assertIsNone(leverage_vehicle(bad))


class LineTest(unittest.TestCase):
    def test_line_mentions_leverage_and_liquidation(self):
        line = leverage_vehicle_line(_call_row(iv_surface_residual=0.05))
        self.assertIsNotNone(line)
        self.assertIn("Leverage read", line)
        self.assertRegex(line, r"\d+(\.\d+)?x")  # an "Nx" leverage figure
        self.assertIn("liq", line.lower())

    def test_line_none_for_spread_like_row(self):
        # a row with no single-leg premium/delta should produce no line
        self.assertIsNone(leverage_vehicle_line({"type": "spread"}))


if __name__ == "__main__":
    unittest.main()
