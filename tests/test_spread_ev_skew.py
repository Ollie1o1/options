"""Unit tests for staged SVI skew-shifted spread EV reporting in src.spread_scoring."""
from __future__ import annotations

import unittest

from src.spread_scoring import _apply_structure_ev, compute_skew_shifted_spread_ev


class TestSpreadEvSkew(unittest.TestCase):
    def test_skew_shifted_ev_eliminates_steep_put_skew_penalty(self) -> None:
        # Steep index put skew: short 95P has IV 22%, long 90P has IV 26%, ATM IV 18%, RV 16%.
        # Under flat RV=16%, long 90P looks wildly overpriced vs 26% IV, hurting spread EV.
        # Under uniform skew shift (scale = 16/18), both legs scale down proportionally -> positive VRP!
        spot, t_y, r = 100.0, 45.0 / 365.0, 0.04
        from src.spread_scoring import _bs_option_price

        short_mkt = _bs_option_price(spot, 95.0, t_y, r, 0.22, "put")
        long_mkt = _bs_option_price(spot, 90.0, t_y, r, 0.26, "put")
        short_leg = {
            "strike": 95.0,
            "type": "put",
            "impliedVolatility": 0.22,
            "mid": short_mkt,
            "underlying_price": spot,
            "dte": 45.0,
            "hv_basis": 0.16,
            "atm_iv": 0.18,
            "ev_gross_instrument_per_contract": (_bs_option_price(spot, 95.0, t_y, r, 0.16, "put") - short_mkt) * 100.0,
            "ev_cost_per_contract": 2.0,
        }
        long_leg = {
            "strike": 90.0,
            "type": "put",
            "impliedVolatility": 0.26,
            "mid": long_mkt,
            "underlying_price": spot,
            "dte": 45.0,
            "hv_basis": 0.16,
            "atm_iv": 0.18,
            "ev_gross_instrument_per_contract": (_bs_option_price(spot, 90.0, t_y, r, 0.16, "put") - long_mkt) * 100.0,
            "ev_cost_per_contract": 2.0,
        }

        res = compute_skew_shifted_spread_ev(
            sold=[short_leg],
            bought=[long_leg],
            spot=spot,
            t_years=t_y,
            r=r,
            sigma_rv=0.16,
            sigma_atm=0.18,
            cost_per_contract=4.0,
        )
        self.assertIsNotNone(res)
        self.assertGreater(res["ev_skew_adj_gross_per_contract"], 0.0)

        # Report mode preserves legacy ev_per_contract while attaching ev_skew_adj_per_contract
        row_report = {}
        _apply_structure_ev(row_report, [short_leg], [long_leg], config={"auto_log": {"spread_ev_vol_model": "report"}})
        self.assertIn("ev_skew_adj_per_contract", row_report)
        self.assertEqual(row_report["ev_per_contract"], row_report["ev_flat_per_contract"])

        # svi_shifted mode replaces ev_per_contract with skew-adjusted value
        row_shift = {}
        _apply_structure_ev(row_shift, [short_leg], [long_leg], config={"auto_log": {"spread_ev_vol_model": "svi_shifted"}})
        self.assertEqual(row_shift["ev_per_contract"], row_shift["ev_skew_adj_per_contract"])


if __name__ == "__main__":
    unittest.main()
