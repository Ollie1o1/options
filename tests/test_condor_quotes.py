"""Iron condors emitted no per-leg quotes, so they could not be priced.

`find_iron_condors` produced strikes, credits and return_on_risk and nothing
else, so `candidate_verdict._legs_of` refused every condor and the cohort could
never be verdict-ranked — the reason the spreads/condors auto-log path was left
on `quality_score` when the single-leg path moved off it.

See docs/ADJUSTMENT_STACK_20260807.md §7.
"""
import unittest

import numpy as np
import pandas as pd

import src.candidate_verdict as cv
from src.options_screener import find_iron_condors

EXP = "2026-09-18"

# strike, put delta, put mid, call delta, call mid
CHAIN = [
    (80, -0.08, 0.30, 0.92, 20.00),
    (85, -0.16, 0.70, 0.85, 16.00),
    (90, -0.26, 1.60, 0.75, 12.00),
    (95, -0.38, 2.80, 0.62, 8.50),
    (100, -0.50, 4.50, 0.50, 5.50),
    (105, -0.62, 7.00, 0.38, 2.80),
    (110, -0.74, 9.50, 0.26, 1.60),
    (115, -0.84, 12.50, 0.16, 0.70),
    (120, -0.92, 16.00, 0.08, 0.30),
]

SCORE_COLS = [
    "pop_score", "ev_score", "rr_score", "liquidity_score", "momentum_score",
    "iv_rank_score", "theta_score", "iv_advantage_score", "vrp_score",
    "iv_mispricing_score", "skew_align_score", "vega_risk_score",
    "term_structure_score", "catalyst_score", "em_realism_score",
    "gamma_theta_score", "gex_score", "gamma_magnitude_score",
    "gamma_pin_score", "iv_velocity_score", "max_pain_score",
    "oi_change_score", "option_rvol_score", "pcr_score",
    "sentiment_score_norm", "spread_score", "trader_pref_score",
]


def _chain(half_spread=0.05):
    rows = []
    for strike, pdelta, pmid, cdelta, cmid in CHAIN:
        base = dict(symbol="TEST", expiration=EXP, strike=strike, volume=1500,
                    openInterest=2500, impliedVolatility=0.30, gamma=0.02,
                    vega=0.15, theta=-0.04)
        for opt_type, delta, mid in (("put", pdelta, pmid), ("call", cdelta, cmid)):
            row = dict(base)
            row.update(type=opt_type, delta=delta, premium=mid,
                       bid=round(mid - half_spread, 4),
                       ask=round(mid + half_spread, 4))
            rows.append(row)
    df = pd.DataFrame(rows)
    rng = np.random.default_rng(42)
    for c in SCORE_COLS:
        df[c] = rng.uniform(0.4, 0.8, len(df))
    df["quality_score"] = rng.uniform(0.5, 0.7, len(df))
    return df


class TestCondorCarriesItsQuotes(unittest.TestCase):
    def setUp(self):
        self.condors = find_iron_condors(_chain())
        if self.condors.empty:
            self.skipTest("fixture produced no condor")

    def test_all_four_legs_carry_a_two_sided_quote(self):
        row = self.condors.iloc[0]
        for prefix in ("short_put", "long_put", "short_call", "long_call"):
            for side in ("bid", "ask"):
                col = f"{prefix}_{side}"
                self.assertIn(col, self.condors.columns)
                self.assertTrue(pd.notna(row[col]), f"{col} is NaN")

    def test_the_condor_can_now_be_priced(self):
        row = self.condors.iloc[0].to_dict()
        row["strategy_name"] = "Iron Condor"
        verdict = cv.verdict_for(row)
        self.assertTrue(verdict.priced, verdict.reason)
        self.assertIsNotNone(verdict.round_trip_pct)

    def test_a_width_is_carried_so_the_breakeven_can_be_computed(self):
        row = self.condors.iloc[0].to_dict()
        row["strategy_name"] = "Iron Condor"
        self.assertIn("spread_width", self.condors.columns)
        self.assertGreater(float(row["spread_width"]), 0.0)
        self.assertIsNotNone(cv.verdict_for(row).breakeven)

    def test_a_wider_market_reports_more_friction(self):
        tight = find_iron_condors(_chain(half_spread=0.02)).iloc[0].to_dict()
        wide = find_iron_condors(_chain(half_spread=0.25)).iloc[0].to_dict()
        tight["strategy_name"] = wide["strategy_name"] = "Iron Condor"
        self.assertLess(cv.verdict_for(tight).round_trip_pct,
                        cv.verdict_for(wide).round_trip_pct)

    def test_the_quotes_belong_to_the_legs_that_were_chosen(self):
        """A quote copied off the wrong strike would price a different condor."""
        chain = _chain()
        row = find_iron_condors(chain).iloc[0]
        for prefix, opt_type in (("short_put", "put"), ("long_put", "put"),
                                 ("short_call", "call"), ("long_call", "call")):
            leg = chain[(chain["type"] == opt_type)
                        & (chain["strike"] == row[f"{prefix}_strike"])].iloc[0]
            self.assertAlmostEqual(float(row[f"{prefix}_bid"]), float(leg["bid"]), places=6)
            self.assertAlmostEqual(float(row[f"{prefix}_ask"]), float(leg["ask"]), places=6)


if __name__ == "__main__":
    unittest.main()
