"""The point of src/absolute_scores is that a contract's score does not depend
on what else was fetched alongside it.  These tests pin that property, the sign
conventions, and the degenerate-input behaviour.

See docs/ABSOLUTE_SCORES_20260807.md.
"""
import json
import unittest

import numpy as np
import pandas as pd

import src.options_screener as options_screener
from src.absolute_scores import (
    THETA_LOG_CENTRE,
    theta_pressure_score,
    vega_risk_score_absolute,
)


def _contract(**overrides):
    """A single scorable contract row.

    calculate_scores has no other test in the suite, so this fixture is built
    from the columns it accesses directly rather than through df.get().
    """
    row = dict(
        strike=100.0, theta=-0.05, premium=2.0, bid=1.95, ask=2.05, volume=500,
        openInterest=2000, spread_pct=0.05, abs_delta=0.45,
        impliedVolatility=0.35, prob_profit=0.45, rr_ratio=2.0,
        underlying=100.0, T_years=0.12, vega=0.2, gamma=0.03, delta=0.45,
        event_flag="", Trend_Aligned=False, decay_warning=False, sr_warning="",
        oi_wall_warning="", macro_warning="", div_warning="",
        squeeze_play=False, is_squeezing=False, Unusual_Whale=False,
        quote_freshness="fresh", symbol="TEST", return_on_risk=0.5,
        em_realism_score=0.6, seasonal_win_rate=0.5, short_interest=0.05,
        hv_30d=0.30, iv_vs_hv=0.05, gamma_ramp=False, ev_per_contract=5.0,
        theta_decay_pressure=0.025, expected_move=5.0, max_loss=200.0,
    )
    row.update(overrides)
    return row


def _score(rows):
    df = pd.DataFrame(rows)
    df["type"] = "call"
    df["expiration"] = "2026-09-18"
    with open("config.json") as fh:
        config = json.load(fh)
    return options_screener.calculate_scores(
        df, config, {"regime": "normal"}, "swing", "Scan", 7, 60)


class TestThetaPressureScore(unittest.TestCase):
    def test_identical_contracts_in_different_chains_score_identically(self):
        """The whole point: the score must not depend on the batch."""
        lone = theta_pressure_score(pd.Series([-0.05]), pd.Series([2.0]), False)
        crowd = theta_pressure_score(
            pd.Series([-0.05, -0.30, -0.01, -0.12]),
            pd.Series([2.0, 1.0, 5.0, 0.5]), False)
        self.assertAlmostEqual(float(lone.iloc[0]), float(crowd.iloc[0]), places=9)

    def test_buyers_score_slow_decay_higher(self):
        s = theta_pressure_score(pd.Series([-0.01, -0.50]), pd.Series([2.0, 2.0]), False)
        self.assertGreater(float(s.iloc[0]), float(s.iloc[1]))

    def test_sellers_score_fast_decay_higher(self):
        s = theta_pressure_score(pd.Series([-0.01, -0.50]), pd.Series([2.0, 2.0]), True)
        self.assertLess(float(s.iloc[0]), float(s.iloc[1]))

    def test_seller_and_buyer_scores_are_complementary(self):
        theta, prem = pd.Series([-0.04, -0.2]), pd.Series([1.5, 3.0])
        buy = theta_pressure_score(theta, prem, False)
        sell = theta_pressure_score(theta, prem, True)
        for b, s in zip(buy, sell):
            self.assertAlmostEqual(b + s, 1.0, places=9)

    def test_centre_of_the_mapping_scores_one_half(self):
        """A contract at the calibrated median pressure is neutral."""
        pressure = 10.0 ** THETA_LOG_CENTRE
        s = theta_pressure_score(pd.Series([-pressure]), pd.Series([1.0]), True)
        self.assertAlmostEqual(float(s.iloc[0]), 0.5, places=6)

    def test_bounded_and_finite_on_degenerate_input(self):
        s = theta_pressure_score(
            pd.Series([0.0, np.nan, -1e9, -0.05]),
            pd.Series([0.0, 1.0, np.nan, 2.0]), False)
        self.assertTrue(s.between(0.0, 1.0).all())
        self.assertTrue(np.isfinite(s).all())

    def test_zero_theta_is_neutral_not_extreme(self):
        """log10(0) is undefined; it must not become a 0 or 1 score."""
        s = theta_pressure_score(pd.Series([0.0]), pd.Series([2.0]), True)
        self.assertAlmostEqual(float(s.iloc[0]), 0.5, places=9)

    def test_preserves_a_non_default_index(self):
        idx = pd.Index([7, 3, 11])
        s = theta_pressure_score(
            pd.Series([-0.05, -0.1, -0.2], index=idx),
            pd.Series([2.0, 1.0, 4.0], index=idx), False)
        self.assertTrue(s.index.equals(idx))


class TestVegaRiskScoreAbsolute(unittest.TestCase):
    def test_high_vega_at_high_iv_percentile_scores_worst(self):
        s = vega_risk_score_absolute(pd.Series([0.5, 0.5]), pd.Series([0.9, 0.1]))
        self.assertLess(float(s.iloc[0]), float(s.iloc[1]))

    def test_missing_iv_percentile_is_neutral_not_zero(self):
        s = vega_risk_score_absolute(pd.Series([0.2]), pd.Series([np.nan]))
        self.assertTrue(0.0 < float(s.iloc[0]) < 1.0)

    def test_independent_of_batch(self):
        lone = vega_risk_score_absolute(pd.Series([0.2]), pd.Series([0.5]))
        crowd = vega_risk_score_absolute(
            pd.Series([0.2, 9.0, 0.01]), pd.Series([0.5, 0.5, 0.5]))
        self.assertAlmostEqual(float(lone.iloc[0]), float(crowd.iloc[0]), places=9)

    def test_zero_iv_percentile_means_no_penalty(self):
        s = vega_risk_score_absolute(pd.Series([5.0]), pd.Series([0.0]))
        self.assertAlmostEqual(float(s.iloc[0]), 1.0, places=9)

    def test_bounded_on_degenerate_input(self):
        s = vega_risk_score_absolute(
            pd.Series([0.0, np.nan, 1e9]), pd.Series([0.5, 2.0, -1.0]))
        self.assertTrue(s.between(0.0, 1.0).all())
        self.assertTrue(np.isfinite(s).all())


class TestCalculateScoresIsBatchIndependent(unittest.TestCase):
    """End-to-end: the defect was that these scores moved with the batch.

    Nothing else in the suite calls calculate_scores, so without these the
    wiring in options_screener is only covered by a live scan.
    """

    def test_theta_score_is_identical_alone_and_in_a_crowd(self):
        # Every other contract decays FASTER than the target, so the old
        # rank_norm put it at rank 0 (score 1.0) here and at 0.5 alone. A crowd
        # that straddles the target would leave it at the median, where the
        # rank happens to agree with neutral and the test proves nothing.
        target = _contract()
        alone = _score([target])
        crowd = _score([
            target,
            _contract(strike=110.0, theta=-0.40, premium=0.4, abs_delta=0.15),
            _contract(strike=90.0, theta=-0.50, premium=2.0, abs_delta=0.80),
        ])
        self.assertAlmostEqual(float(alone["theta_score"].iloc[0]),
                               float(crowd["theta_score"].iloc[0]), places=9)

    def test_vega_risk_score_is_identical_alone_and_in_a_crowd(self):
        target = _contract()
        alone = _score([target])
        crowd = _score([
            target,
            _contract(strike=110.0, vega=2.5),
            _contract(strike=90.0, vega=0.01),
        ])
        self.assertAlmostEqual(float(alone["vega_risk_score"].iloc[0]),
                               float(crowd["vega_risk_score"].iloc[0]), places=9)

    def test_theta_score_matches_the_standalone_mapping(self):
        """The wiring must read theta/premium, not some other pair."""
        scored = _score([_contract()])
        expected = theta_pressure_score(
            pd.Series([-0.05]), pd.Series([2.0]), False)
        self.assertAlmostEqual(float(scored["theta_score"].iloc[0]),
                               float(expected.iloc[0]), places=9)

    def test_a_single_contract_scan_still_scores(self):
        """rank_norm returned a flat 0.5 for n == 1; the mapping must not."""
        scored = _score([_contract(theta=-0.30, premium=1.0)])
        self.assertNotAlmostEqual(float(scored["theta_score"].iloc[0]), 0.5, places=3)


if __name__ == "__main__":
    unittest.main()
