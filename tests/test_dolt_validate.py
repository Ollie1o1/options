"""Tests for src/dolt_validate.py — reconstructable scorer features + IC harness.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_validate -v
"""
import unittest

from src import dolt_validate as dv


def _chain(date="2024-03-15"):
    return [
        {"symbol": "X", "date": date, "expiration": "2024-04-19", "strike": 100.0,
         "type": "call", "bid": 3.0, "ask": 3.2, "mid": 3.1, "iv": 0.30,
         "delta": 0.50, "gamma": 0.02, "theta": -0.04, "vega": 0.12, "rho": 0.02},
        {"symbol": "X", "date": date, "expiration": "2024-04-19", "strike": 105.0,
         "type": "call", "bid": 1.0, "ask": 1.2, "mid": 1.1, "iv": 0.33,
         "delta": 0.32, "gamma": 0.02, "theta": -0.03, "vega": 0.10, "rho": 0.01},
        {"symbol": "X", "date": date, "expiration": "2024-06-21", "strike": 100.0,
         "type": "call", "bid": 5.0, "ask": 5.3, "mid": 5.15, "iv": 0.34,
         "delta": 0.55, "gamma": 0.01, "theta": -0.02, "vega": 0.25, "rho": 0.04},
        {"symbol": "X", "date": date, "expiration": "2024-04-19", "strike": 95.0,
         "type": "put", "bid": 1.1, "ask": 1.3, "mid": 1.2, "iv": 0.37,
         "delta": -0.30, "gamma": 0.02, "theta": -0.03, "vega": 0.10, "rho": -0.01},
    ]


class FeatureTest(unittest.TestCase):
    def test_term_structure_slope_in_unit_range(self):
        f = dv.term_structure_score(_chain(), spot=100.0, asof="2024-03-15")
        self.assertTrue(0.0 <= f <= 1.0)

    def test_term_structure_contango_scores_above_half(self):
        # far IV (0.34) > near IV (0.30) → contango → > 0.5
        f = dv.term_structure_score(_chain(), spot=100.0, asof="2024-03-15")
        self.assertGreater(f, 0.5)

    def test_skew_score_in_unit_range(self):
        f = dv.skew_score(_chain(), spot=100.0)
        self.assertTrue(0.0 <= f <= 1.0)

    def test_em_realism_peaks_when_realized_equals_em(self):
        em = 100.0 * 0.30 * (30 / 365.0) ** 0.5
        best = dv.em_realism_score(0.30, 30, 100.0, em)
        worse = dv.em_realism_score(0.30, 30, 100.0, em * 3)
        self.assertGreater(best, worse)
        self.assertTrue(0.0 <= best <= 1.0)

    def test_moneyness_peaks_at_target_delta(self):
        self.assertAlmostEqual(dv.moneyness_score(0.30, target=0.30), 1.0)
        self.assertLess(dv.moneyness_score(0.60, target=0.30), 1.0)

    def test_iv_level_low_iv_scores_high(self):
        hist = [0.40, 0.45, 0.50, 0.42, 0.48]
        cheap = dv.iv_level_score(0.30, hist)   # below all → high
        rich = dv.iv_level_score(0.55, hist)    # above all → low
        self.assertGreater(cheap, rich)

    def test_combine_features_returns_unit_score(self):
        feats = {"term_structure": 0.6, "skew": 0.4, "moneyness": 0.5,
                 "theta": 0.5, "iv_level": 0.5}
        s = dv.combine_features(feats)
        self.assertTrue(0.0 <= s <= 1.0)


class ICAggregationTest(unittest.TestCase):
    def test_compute_ic_pearson(self):
        samples = [{"score": s, "ret": s * 2 - 1} for s in [0.1, 0.3, 0.5, 0.7, 0.9]]
        out = dv.compute_ic(samples)
        self.assertGreater(out["ic_pearson"], 0.99)
        self.assertEqual(out["n"], 5)

    def test_compute_ic_handles_too_few(self):
        out = dv.compute_ic([{"score": 0.5, "ret": 0.1}])
        self.assertEqual(out["n"], 1)
        self.assertIsNone(out["ic_pearson"])

    def test_quintiles_partition(self):
        samples = [{"score": i / 100.0, "ret": (i - 50) / 100.0} for i in range(100)]
        qs = dv._quintiles(samples)
        self.assertEqual(len(qs), 5)
        self.assertLess(qs[0]["avg_score"], qs[-1]["avg_score"])


if __name__ == "__main__":
    unittest.main()
