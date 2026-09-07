"""The one genuinely new piece of logic in scripts/ensemble_test.py: finding
the best single residualized feature on a holdout population, for the
"does the ensemble beat the best single feature" half of
docs/PREREG_ENSEMBLE_20260905.md's decision rule.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_ensemble_test_script -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.ensemble_test import best_single_residualized_ic


class _T:
    def __init__(self, entry_date, roc, capital_at_risk=100.0, **feats):
        self.entry_date = entry_date
        self.capital_at_risk = capital_at_risk
        self.pnl = roc * capital_at_risk
        self.features = feats
        self.exit_date = "2024-12-31"


def _dates(n):
    return [f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)]


class BestSingleResidualizedIcTest(unittest.TestCase):

    def test_picks_the_larger_of_two_real_features(self):
        rng = random.Random(9)
        trades = []
        for d in _dates(200):
            rich, weak, strong = rng.random(), rng.random(), rng.random()
            roc = (0.2 * (rich - 0.5) + 0.2 * (weak - 0.5)
                  + 0.8 * (strong - 0.5))
            trades.append(_T(d, roc=roc, credit_pct_width=rich, atm_iv=rich,
                            weak=weak, strong=strong))
        best = best_single_residualized_ic(trades, ["weak", "strong"])
        # The stronger driver must set the max, not be averaged away.
        from src.alloc.attribution import residual_ic
        strong_ic = abs(residual_ic(trades, "strong")["ic"])
        self.assertAlmostEqual(best, strong_ic, places=6)

    def test_a_feature_that_cannot_be_measured_contributes_zero_not_a_crash(self):
        trades = [_T(d, roc=0.1, credit_pct_width=0.5, atm_iv=0.5)
                 for d in _dates(20)]
        best = best_single_residualized_ic(trades, ["never_present"])
        self.assertEqual(best, 0.0)

    def test_open_trades_are_excluded_before_measuring(self):
        t = _T("2024-01-01", roc=0.1, credit_pct_width=0.5, atm_iv=0.5,
              own=0.9)
        t.exit_date = None
        best = best_single_residualized_ic([t], ["own"])
        self.assertEqual(best, 0.0)


if __name__ == "__main__":
    unittest.main()
