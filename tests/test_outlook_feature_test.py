"""The one genuinely new piece of logic in scripts/outlook_feature_test.py:
the day-clustered t-stat on the RESIDUAL, which residual_ic itself does not
report (only a naive, non-clustered p-value) — this is the actual statistic
docs/PREREG_OUTLOOK_FEATURE_20260905.md's decision rule needs.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_outlook_feature_test -v
"""
from __future__ import annotations

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.outlook_feature_test import residualized_clustered_t


class _T:
    def __init__(self, entry_date, roc, capital_at_risk=100.0, **feats):
        self.entry_date = entry_date
        self.capital_at_risk = capital_at_risk
        self.pnl = roc * capital_at_risk
        self.features = feats
        self.exit_date = "2024-12-31"


def _dates(n):
    return [f"2024-{1 + i % 12:02d}-{1 + i % 28:02d}" for i in range(n)]


class ResidualizedClusteredTTest(unittest.TestCase):

    def test_a_feature_independent_of_the_controls_reports_a_real_t(self):
        rng = random.Random(3)
        trades = []
        for d in _dates(200):
            rich, own = rng.random(), rng.random()
            roc = 0.3 * (rich - 0.5) + 0.9 * (own - 0.5)
            trades.append(_T(d, roc=roc, credit_pct_width=rich, atm_iv=rich,
                            outlook_composite=own))
        row = residualized_clustered_t(trades)
        self.assertIsNotNone(row["t_clustered"])
        self.assertGreater(abs(row["t_clustered"]), 3.0)

    def test_a_feature_that_is_just_the_control_reports_none(self):
        rng = random.Random(4)
        trades = [_T(d, roc=rng.random() - 0.5, credit_pct_width=r,
                    atm_iv=r, outlook_composite=r)
                 for d, r in zip(_dates(200), [rng.random() for _ in range(200)])]
        row = residualized_clustered_t(trades)
        self.assertIsNone(row["ic"])

    def test_too_few_trades_reports_none_not_zero(self):
        row = residualized_clustered_t(
            [_T(d, roc=0.1, credit_pct_width=0.5, atm_iv=0.5,
               outlook_composite=0.5) for d in _dates(4)])
        self.assertEqual(row["n"], 0)
        self.assertIsNone(row["t_clustered"])

    def test_open_trades_are_excluded(self):
        t = _T("2024-01-01", roc=0.1, credit_pct_width=0.5, atm_iv=0.5,
              outlook_composite=0.9)
        t.exit_date = None
        row = residualized_clustered_t([t])
        self.assertEqual(row["n"], 0)


if __name__ == "__main__":
    unittest.main()
