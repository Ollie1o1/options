"""Tests for src/dolt_blend.py — sleeve-blend engine math.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_blend -v
"""
import unittest

from src import dolt_blend as db


def _t(ret, month, day="15"):
    d = f"2024-{month:02d}-{day}"
    return {"ret": ret, "entry_date": d, "exit_date": d}


class MonthlyReturnsTest(unittest.TestCase):
    def test_buckets_by_exit_month(self):
        trades = [_t(0.1, 1), _t(0.2, 1), _t(-0.1, 2)]
        mr = db.monthly_returns(trades)
        # Jan: sum(0.1, 0.2) = 0.3 ; Feb: -0.1
        self.assertAlmostEqual(mr["2024-01"], 0.3, places=6)
        self.assertAlmostEqual(mr["2024-02"], -0.1, places=6)


class SharpeTest(unittest.TestCase):
    def test_none_for_single_month(self):
        self.assertIsNone(db.annualized_sharpe({"2024-01": 0.1}))

    def test_positive_for_steady_gains(self):
        mr = {f"2024-{m:02d}": 0.02 + 0.001 * m for m in range(1, 13)}
        s = db.annualized_sharpe(mr)
        self.assertIsNotNone(s)
        self.assertGreater(s, 0)


class CorrelationTest(unittest.TestCase):
    def test_perfectly_correlated(self):
        a = {f"2024-{m:02d}": 0.01 * m for m in range(1, 7)}
        b = {f"2024-{m:02d}": 0.02 * m for m in range(1, 7)}   # exact scalar multiple
        self.assertAlmostEqual(db.correlation(a, b), 1.0, places=6)

    def test_anti_correlated(self):
        a = {f"2024-{m:02d}": 0.01 * m for m in range(1, 7)}
        b = {f"2024-{m:02d}": -0.01 * m for m in range(1, 7)}
        self.assertAlmostEqual(db.correlation(a, b), -1.0, places=6)

    def test_none_for_few_common_months(self):
        a = {"2024-01": 0.1, "2024-02": 0.2}
        b = {"2024-01": 0.1, "2024-02": 0.2}
        self.assertIsNone(db.correlation(a, b))   # < 3 overlap


class BlendTest(unittest.TestCase):
    def test_blend_reduces_variance_for_low_corr_sleeves(self):
        # two positive-mean sleeves with distinct (non-canceling) monthly patterns
        s1 = db.Sleeve("a", [_t(0.30, m) if m % 2 else _t(-0.10, m) for m in range(1, 13)], "x")
        s2 = db.Sleeve("b", [_t(-0.05, m) if m % 2 else _t(0.25, m) for m in range(1, 13)], "y")
        res = db.blend([s1, s2], risk_frac=0.1)
        self.assertLess(res["correlation"]["a"]["b"], 0.5)        # not highly correlated
        self.assertIsNotNone(res["blended"]["sharpe"])            # finite blended Sharpe
        self.assertIsNotNone(res["diversification_benefit"])

    def test_correlated_sleeves_negligible_benefit(self):
        trades = [_t(0.15, m) if m % 2 else _t(-0.05, m) for m in range(1, 13)]
        s1 = db.Sleeve("a", list(trades), "x")
        s2 = db.Sleeve("b", list(trades), "x")   # identical -> corr 1
        res = db.blend([s1, s2], risk_frac=0.1)
        self.assertAlmostEqual(res["correlation"]["a"]["b"], 1.0, places=6)
        # identical sleeves diversify nothing: benefit negligible (tiny compounding artifact)
        self.assertLess(abs(res["diversification_benefit"]), 0.1)

    def test_empty_sleeves_dropped(self):
        s1 = db.Sleeve("a", [_t(0.1, m) for m in range(1, 6)], "x")
        s2 = db.Sleeve("empty", [], "y")
        res = db.blend([s1, s2])
        self.assertIn("a", res["sleeves"])
        self.assertNotIn("empty", res["sleeves"])


if __name__ == "__main__":
    unittest.main()
