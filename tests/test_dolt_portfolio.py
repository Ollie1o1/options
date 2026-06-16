"""Tests for src/dolt_portfolio.py — sizing + equity curve.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_dolt_portfolio -v
"""
import unittest

from src import dolt_portfolio as pf


def _t(ret, entry, exit_):
    return {"ret": ret, "entry_date": entry, "exit_date": exit_}


def _tm(ret, entry, exit_, max_risk, exit_reason="take_profit"):
    return {"ret": ret, "entry_date": entry, "exit_date": exit_,
            "max_risk": max_risk, "exit_reason": exit_reason}


class MaxConcurrentTest(unittest.TestCase):
    def test_non_overlapping(self):
        trades = [_t(0.1, "2024-01-01", "2024-01-05"),
                  _t(0.1, "2024-01-06", "2024-01-10")]
        self.assertEqual(pf.max_concurrent(trades), 1)

    def test_overlapping(self):
        trades = [_t(0.1, "2024-01-01", "2024-01-10"),
                  _t(0.1, "2024-01-03", "2024-01-08"),
                  _t(0.1, "2024-01-05", "2024-01-06")]
        self.assertEqual(pf.max_concurrent(trades), 3)

    def test_close_frees_slot_same_day(self):
        # one closes the same day another opens -> not counted as concurrent
        trades = [_t(0.1, "2024-01-01", "2024-01-05"),
                  _t(0.1, "2024-01-05", "2024-01-09")]
        self.assertEqual(pf.max_concurrent(trades), 1)


class EquityCurveTest(unittest.TestCase):
    def test_compounds_in_exit_order(self):
        # +100% on max-risk at 1% risk -> +1% equity per trade, twice
        trades = [_t(1.0, "2024-01-01", "2024-01-10"),
                  _t(1.0, "2024-01-05", "2024-01-08")]
        out = pf.equity_curve(trades, start_equity=100_000.0, risk_frac=0.01)
        # exit order: 01-08 then 01-10; 100000 * 1.01 * 1.01
        self.assertAlmostEqual(out["end_equity"], 100_000 * 1.01 * 1.01, places=4)
        self.assertEqual(out["n"], 2)
        self.assertAlmostEqual(out["total_return"], 1.01 * 1.01 - 1.0, places=6)

    def test_max_drawdown_negative(self):
        trades = [_t(1.0, "2024-01-01", "2024-01-02"),   # up
                  _t(-1.0, "2024-01-03", "2024-01-04"),  # down (full max-risk loss)
                  _t(-1.0, "2024-01-05", "2024-01-06")]  # down again
        out = pf.equity_curve(trades, risk_frac=0.10)
        self.assertLess(out["max_drawdown"], 0.0)

    def test_empty(self):
        out = pf.equity_curve([], start_equity=100_000.0)
        self.assertEqual(out["n"], 0)
        self.assertEqual(out["end_equity"], 100_000.0)
        self.assertEqual(out["max_drawdown"], 0.0)

    def test_risk_frac_scales_pnl(self):
        trades = [_t(0.5, "2024-01-01", "2024-01-05")]
        small = pf.equity_curve(trades, risk_frac=0.01)["end_equity"]
        big = pf.equity_curve(trades, risk_frac=0.05)["end_equity"]
        self.assertGreater(big, small)


class MarginProfileTest(unittest.TestCase):
    def test_peak_margin_sums_overlap(self):
        # two spreads max_risk 8.5 each, overlapping -> peak = 2 * 850
        trades = [_tm(0.1, "2024-01-01", "2024-01-10", 8.5),
                  _tm(0.1, "2024-01-03", "2024-01-08", 8.5)]
        mp = pf.margin_profile(trades)
        self.assertAlmostEqual(mp["peak_margin"], 2 * 8.5 * 100, places=4)

    def test_peak_margin_non_overlap(self):
        trades = [_tm(0.1, "2024-01-01", "2024-01-05", 8.5),
                  _tm(0.1, "2024-01-06", "2024-01-10", 8.5)]
        mp = pf.margin_profile(trades)
        self.assertAlmostEqual(mp["peak_margin"], 8.5 * 100, places=4)

    def test_return_on_peak_capital(self):
        # single trade ret +0.5 on max_risk 10 -> pnl = 0.5*10*100 = 500; peak = 1000
        trades = [_tm(0.5, "2024-01-01", "2024-01-05", 10.0)]
        mp = pf.margin_profile(trades)
        self.assertAlmostEqual(mp["total_pnl"], 500.0, places=4)
        self.assertAlmostEqual(mp["return_on_peak_margin"], 0.5, places=6)

    def test_assignment_risk_counts_stop_loss(self):
        trades = [_tm(0.1, "2024-01-01", "2024-01-05", 8.5, "take_profit"),
                  _tm(-1.0, "2024-01-02", "2024-01-06", 8.5, "stop_loss")]
        mp = pf.margin_profile(trades)
        self.assertEqual(mp["assignment_risk_trades"], 1)
        self.assertAlmostEqual(mp["assignment_risk_frac"], 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
