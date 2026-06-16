"""Tests for src/portfolio_guard.py — basket concentration guard.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_portfolio_guard -v
"""
import unittest

from src import portfolio_guard as pg


def _pick(symbol="SPY", delta=0.5, vega=0.1, theta=-0.05, gamma=0.01):
    return {"symbol": symbol, "delta": delta, "vega": vega, "theta": theta, "gamma": gamma}


class ExposureTest(unittest.TestCase):
    def test_long_mode_sums_signed(self):
        exp = pg.compute_exposure([_pick(delta=0.5), _pick(delta=0.4)], mode="Discovery")
        self.assertEqual(exp["n"], 2)
        self.assertAlmostEqual(exp["net"]["delta"], (0.5 + 0.4) * 100, places=6)

    def test_premium_selling_flips_sign(self):
        exp = pg.compute_exposure([_pick(delta=0.5, vega=0.1)], mode="Premium Selling")
        self.assertAlmostEqual(exp["net"]["delta"], -0.5 * 100, places=6)
        self.assertAlmostEqual(exp["net"]["vega"], -0.1 * 100, places=6)   # short vol

    def test_skips_rows_without_delta(self):
        exp = pg.compute_exposure([_pick(), {"symbol": "X"}], mode="Discovery")
        self.assertEqual(exp["n"], 1)


class WarningsTest(unittest.TestCase):
    def test_directional_concentration_all_calls(self):
        picks = [_pick(symbol=s, delta=0.5) for s in ("SPY", "AAPL", "QQQ")]
        warns = pg.guard_warnings(pg.compute_exposure(picks, "Discovery"))
        self.assertTrue(any("Directional concentration" in w for w in warns))

    def test_vol_concentration_all_short_premium(self):
        # premium selling -> all short vol -> net vega == gross vega
        picks = [_pick(symbol=s, delta=0.2, vega=0.1) for s in ("SPY", "AAPL", "MSFT")]
        warns = pg.guard_warnings(pg.compute_exposure(picks, "Premium Selling"))
        self.assertTrue(any("Vol concentration" in w and "short-vol" in w for w in warns))

    def test_diversified_delta_no_directional_flag(self):
        # offsetting deltas (a put with negative delta) -> low net/gross
        picks = [_pick(symbol="A", delta=0.5, vega=0.1),
                 _pick(symbol="B", delta=-0.5, vega=-0.1)]
        warns = pg.guard_warnings(pg.compute_exposure(picks, "Discovery"))
        self.assertFalse(any("Directional concentration" in w for w in warns))

    def test_underlying_concentration(self):
        picks = [_pick(symbol="SPY", delta=0.5), _pick(symbol="SPY", delta=-0.5),
                 _pick(symbol="SPY", delta=0.5), _pick(symbol="AAPL", delta=0.1)]
        warns = pg.guard_warnings(pg.compute_exposure(picks, "Discovery"))
        self.assertTrue(any("Underlying concentration" in w and "SPY" in w for w in warns))

    def test_single_pick_no_warnings(self):
        self.assertEqual(pg.guard_warnings(pg.compute_exposure([_pick()], "Discovery")), [])


class FormatTest(unittest.TestCase):
    def test_lines_include_net_greeks(self):
        lines = pg.format_guard_lines([_pick(delta=0.5), _pick(delta=0.4)], "Discovery")
        self.assertTrue(lines)
        self.assertIn("Portfolio guard", lines[0])
        self.assertIn("net", lines[0])

    def test_empty_for_single_pick(self):
        self.assertEqual(pg.format_guard_lines([_pick()], "Discovery"), [])

    def test_clean_basket_gets_checkmark(self):
        picks = [_pick(symbol="A", delta=0.5, vega=0.1),
                 _pick(symbol="B", delta=-0.5, vega=-0.1)]
        lines = pg.format_guard_lines(picks, "Discovery")
        self.assertTrue(any("diversified" in l for l in lines))


if __name__ == "__main__":
    unittest.main()
