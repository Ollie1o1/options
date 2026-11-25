"""Portfolio fit on the Evidence Card: concentration and co-movement against real-money holdings."""
from __future__ import annotations

import os
import tempfile
import unittest

import numpy as np

from src.topdown.portfolio_fit import Holding, load_holdings, portfolio_fit
from src.topdown.synthetic import business_days

SECTOR = {"AAA": "XLK", "BBB": "XLK", "CCC": "XLK", "DDD": "XLE", "XLK": None, "XLE": None, "GLD": None, "TLT": None}


def _ohlcv(n: int = 200):
    rng = np.random.default_rng(4)
    days = business_days(n)
    f = rng.normal(0, 0.01, n)
    g = rng.normal(0, 0.01, n)
    vol = np.full(n, 1e6)

    def px(r):
        return 100.0 * np.exp(np.cumsum(r))

    return {
        "AAA": (days, px(f + rng.normal(0, 0.002, n)), vol),
        "BBB": (days, px(f + rng.normal(0, 0.002, n)), vol),   # ~ AAA
        "CCC": (days, px(f + rng.normal(0, 0.004, n)), vol),
        "DDD": (days, px(g), vol),                              # independent
        "XLK": (days, px(f), vol),
        "XLE": (days, px(g + rng.normal(0, 0.002, n)), vol),
        "GLD": (days, px(-0.3 * f + rng.normal(0, 0.01, n)), vol),
        "TLT": (days, px(rng.normal(0, 0.01, n)), vol),
    }


class PortfolioFitTests(unittest.TestCase):
    def setUp(self) -> None:
        self.ohlcv = _ohlcv()

    def test_no_holdings_says_so(self) -> None:
        fit = portfolio_fit("AAA", holdings=[], ohlcv=self.ohlcv, sector_of=SECTOR.get)
        self.assertEqual(fit.warnings, [])
        self.assertIn("No holdings recorded", fit.notes[0])

    def test_same_sector_and_high_correlation_warn(self) -> None:
        hold = [Holding("BBB", "own picks", shares=10), Holding("CCC", "long-term plan", shares=5)]
        fit = portfolio_fit("AAA", holdings=hold, ohlcv=self.ohlcv, sector_of=SECTOR.get,
                            universe=("XLK", "XLE", "GLD", "TLT"))
        self.assertTrue(any("2 positions in XLK" in w for w in fit.warnings))
        self.assertTrue(any("BBB (rho" in w for w in fit.warnings))
        self.assertEqual(fit.correlations[0][0], "BBB")
        self.assertGreater(fit.correlations[0][1], 0.9)
        self.assertEqual(fit.diversifiers[0][0], "GLD")        # negatively related to the XLK factor
        self.assertNotIn("XLK", [d for d, _ in fit.diversifiers if d == "AAA"])

    def test_unrelated_holding_does_not_warn_and_sleeve_overlap_is_noted(self) -> None:
        hold = [Holding("DDD", "own picks", shares=3), Holding("XLK", "core sleeve (target)", weight=0.262)]
        fit = portfolio_fit("AAA", holdings=hold, ohlcv=self.ohlcv, sector_of=SECTOR.get)
        self.assertFalse(any("positions in" in w for w in fit.warnings))   # 1 same-sector (the sleeve ETF)
        self.assertEqual(fit.sleeve_overlap, ("XLK", 0.262))
        self.assertTrue(any("core sleeve holds XLK at 26%" in n for n in fit.notes))
        self.assertTrue(any("XLK (rho" in w for w in fit.warnings))        # AAA tracks its sector ETF

    def test_load_holdings_reads_open_long_picks_and_sleeve(self) -> None:
        from src.own_picks import log_equity_pick

        class _Row:
            def __init__(self, etf: str, w: float) -> None:
                self.etf, self.weight = etf, w

        class _Board:
            rows = [_Row("XLE", 0.262), _Row("XLK", 0.0)]

        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "picks.db")
            log_equity_pick("aaa", 100.0, shares=4, playbook_id="discretionary", db_path=db)
            log_equity_pick("AAA", 101.0, shares=2, playbook_id="discretionary", db_path=db)
            hs = load_holdings(own_picks_db=db, longterm_db=os.path.join(td, "none.db"), board=_Board())
        self.assertIn(Holding("AAA", "own picks", shares=6.0), hs)
        self.assertIn(Holding("XLE", "core sleeve (target)", weight=0.262), hs)
        self.assertEqual(len(hs), 2)


if __name__ == "__main__":
    unittest.main()
