"""Tests for the delta-hedged short-straddle engine — deterministic tripwires."""
from __future__ import annotations
import unittest
from src.equity_vol.data import Entry
from src.equity_vol.engine import simulate_straddle
from src.equity_vol.costs import CostModel

NOCOST = CostModel(option_commission=0.0, hedge_slippage_bps=0.0)


def _entry(strike=100, exp="2022-02-01", iv=0.30, prem=9.0):
    return Entry(symbol="AAA", date="2022-01-02", expiration=exp, strike=strike,
                 spot=100.0, straddle_bid=prem, straddle_ask=prem + 0.8, iv=iv, dte=30)


def _flat_path(exp="2022-02-01"):
    # constant price == strike from entry to expiry -> short straddle keeps premium
    import datetime as dt
    d0 = dt.date(2022, 1, 2); d1 = dt.date(2022, 2, 1)
    out = {}
    d = d0
    while d <= d1:
        out[d.isoformat()] = 100.0
        d += dt.timedelta(days=1)
    return out


class EngineTests(unittest.TestCase):
    def test_flat_path_keeps_premium(self):
        r = simulate_straddle(_entry(prem=9.0), _flat_path(), cost=NOCOST)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.terminal_intrinsic, 0.0, places=6)
        self.assertAlmostEqual(r.pnl, 9.0, delta=0.2)   # ~all premium, tiny hedge drift
        self.assertGreater(r.ret, 0.0)

    def test_large_move_loses(self):
        path = _flat_path()
        # ramp the last day far through the strike -> big intrinsic, short loses
        path["2022-02-01"] = 130.0
        r = simulate_straddle(_entry(prem=9.0), path, cost=NOCOST)
        self.assertGreater(r.terminal_intrinsic, 20.0)
        self.assertLess(r.pnl, 0.0)

    def test_costs_reduce_pnl(self):
        flat = _flat_path()
        gross = simulate_straddle(_entry(), flat, cost=NOCOST).pnl
        net = simulate_straddle(_entry(), flat, cost=CostModel()).pnl
        self.assertLessEqual(net, gross)

    def test_no_lookahead_hedge_uses_only_past(self):
        # truncating the path AFTER expiry must not change the result
        full = _flat_path()
        full["2022-02-02"] = 999.0   # a date beyond expiry
        r_full = simulate_straddle(_entry(), full, cost=NOCOST)
        trimmed = {k: v for k, v in full.items() if k <= "2022-02-01"}
        r_trim = simulate_straddle(_entry(), trimmed, cost=NOCOST)
        self.assertAlmostEqual(r_full.pnl, r_trim.pnl, places=6)


if __name__ == "__main__":
    unittest.main()
