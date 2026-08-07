"""Edge means nothing if the capital could not have been deployed into it.

An earlier index put spread here showed profit factor 4.29 and, sized
responsibly, produced ~0.3% CAGR on 31 trades in three years. Capacity is
reported beside edge so that cannot happen quietly again.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_alloc_portfolio -v
"""
from __future__ import annotations

import unittest

from src.alloc.engine import Trade
from src.alloc.fills import Leg
from src.alloc.portfolio import apply_capacity, capacity_stats

LEGS = [Leg("2024-03-15", 100.0, "put", "sell"),
        Leg("2024-03-15", 95.0, "put", "buy")]


def _t(entry, exit_, pnl, car=400.0, sym="AAA"):
    return Trade(symbol=sym, entry_date=entry, entry_price=1.0,
                 capital_at_risk=car, legs=LEGS, expiration="2024-03-15",
                 exit_date=exit_, exit_price=-0.5, pnl=pnl,
                 exit_reason="expiry")


class ConcurrencyTest(unittest.TestCase):
    def test_overlapping_trades_beyond_the_cap_are_dropped(self):
        trades = [_t("2024-01-05", "2024-03-01", 50) for _ in range(10)]
        kept, stats = apply_capacity(trades, max_concurrent=3,
                                     max_capital=1_000_000)
        self.assertEqual(len(kept), 3)
        self.assertEqual(stats["dropped_concurrency"], 7)

    def test_sequential_trades_do_not_compete(self):
        trades = [_t("2024-01-05", "2024-01-10", 50),
                  _t("2024-02-05", "2024-02-10", 50)]
        kept, _ = apply_capacity(trades, max_concurrent=1,
                                 max_capital=1_000_000)
        self.assertEqual(len(kept), 2)

    def test_a_position_frees_its_slot_on_the_exit_date(self):
        trades = [_t("2024-01-05", "2024-01-10", 50),
                  _t("2024-01-10", "2024-01-20", 50)]
        kept, _ = apply_capacity(trades, max_concurrent=1,
                                 max_capital=1_000_000)
        self.assertEqual(len(kept), 2)


class CapitalTest(unittest.TestCase):
    def test_capital_cap_drops_the_excess(self):
        trades = [_t("2024-01-05", "2024-03-01", 50, car=400.0)
                  for _ in range(10)]
        kept, stats = apply_capacity(trades, max_concurrent=100,
                                     max_capital=1000)
        self.assertEqual(len(kept), 2)          # 2 x 400 fits under 1000
        self.assertEqual(stats["dropped_capital"], 8)

    def test_a_single_oversized_trade_is_rejected(self):
        kept, stats = apply_capacity([_t("2024-01-05", "2024-02-01", 0,
                                         car=9999.0)],
                                     max_concurrent=5, max_capital=4000)
        self.assertEqual(kept, [])
        self.assertEqual(stats["dropped_capital"], 1)

    def test_kept_count_is_reported(self):
        _, stats = apply_capacity([_t("2024-01-05", "2024-02-01", 10)],
                                  max_concurrent=5, max_capital=4000)
        self.assertEqual(stats["kept"], 1)

    def test_no_trades_is_not_an_error(self):
        kept, stats = apply_capacity([], 5, 4000)
        self.assertEqual(kept, [])
        self.assertEqual(stats["kept"], 0)


class CapacityStatsTest(unittest.TestCase):
    def test_trades_per_year(self):
        trades = [_t("2024-01-01", "2024-01-10", 50),
                  _t("2024-07-01", "2024-07-10", 50),
                  _t("2024-12-20", "2024-12-31", 50)]
        s = capacity_stats(trades, max_capital=4000)
        self.assertAlmostEqual(s["trades_per_year"], 3.0, places=0)

    def test_peak_deployment_is_the_simultaneous_maximum(self):
        trades = [_t("2024-01-05", "2024-03-01", 0, car=400.0),
                  _t("2024-01-06", "2024-03-01", 0, car=400.0),
                  _t("2024-06-01", "2024-07-01", 0, car=400.0)]
        s = capacity_stats(trades, max_capital=4000)
        self.assertAlmostEqual(s["peak_deployed"], 800.0)
        self.assertEqual(s["max_concurrent"], 2)

    def test_return_is_measured_against_the_account_not_the_peak(self):
        """Idle capital is a real cost. A great trade you make twice a year is
        not a great return on the account that had to sit there waiting."""
        trades = [_t("2024-01-01", "2024-12-31", 400.0, car=400.0)]
        s = capacity_stats(trades, max_capital=4000)
        self.assertAlmostEqual(s["return_on_cap"], 0.10, places=2)

    def test_a_strong_per_trade_edge_with_no_capacity_is_a_small_return(self):
        """The capacity wall, made visible.

        Two trades a year, each returning a healthy 12.5% on the $400 actually
        committed, still yields ~2% a year on the $4,000 account — because the
        other $3,600 sat idle. This is the shape that a profit factor of 4.29
        hid behind ~0.3% CAGR in the earlier index research.
        """
        trades = [_t("2024-01-01", "2024-02-01", 50.0, car=400.0),
                  _t("2025-01-01", "2025-02-01", 50.0, car=400.0)]
        s = capacity_stats(trades, max_capital=4000)
        per_trade_return = 50.0 / 400.0
        self.assertGreater(per_trade_return, 0.12)      # excellent per trade
        self.assertLess(s["return_on_cap"], 0.03)       # negligible on account
        self.assertLess(s["trades_per_year"], 3)

    def test_no_closed_trades_returns_zeros(self):
        self.assertEqual(capacity_stats([], 4000)["trades_per_year"], 0.0)

    def test_open_trades_are_ignored(self):
        open_trade = _t("2024-01-05", None, None)
        self.assertEqual(capacity_stats([open_trade], 4000)["trades_per_year"],
                         0.0)


if __name__ == "__main__":
    unittest.main()
