"""The auto-log tradeability guard.

A credit spread whose round-trip friction exceeds the credit received cannot
profit at any win rate. 31 of 188 logged short-premium trades were in that
state. This guard stops the feeder adding more, and its job is to do that
WITHOUT touching debit structures, manual entries, or trades that are merely
small.
"""
import datetime as _dt
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import (DEFAULT_MAX_FRICTION_TO_CREDIT,  # noqa: E402
                               PaperManager)

# 30 days out, relative to today. A hardcoded expiration drifts out of the cost
# model's calibrated DTE range — and eventually into the past — neither of
# which has anything to do with what these tests measure.
_NEAR_EXP = (_dt.date.today() + _dt.timedelta(days=30)).isoformat()


class _Mgr(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.mgr = PaperManager(db_path=os.path.join(self.tmp.name, "t.db"))
        # Pin costs so the ratios below are arithmetic, not config-dependent.
        self.mgr._slippage_per_share = 0.05
        self.mgr._commission_per_contract = 0.0
        self.mgr._fx_conversion_rate = 0.0
        self.mgr._max_friction_to_credit = DEFAULT_MAX_FRICTION_TO_CREDIT

    def tearDown(self):
        self.tmp.cleanup()

    def ratio(self, **kw):
        base = {"strategy_name": "Bull Put", "net_credit": 1.00}
        base.update(kw)
        return self.mgr._friction_to_credit_ratio(base)


class RatioTest(_Mgr):
    def test_a_wide_credit_is_cheap_to_trade(self):
        # 2 legs x 2 sides x $0.05 = $0.20 against $2.00 of credit
        self.assertAlmostEqual(self.ratio(net_credit=2.00), 0.10)

    def test_a_micro_credit_is_swallowed_by_the_spread(self):
        self.assertGreater(self.ratio(net_credit=0.24), 0.5)

    def test_a_four_legged_condor_costs_twice_a_vertical(self):
        self.assertAlmostEqual(self.ratio(strategy_name="Iron Condor", net_credit=1.0),
                               2 * self.ratio(strategy_name="Bull Put", net_credit=1.0))

    def test_a_single_leg_short_costs_half(self):
        self.assertAlmostEqual(self.ratio(strategy_name="Short Put", net_credit=1.0),
                               self.ratio(strategy_name="Bull Put", net_credit=1.0) / 2)

    def test_debit_structures_are_not_judged(self):
        # The guard is about credit received; a long call has none.
        self.assertIsNone(self.ratio(strategy_name="Long Call", entry_price=3.0))
        self.assertIsNone(self.ratio(strategy_name="Long Put", entry_price=3.0))

    def test_a_missing_credit_is_not_a_free_trade(self):
        self.assertIsNone(self.ratio(net_credit=None, entry_price=None))
        self.assertIsNone(self.ratio(net_credit=0))

    def test_entry_price_stands_in_when_net_credit_is_absent(self):
        self.assertAlmostEqual(self.ratio(net_credit=None, entry_price=2.00), 0.10)


class GuardTest(_Mgr):
    def _log(self, **kw):
        trade = {"ticker": "TSTX", "strategy_name": "Bull Put", "type": "put",
                 "strike": 100.0, "expiration": _NEAR_EXP, "entry_price": 1.00,
                 "net_credit": 1.00, "max_loss_usd": 400.0, "quantity": 1.0,
                 "quality_score": 70.0}
        trade.update(kw)
        return self.mgr.log_trade(trade)

    def test_a_healthy_credit_spread_still_logs(self):
        self.assertTrue(self._log(net_credit=2.00, entry_price=2.00))
        self.assertEqual(self.mgr.untradeable_rejected, 0)

    def test_a_spread_the_market_would_eat_is_refused(self):
        self.assertFalse(self._log(net_credit=0.15, entry_price=0.15))
        self.assertEqual(self.mgr.untradeable_rejected, 1)

    def test_the_refusal_is_counted_separately_from_the_other_gates(self):
        # A quiet feeder must be able to say WHICH gate held it back.
        self._log(net_credit=0.15, entry_price=0.15)
        self.assertEqual(self.mgr.untradeable_rejected, 1)
        self.assertEqual(self.mgr.unaffordable_rejected, 0)
        self.assertEqual(self.mgr.duplicate_rejected, 0)

    def test_a_deliberate_manual_entry_can_bypass_it(self):
        self.assertTrue(self._log(net_credit=0.15, entry_price=0.15,
                                  allow_untradeable=True))

    def test_disabling_the_threshold_restores_old_behaviour(self):
        self.mgr._max_friction_to_credit = None
        self.assertTrue(self._log(net_credit=0.15, entry_price=0.15))

    def test_a_long_call_is_never_refused_by_this_guard(self):
        # Debit structures have no credit to compare against; the guard must
        # not accidentally gate the strategy the Phase-1 cohort is built from.
        self.assertTrue(self._log(strategy_name="Long Call", type="call",
                                  net_credit=None, entry_price=0.10))
        self.assertEqual(self.mgr.untradeable_rejected, 0)


if __name__ == "__main__":
    unittest.main()
