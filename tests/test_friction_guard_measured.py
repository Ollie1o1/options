"""The friction guard should measure the spread it is guarding against.

`_friction_to_credit_ratio` estimates round-trip cost as
`2 * slippage_per_share * n_legs`, with `slippage_per_share` configured at
$0.05. Measured against archived CBOE quotes for 30 logged Bull Puts, the real
ENTRY cost alone is $0.35/share — the flat estimate understates a two-leg
credit spread's round trip by roughly 3.5x, and it is the number deciding which
trades the auto-logger refuses as untradeable.

When the payload carries real leg quotes the guard should use them. When it
does not, the flat estimate stands — this changes nothing for callers that
cannot supply quotes.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager


class MeasuredFrictionTest(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "pm.db")
        self.pm = PaperManager(db_path=self.db)

    def _bull_put(self, **kw):
        d = {"strategy_name": "Bull Put", "ticker": "ORCL", "net_credit": 1.00}
        d.update(kw)
        return d

    def test_without_quotes_the_flat_estimate_is_unchanged(self):
        """2 legs * 2 sides * $0.05 = $0.20 of friction on $1.00 of credit."""
        r = self.pm._friction_to_credit_ratio(self._bull_put())
        self.assertAlmostEqual(r, 0.20, places=6)

    def test_with_quotes_the_real_half_spreads_are_used(self):
        """Legs quoted 1.40/1.60 and 0.40/0.60 carry $0.10 of half-spread each.
        Entry costs $0.20, the round trip $0.40 — double the flat estimate."""
        r = self.pm._friction_to_credit_ratio(self._bull_put(legs=[
            {"bid": 1.40, "ask": 1.60, "side": "sell"},
            {"bid": 0.40, "ask": 0.60, "side": "buy"},
        ]))
        self.assertAlmostEqual(r, 0.40, places=6)

    def test_a_wide_market_is_caught_that_the_flat_estimate_misses(self):
        """The case this exists for. A $1.00 credit quoted a dollar wide costs
        more to trade than it pays, but the flat estimate scores it at 20%."""
        wide = self._bull_put(legs=[
            {"bid": 1.00, "ask": 2.00, "side": "sell"},
            {"bid": 0.10, "ask": 1.10, "side": "buy"},
        ])
        self.assertAlmostEqual(self.pm._friction_to_credit_ratio(wide), 2.00, places=6)
        self.assertGreater(self.pm._friction_to_credit_ratio(wide),
                           self.pm._max_friction_to_credit)

    def test_a_partially_quoted_structure_falls_back_rather_than_half_counting(self):
        r = self.pm._friction_to_credit_ratio(self._bull_put(legs=[
            {"bid": 1.40, "ask": 1.60, "side": "sell"},
            {"bid": None, "ask": None, "side": "buy"},
        ]))
        self.assertAlmostEqual(r, 0.20, places=6)

    def test_debit_structures_are_still_not_judged(self):
        self.assertIsNone(self.pm._friction_to_credit_ratio(
            {"strategy_name": "Long Call", "ticker": "AAPL", "entry_price": 2.34}))

    def test_the_guard_refuses_a_trade_whose_measured_spread_eats_the_credit(self):
        ok = self.pm.log_trade_if_new({
            "date": "2026-08-04", "ticker": "ORCL", "expiration": "2026-09-18",
            "strike": 80.0, "type": "put", "entry_price": 1.00,
            "strategy_name": "Bull Put", "net_credit": 1.00, "long_strike": 79.0,
            "spread_width": 1.0, "max_loss_usd": 0.0,
            "legs": [{"bid": 1.00, "ask": 2.00, "side": "sell"},
                     {"bid": 0.10, "ask": 1.10, "side": "buy"}],
        }, auto_log=True)
        self.assertFalse(ok)
        self.assertEqual(self.pm.untradeable_rejected, 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
