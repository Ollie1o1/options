import unittest
import pandas as pd
from src.leverage.signals import Signal
from src.leverage.sizing import Sizing
from src.leverage.ticket import render


class TestTicket(unittest.TestCase):
    def _sig(self):
        return Signal("BTCUSDT", "long", pd.Timestamp("2026-05-01 13:35", tz="UTC"),
                      entry=67420, atr=337, stop=67016, target=68161,
                      trail_trigger=67925, session="us-open", confidence=0.71)

    def _sizing(self):
        return Sizing(risk_frac=0.02, risk_usd=30.0, eff_leverage=4.0,
                      notional=6000.0, qty=0.089)

    def test_ticket_contains_key_fields(self):
        t = render(self._sig(), self._sizing(), liq_price=54100.0, safe=True)
        self.assertIn("BTC", t)
        self.assertIn("LONG", t)
        self.assertIn("67,420", t)
        self.assertIn("stop", t.lower())
        self.assertIn("liq", t.lower())
        self.assertIn("4.0x", t)
        self.assertIn("SAFE", t)

    def test_reject_flag_when_unsafe(self):
        t = render(self._sig(), self._sizing(), liq_price=66000.0, safe=False)
        self.assertIn("REJECT", t)

    def test_shows_reward_risk_ratio(self):
        # stop -0.60%, target +1.10% -> R:R ~ 1.83 : 1
        t = render(self._sig(), self._sizing(), liq_price=54100.0, safe=True)
        self.assertIn("R:R", t)
        self.assertIn("1.8", t)

    def test_shows_leverage_band_status(self):
        t = render(self._sig(), self._sizing(), liq_price=54100.0, safe=True)
        self.assertIn("band", t.lower())


if __name__ == "__main__":
    unittest.main()
