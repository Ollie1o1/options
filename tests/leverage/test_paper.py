import unittest
import os
import tempfile
import pandas as pd
from src.leverage.signals import Signal
from src.leverage.sizing import Sizing
from src.leverage.paper import PaperLedger


class TestPaperLedger(unittest.TestCase):
    def setUp(self):
        self.db = os.path.join(tempfile.mkdtemp(), "lev.db")
        self.ledger = PaperLedger(self.db)

    def _sig(self):
        return Signal("BTCUSDT", "long", pd.Timestamp("2026-05-01 13:35", tz="UTC"),
                      entry=100, atr=10, stop=88, target=122, trail_trigger=115,
                      session="us-open", confidence=0.5)

    def _sizing(self):
        return Sizing(0.02, 30.0, 4.0, 6000.0, 60.0)

    def test_open_and_list(self):
        tid = self.ledger.open_position(self._sig(), self._sizing(), liq_price=80.0)
        opens = self.ledger.open_positions()
        self.assertEqual(len(opens), 1)
        self.assertEqual(opens[0]["id"], tid)
        self.assertEqual(opens[0]["status"], "open")

    def test_close_records_realized_pnl(self):
        tid = self.ledger.open_position(self._sig(), self._sizing(), liq_price=80.0)
        self.ledger.close_position(tid, exit_price=122.0, reason="target")
        self.assertEqual(len(self.ledger.open_positions()), 0)
        closed = self.ledger.closed_positions()
        self.assertEqual(len(closed), 1)
        # long: (122-100)/100 = +22% per unit; pnl_usd = 0.22 * qty(60) = 13.2
        self.assertAlmostEqual(closed[0]["pnl_pct"], 0.22, places=4)
        self.assertGreater(closed[0]["pnl_usd"], 0)


if __name__ == "__main__":
    unittest.main()
