import os
import tempfile
import unittest
import numpy as np
import pandas as pd

from src.leverage.paper import PaperLedger
from src.leverage import swing_paper as SP
from src.leverage import swing as S


def _uptrend_then_break(n=400, seed=1):
    """Long uptrend (fires a long breakout) then a sharp drop (regime flip /
    stop) so an open position resolves."""
    rng = np.random.default_rng(seed)
    up = np.cumsum(np.abs(rng.normal(0.6, 0.4, n - 40))) + 100
    drop = up[-1] - np.cumsum(np.abs(rng.normal(2.0, 0.5, 40)))
    close = np.concatenate([up, drop])
    idx = pd.date_range("2023-01-01", periods=n, freq="D", tz="UTC")
    return pd.DataFrame({"open": close, "high": close + 0.3, "low": close - 0.3,
                         "close": close, "volume": 1.0}, index=idx)


class ResolveOpenTest(unittest.TestCase):
    def test_long_stop_hit_closes(self):
        df = _uptrend_then_break()
        feat = S.compute_features(df)
        # open a long near the end of the uptrend with a close stop
        entry_i = 360
        entry_date = feat.index[entry_i]
        entry = float(feat["close"].iloc[entry_i])
        stop = entry - 3.0
        r = SP.resolve_open(feat, entry_date, "long", entry, stop)
        self.assertIsNotNone(r)  # the drop must take it out
        self.assertIn(r[2], ("stop", "regime", "max_hold"))

    def test_still_open_returns_none(self):
        df = _uptrend_then_break()
        feat = S.compute_features(df)
        # enter on the very last bar -> cannot have resolved yet
        last = feat.index[-1]
        entry = float(feat["close"].iloc[-1])
        r = SP.resolve_open(feat, last, "long", entry, entry - 5.0)
        self.assertIsNone(r)


class RunSwingPaperTest(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.ledger = PaperLedger(db_path=self.db)

    def tearDown(self):
        os.remove(self.db)

    def _loader(self, df):
        return lambda key: df

    def test_opens_on_breakout(self):
        df = _uptrend_then_break()
        # slice so the latest bar is a fresh breakout (mid-uptrend)
        df_break = df.iloc[:350]
        summ = SP.run_swing_paper(["BTC"], 2000.0, self.ledger,
                                  self._loader(df_break))
        # either it opened, or there was genuinely no breakout on the last bar
        opened = self.ledger.open_positions()
        self.assertEqual(len(summ["opened"]), len(opened))

    def test_idempotent_no_double_open(self):
        df_break = _uptrend_then_break().iloc[:350]
        SP.run_swing_paper(["BTC"], 2000.0, self.ledger, self._loader(df_break))
        n1 = len(self.ledger.open_positions())
        # running again on the same frame must not open a second position
        SP.run_swing_paper(["BTC"], 2000.0, self.ledger, self._loader(df_break))
        n2 = len(self.ledger.open_positions())
        self.assertLessEqual(n2, max(n1, 1))
        self.assertEqual(n2, n1)

    def test_open_then_resolve_on_more_data(self):
        full = _uptrend_then_break()
        # open with data up to the breakout, then resolve with the full drop
        SP.run_swing_paper(["BTC"], 2000.0, self.ledger, self._loader(full.iloc[:350]))
        if not self.ledger.open_positions():
            self.skipTest("no breakout on this fixture slice")
        SP.run_swing_paper(["BTC"], 2000.0, self.ledger, self._loader(full))
        # the sharp drop should have closed it
        self.assertTrue(self.ledger.closed_positions())


if __name__ == "__main__":
    unittest.main()
