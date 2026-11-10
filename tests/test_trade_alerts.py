"""Trade alerts: selection, de-duplication, cursor, and that nothing is sent
without a channel. Uses a temp candidates.db; never the real one."""
import datetime as dt
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src import trade_alerts as T

NOW = dt.datetime(2026, 11, 9, 20, 0, tzinfo=dt.timezone.utc)


def _db(path: Path, rows):
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE candidates (scan_id TEXT, ts TEXT, board TEXT,
        contract_key TEXT, symbol TEXT, strategy_name TEXT, expiration TEXT,
        premium REAL, ev_net REAL, round_trip_pct REAL, gate_passed INTEGER,
        features_json TEXT)""")
    for r in rows:
        con.execute("INSERT INTO candidates VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", (
            "s", r.get("ts", "2026-11-09T19:49:01+00:00"), "TOP PICK", r["key"], r["sym"],
            r.get("strat", "Bear Call"), r.get("exp", "2026-11-27"), 3.2, r.get("ev", 10.0),
            None, r.get("gp", 1), json.dumps({"short_strike": 100, "long_strike": 105,
                                               "short_bid": 2.0, "short_ask": 2.1,
                                               "long_bid": 0.5, "long_ask": 0.6,
                                               "spread_width": 5.0})))
    con.commit()
    con.close()


class TradeAlertsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        d = Path(self.tmp.name)
        self.db, self.state = d / "c.db", d / "state.json"
        self.s = dict(T.DEFAULTS)
        self.sent = []

    def tearDown(self):
        self.tmp.cleanup()

    def _run(self, **kw):
        T.SENDERS["fake"] = lambda cfg, text: self.sent.append(text)
        try:
            return T.run(self.db, self.state, now=NOW, settings=self.s,
                         chans={"fake": {}}, **kw)
        finally:
            T.SENDERS.pop("fake")

    def test_filters_and_caps(self):
        _db(self.db, [
            {"key": "SPY|a", "sym": "SPY", "ev": 5},
            {"key": "SPY|b", "sym": "SPY", "ev": 9},          # same symbol: best EV wins
            {"key": "QQQ|a", "sym": "QQQ", "ev": 1},
            {"key": "AAPL|a", "sym": "AAPL", "ev": 99},       # not an ETF
            {"key": "IWM|a", "sym": "IWM", "strat": "Bull Put"},  # strategy off
            {"key": "DIA|a", "sym": "DIA", "gp": 0},          # refused by the gate
            {"key": "XLF|a", "sym": "XLF", "exp": "2026-11-19"},  # 10 DTE < 14 min
            {"key": "XLE|a", "sym": "XLE", "ts": "2026-11-09T10:00:00+00:00"},  # stale
            {"key": "XLK|a", "sym": "XLK", "exp": "2027-01-29"},  # 81 DTE > 60 max
        ])
        self.assertEqual(self._run(), 2)
        self.assertIn("SPY Bear Call 100/105C", self.sent[0])
        self.assertIn("QQQ", self.sent[1])
        self.assertIn("max loss $", self.sent[0])

    def test_dedupe_and_cursor(self):
        _db(self.db, [{"key": "SPY|a", "sym": "SPY"}])
        self.assertEqual(self._run(), 1)
        self.assertEqual(self._run(), 0)                      # cursor moved
        st = json.loads(self.state.read_text())
        st["cursor"] = None
        self.state.write_text(json.dumps(st))
        self.assertEqual(self._run(), 0)                      # still de-duplicated

    def test_no_channel_sends_nothing_and_keeps_cursor(self):
        _db(self.db, [{"key": "SPY|a", "sym": "SPY"}])
        n = T.run(self.db, self.state, now=NOW, settings=self.s, chans={})
        self.assertEqual(n, 0)
        self.assertFalse(self.state.exists())

    def test_failed_send_is_retried_next_run(self):
        _db(self.db, [{"key": "SPY|a", "sym": "SPY"}])

        def boom(cfg, text):
            raise RuntimeError("HTTP 500")
        T.SENDERS["fake"] = boom
        try:
            T.run(self.db, self.state, now=NOW, settings=self.s, chans={"fake": {}})
        finally:
            T.SENDERS.pop("fake")
        st = json.loads(self.state.read_text())
        self.assertNotIn("SPY|a", st["sent"])

    def test_dry_run_writes_no_state(self):
        _db(self.db, [{"key": "SPY|a", "sym": "SPY"}])
        T.run(self.db, self.state, now=NOW, settings=self.s, chans={}, dry_run=True)
        self.assertFalse(self.state.exists())


if __name__ == "__main__":
    unittest.main()
