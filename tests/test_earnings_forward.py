"""E2 plumbing: the earnings rule pointed at the chain archive.

Synthetic databases only — no archive, no network, and no forward outcome is
computed from real data here.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/run_tests.py earnings_forward
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import pandas as pd

from scripts import earnings_premium_dolt as e


def _calendar(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE earnings_cal (symbol TEXT, date TEXT, whn TEXT)")
    con.executemany("INSERT INTO earnings_cal VALUES (?,?,?)", rows)
    con.commit()
    return con


def _archive(path, rows):
    con = sqlite3.connect(path)
    con.execute("CREATE TABLE chain_snapshots (symbol TEXT, snap_date TEXT, "
                "type TEXT, strike REAL, expiration TEXT, bid REAL, ask REAL, "
                "delta REAL)")
    con.executemany("INSERT INTO chain_snapshots VALUES (?,?,?,?,?,?,?,?)", rows)
    con.commit()
    return con


def _chain(sym, day, spot, exp="2026-11-20"):
    """Seven strikes a dollar apart with tight quotes and monotone deltas."""
    out = []
    for i, k in enumerate(range(int(spot) - 3, int(spot) + 4)):
        cd = round(0.95 - 0.15 * i, 2)                    # 0.95 .. 0.05
        c_mid = max(spot - k, 0) + 1.0 - 0.1 * abs(k - spot)
        p_mid = max(k - spot, 0) + 1.0 - 0.1 * abs(k - spot)
        out.append((sym, day, "call", float(k), exp, c_mid - 0.02, c_mid + 0.02, cd))
        out.append((sym, day, "put", float(k), exp, p_mid - 0.02, p_mid + 0.02,
                    round(cd - 1.0, 2)))
    return out


class EventWindowTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.cal = _calendar(os.path.join(self.dir.name, "cal.db"), [
            ("AAA", "2026-11-04", "After market close"),
            ("BBB", "2026-11-04", "Before market open"),
            ("CCC", "2026-09-15", "After market close"),     # before the window
            ("DDD", "2026-11-04", ""),                       # session unknown
        ])

    def tearDown(self):
        self.cal.close()
        self.dir.cleanup()

    def _snaps(self, days):
        return pd.DataFrame([(s, d) for s in ("AAA", "BBB", "CCC", "DDD")
                             for d in days], columns=["symbol", "date"])

    def test_after_close_enters_on_report_day_and_exits_the_day_after(self):
        ev = e.events(self.cal, self._snaps(["2026-11-03", "2026-11-04", "2026-11-05"]),
                      e.E2_FROM, e.E2_TO)
        a = ev[ev.symbol == "AAA"].iloc[0]
        self.assertEqual((a.pre, a.post), ("2026-11-04", "2026-11-05"))

    def test_before_open_enters_the_day_before_and_exits_on_report_day(self):
        ev = e.events(self.cal, self._snaps(["2026-11-03", "2026-11-04", "2026-11-05"]),
                      e.E2_FROM, e.E2_TO)
        b = ev[ev.symbol == "BBB"].iloc[0]
        self.assertEqual((b.pre, b.post), ("2026-11-03", "2026-11-04"))

    def test_events_outside_the_window_or_untimed_are_not_events(self):
        ev = e.events(self.cal, self._snaps(["2026-09-15", "2026-09-16",
                                             "2026-11-03", "2026-11-04", "2026-11-05"]),
                      e.E2_FROM, e.E2_TO)
        self.assertEqual(sorted(ev.symbol), ["AAA", "BBB"])

    def test_a_missing_archive_day_on_one_side_drops_the_event(self):
        # The archive skipped everything after the report: no exit, no trade.
        ev = e.events(self.cal, self._snaps(["2026-11-03", "2026-11-04"]),
                      e.E2_FROM, e.E2_TO)
        self.assertEqual(sorted(ev.symbol), ["BBB"])

    def test_no_events_is_an_empty_frame_with_its_columns(self):
        ev = e.events(self.cal, self._snaps(["2026-12-01"]), e.E2_FROM, e.E2_TO)
        self.assertEqual(len(ev), 0)
        self.assertIn("symbol", ev.columns)


class ArchiveChainTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        rows = _chain("AAA", "2026-11-04", 100.0) + _chain("AAA", "2026-11-05", 100.0)
        rows.append(("AAA", "2026-11-04", "CALL", 110.0, "2026-11-20", None, 0.05, 0.01))
        self.arc = _archive(os.path.join(self.dir.name, "arc.db"), rows)

    def tearDown(self):
        self.arc.close()
        self.dir.cleanup()

    def test_it_returns_the_columns_the_rule_reads_with_a_quote_midpoint(self):
        c = e.archive_chain(self.arc, "AAA", "2026-11-04")
        self.assertEqual(list(c.columns),
                         ["expiration", "strike", "type", "bid", "ask", "mid", "delta"])
        row = c[(c.strike == 100.0) & (c.type == "call")].iloc[0]
        self.assertAlmostEqual(row.mid, (row.bid + row.ask) / 2)

    def test_a_row_with_no_bid_is_not_a_quote(self):
        c = e.archive_chain(self.arc, "AAA", "2026-11-04")
        self.assertNotIn(110.0, set(c.strike))

    def test_the_rule_trades_an_archived_event_and_pays_the_crossing(self):
        ev = pd.DataFrame([{"symbol": "AAA", "event": "2026-11-04", "amc": True,
                            "D": "2026-11-05", "pre": "2026-11-04",
                            "post": "2026-11-05"}])
        touch = e.run(self.arc, ev, "fly", get_chain=e.archive_chain)
        mid = e.run(self.arc, ev, "fly", mid=True, get_chain=e.archive_chain)
        self.assertEqual((len(touch), len(mid)), (1, 1))
        # Identical quotes on both days: nothing is earned at mid, and the
        # touch fill loses exactly the four spreads crossed twice.
        self.assertAlmostEqual(mid.pnl.iloc[0], 0.0, places=6)
        self.assertAlmostEqual(touch.pnl.iloc[0], -100 * 4 * 0.04, places=6)

    def test_an_event_set_that_trades_nothing_still_returns_a_frame(self):
        ev = pd.DataFrame([{"symbol": "ZZZ", "event": "2026-11-04", "amc": True,
                            "D": "2026-11-05", "pre": "2026-11-04",
                            "post": "2026-11-05"}])
        self.assertEqual(len(e.run(self.arc, ev, "condor",
                                   get_chain=e.archive_chain)), 0)


if __name__ == "__main__":
    unittest.main()
