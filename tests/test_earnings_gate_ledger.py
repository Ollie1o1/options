"""The earnings gate at the chokepoint: what `log_trade` does with a verdict.

Placement matters as much as the rule. This gate sits AFTER budget and
tradeability and BEFORE sizing, so a candidate that fails a cheaper test is
still refused for that reason, and a candidate refused here is refused for the
event rather than for its size. Refusal reasons in this ledger are diagnostic —
that is how a quiet window gets explained.

Every test builds its own temp ledger, temp config and temp earnings cache.
Nothing here touches the real book or the real calendar.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager

# 30 DTE, inside the 10-67 cost-calibration window so the tradeability gate
# that runs BEFORE this one cannot be what refuses these trades.
_ENTRY = datetime.now()
_EXPIRY_D = (_ENTRY + timedelta(days=30)).date()
_EXPIRY = _EXPIRY_D.isoformat()
#: An event a week after entry — inside the holding period on any horizon.
_DURING = (_ENTRY + timedelta(days=7)).date().isoformat()
#: An event well after the contract has expired.
_AFTER = (_EXPIRY_D + timedelta(days=40)).isoformat()
#: A past event, so the cache demonstrably reaches this trade.
_BEFORE = (_ENTRY - timedelta(days=95)).date().isoformat()


def _write_config(path, cache_path, **over):
    auto_log = {
        "max_capital_at_risk": None,
        "max_friction_to_credit": None,
        "refuse_through_earnings": True,
        "earnings_horizon": "expiration",
        "earnings_cache_path": cache_path,
    }
    auto_log.update(over)
    with open(path, "w") as f:
        json.dump({
            "exit_rules": {"take_profit": 0.5, "stop_loss": -0.25,
                           "time_exit_dte": 21},
            "paper_trading": {"commission_per_contract": 0.0,
                              "slippage_per_share": 0.0},
            "auto_log": auto_log,
            # Sizing off: it runs after this gate and is covered elsewhere.
            "position_sizing": {"enabled": False},
        }, f)
    return path


def _seed_cache(path, rows):
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS earnings_cal (symbol TEXT, "
                     "date TEXT, whn TEXT, PRIMARY KEY (symbol, date))")
        conn.executemany("INSERT OR REPLACE INTO earnings_cal VALUES (?,?,?)",
                         rows)


def _bull_put(**over):
    trade = {"ticker": "WMT", "expiration": _EXPIRY, "strike": 110.0,
             "type": "put", "entry_price": 1.08, "quality_score": 0.75,
             "strategy_name": "Bull Put", "long_strike": 105.0,
             "spread_width": 5.0, "net_credit": 1.08, "max_loss_usd": 392.0}
    trade.update(over)
    return trade


class EarningsRefusal(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "book.db")
        self.cache = os.path.join(self.dir.name, "earnings.db")
        self.cfg = _write_config(os.path.join(self.dir.name, "config.json"),
                                 self.cache)

    def tearDown(self):
        self.dir.cleanup()

    def _pm(self, **over):
        if over:
            _write_config(self.cfg, self.cache, **over)
        return PaperManager(db_path=self.db, config_path=self.cfg)

    def _rows(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM trades")]
        finally:
            conn.close()

    def test_a_spread_held_through_earnings_is_not_logged(self):
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        pm = self._pm()
        self.assertFalse(pm.log_trade(_bull_put()))
        self.assertEqual(self._rows(), [])
        self.assertEqual(pm.through_earnings_rejected, 1)

    def test_a_spread_clear_of_earnings_is_logged(self):
        # The cache reaches past this trade — a past event and a future one
        # outside the contract — so its silence about the window is real.
        _seed_cache(self.cache, [("WMT", _BEFORE, "amc"), ("WMT", _AFTER, "amc")])
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(len(self._rows()), 1)
        self.assertEqual(pm.through_earnings_rejected, 0)
        self.assertEqual(pm.earnings_unknown, 0)

    def test_an_uncovered_symbol_is_logged_but_counted_as_unknown(self):
        # 72% of the book is in this state. Refusing on no-data would stop the
        # feeder over a data gap; treating it as clear would make the gate
        # silently inert. It logs, and says so.
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(pm.earnings_unknown, 1)
        self.assertEqual(pm.through_earnings_rejected, 0)

    def test_a_stale_cache_counts_as_unknown_not_clear(self):
        # Every cached date predates the entry: the cache has stopped being
        # updated and knows nothing about this holding period.
        _seed_cache(self.cache, [("WMT", _BEFORE, "amc")])
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(pm.earnings_unknown, 1)

    def test_long_premium_is_not_gated(self):
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put(
            strategy_name="Long Call", type="call", net_credit=None,
            spread_width=None, long_strike=None, max_loss_usd=None)))
        self.assertEqual(pm.through_earnings_rejected, 0)
        # ...and it is not counted as unknown either: the gate did not apply.
        self.assertEqual(pm.earnings_unknown, 0)

    def test_allow_through_earnings_is_the_escape_hatch(self):
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put(allow_through_earnings=True)))
        self.assertEqual(len(self._rows()), 1)
        self.assertEqual(pm.through_earnings_rejected, 0)

    def test_the_gate_off_logs_the_trade(self):
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        pm = self._pm(refuse_through_earnings=False)
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(pm.through_earnings_rejected, 0)

    def test_the_time_exit_horizon_clears_an_event_after_the_forced_close(self):
        # 30 DTE with a 21-day time exit: the position is closed ~9 days in, so
        # an event at day 20 is never held through.
        late = (_ENTRY + timedelta(days=20)).date().isoformat()
        _seed_cache(self.cache, [("WMT", _BEFORE, "amc"), ("WMT", late, "amc")])
        pm = self._pm(earnings_horizon="time_exit")
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(pm.through_earnings_rejected, 0)

    def test_the_expiration_horizon_refuses_that_same_trade(self):
        # The knob is real: identical inputs, opposite outcomes.
        late = (_ENTRY + timedelta(days=20)).date().isoformat()
        _seed_cache(self.cache, [("WMT", _BEFORE, "amc"), ("WMT", late, "amc")])
        pm = self._pm(earnings_horizon="expiration")
        self.assertFalse(pm.log_trade(_bull_put()))
        self.assertEqual(pm.through_earnings_rejected, 1)

    def test_a_cheaper_gate_still_refuses_first(self):
        # Both untradeable and through earnings: the refusal must name the
        # friction, which is the cheaper and more fundamental problem.
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        # Real slippage, so the friction ratio is non-zero and a 1% ceiling
        # bites; the rest of this file runs frictionless on purpose.
        with open(self.cfg) as f:
            cfg = json.load(f)
        cfg["paper_trading"]["slippage_per_share"] = 0.05
        cfg["auto_log"]["max_friction_to_credit"] = 0.01
        with open(self.cfg, "w") as f:
            json.dump(cfg, f)
        pm = PaperManager(db_path=self.db, config_path=self.cfg)
        self.assertFalse(pm.log_trade(_bull_put()))
        self.assertEqual(pm.untradeable_rejected, 1)
        self.assertEqual(pm.through_earnings_rejected, 0)

    def test_the_refusal_names_the_date_it_refused_on(self):
        # A quiet feeder has to be explainable without a debugger.
        import contextlib
        import io
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._pm().log_trade(_bull_put())
        out = buf.getvalue()
        self.assertIn(_DURING, out)
        self.assertIn("WMT", out)

    def test_a_spread_clear_of_earnings_records_its_state(self):
        # The verdict is computed and then thrown away today — nothing
        # persists it, so a future test of "does clear beat unknown" has no
        # data to run on. This records it.
        _seed_cache(self.cache, [("WMT", _BEFORE, "amc"), ("WMT", _AFTER, "amc")])
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(self._rows()[0]["earnings_state"], "clear_of_earnings")

    def test_an_uncovered_symbol_records_unknown_state(self):
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertEqual(self._rows()[0]["earnings_state"], "earnings_unknown")

    def test_allow_through_earnings_records_that_the_check_was_bypassed(self):
        # The escape hatch skips verdict_for_trade entirely — this must not
        # read as "clear", or a bypassed trade would look like evidence for
        # the gate.
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put(allow_through_earnings=True)))
        self.assertIsNone(self._rows()[0]["earnings_state"])

    def test_the_gate_off_records_no_state(self):
        # refuse_through_earnings is the gate's enable switch (cfg["enabled"]),
        # not just a refuse/report toggle — off means verdict_for_trade never
        # runs at all, so there is nothing honest to record but None.
        _seed_cache(self.cache, [("WMT", _DURING, "amc")])
        pm = self._pm(refuse_through_earnings=False)
        self.assertTrue(pm.log_trade(_bull_put()))
        self.assertIsNone(self._rows()[0]["earnings_state"])


class ProjectedEarnings(unittest.TestCase):
    """A projected report counts, and only refuses when told to.

    The estimate is trustworthy to about a day for regular reporters, but it is
    still an estimate — so it ships counting and printing, and the operator
    flips it to refusing once they have watched it on the live board.
    """

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "book.db")
        self.cache = os.path.join(self.dir.name, "earnings.db")
        self.cfg = _write_config(os.path.join(self.dir.name, "config.json"),
                                 self.cache, earnings_projection="report")
        # Nine quarterly reports, the most recent 76 days ago: a demonstrated
        # cadence with no announced future date — the 93% case. 76 + 91 puts
        # the projection ~15 days out, inside this trade's 30-day window, and
        # 76 days is inside the 120-day staleness guard.
        anchor = (_ENTRY - timedelta(days=76)).date()
        _seed_cache(self.cache, [
            ("AAA", (anchor - timedelta(days=91 * i)).isoformat(), "amc")
            for i in range(9)])

    def tearDown(self):
        self.dir.cleanup()

    def _pm(self, **over):
        if over:
            _write_config(self.cfg, self.cache, **over)
        return PaperManager(db_path=self.db, config_path=self.cfg)

    def _count(self):
        conn = sqlite3.connect(self.db)
        try:
            return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
        finally:
            conn.close()

    def _rows(self):
        conn = sqlite3.connect(self.db)
        conn.row_factory = sqlite3.Row
        try:
            return [dict(r) for r in conn.execute("SELECT * FROM trades")]
        finally:
            conn.close()

    def test_report_mode_flags_it_and_still_logs(self):
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put(ticker="AAA")))
        self.assertEqual(self._count(), 1)
        self.assertEqual(pm.projected_earnings_flagged, 1)
        self.assertEqual(pm.through_earnings_rejected, 0)

    def test_refuse_mode_turns_it_away(self):
        pm = self._pm(earnings_projection="refuse")
        self.assertFalse(pm.log_trade(_bull_put(ticker="AAA")))
        self.assertEqual(self._count(), 0)
        self.assertEqual(pm.through_earnings_rejected, 1)
        self.assertEqual(pm.projected_earnings_flagged, 1)

    def test_off_means_the_symbol_is_simply_unknown(self):
        pm = self._pm(earnings_projection="off")
        self.assertTrue(pm.log_trade(_bull_put(ticker="AAA")))
        self.assertEqual(pm.projected_earnings_flagged, 0)
        self.assertEqual(pm.earnings_unknown, 1)

    def test_report_mode_records_the_projected_state(self):
        pm = self._pm()
        self.assertTrue(pm.log_trade(_bull_put(ticker="AAA")))
        self.assertEqual(self._rows()[0]["earnings_state"],
                         "projected_through_earnings")

    def test_the_flag_names_the_projected_date(self):
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            self._pm().log_trade(_bull_put(ticker="AAA"))
        out = buf.getvalue()
        self.assertIn("cadence projects", out)
        self.assertIn("AAA", out)


if __name__ == "__main__":
    unittest.main()
