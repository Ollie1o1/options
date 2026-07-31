"""Duplicate rows in the ledger: the auto-log guard, the audit, the TSV append.

Three defences against one failure mode — one decision showing up as two rows
and double-counting itself in every statistic:

* the auto-log near-duplicate guard, which refuses the second row at the door
  (automated feeders only — a manual entry is always the operator's call);
* ``scripts/audit_duplicate_trades.py``, which finds the rows already there
  without touching them;
* ``write_checkpoint``'s per-day history append, which stopped writing a second
  identical row every time maintenance re-ran on the same day.

unittest style on purpose — the options venv has no pytest, so these have to be
runnable locally as well as in CI.
"""
import json
import os
import sqlite3
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.paper_manager import PaperManager

_AUDIT_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "scripts", "audit_duplicate_trades.py",
)


def _load_audit():
    """Load the audit script by path — scripts/ is not an importable package."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("audit_duplicate_trades", _AUDIT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


audit = _load_audit()


def _config(path, dedup_window_days=3, max_capital_at_risk=None):
    cfg = {
        "exit_rules": {"take_profit": 0.50, "stop_loss": -0.25, "time_exit_dte": 21},
        "paper_trading": {"commission_per_contract": 0.0, "slippage_per_share": 0.05},
        "auto_log": {"allowed_strategies": ["Long Call"]},
    }
    if dedup_window_days is not None:
        cfg["auto_log"]["dedup_window_days"] = dedup_window_days
    if max_capital_at_risk is not None:
        cfg["auto_log"]["max_capital_at_risk"] = max_capital_at_risk
    with open(path, "w") as f:
        json.dump(cfg, f)
    return path


def _day(offset=0):
    return (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")


def _long_call(**over):
    """The ABBV pair's shape: same contract, logged a day apart."""
    trade = {
        "date": _day(0),
        "ticker": "ABBV",
        "expiration": "2026-08-21",
        "strike": 260.0,
        "type": "call",
        "entry_price": 8.30,
        "quality_score": 0.71,
        "strategy_name": "Long Call",
    }
    trade.update(over)
    return trade


class _LedgerCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")
        self.cfg = os.path.join(self.tmp.name, "config.json")

    def tearDown(self):
        self.tmp.cleanup()

    def n_rows(self):
        with sqlite3.connect(self.db) as conn:
            return conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    def pm(self, **cfg_kw):
        _config(self.cfg, **cfg_kw)
        return PaperManager(db_path=self.db, config_path=self.cfg)


# ── the guard ───────────────────────────────────────────────────────────────

class TestAutoLogDedupGuard(_LedgerCase):
    def test_fires_on_a_re_log_the_next_day(self):
        """The catch-up case: same contract, next day, price drifted a cent.

        The per-day dedup cannot see this — it is the reason the guard exists.
        """
        pm = self.pm()
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True))
        self.assertFalse(
            pm.log_trade_if_new(_long_call(date=_day(0), entry_price=8.35), auto_log=True))
        self.assertEqual(self.n_rows(), 1)
        self.assertEqual(pm.duplicate_rejected, 1)

    def test_fires_at_the_window_edge_and_releases_past_it(self):
        pm = self.pm(dedup_window_days=3)
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-3)), auto_log=True))
        self.assertFalse(pm.log_trade_if_new(_long_call(date=_day(0)), auto_log=True))

        pm2 = self.pm(dedup_window_days=3)
        self.assertTrue(pm2.log_trade_if_new(
            _long_call(ticker="MSFT", date=_day(-4)), auto_log=True))
        self.assertTrue(pm2.log_trade_if_new(
            _long_call(ticker="MSFT", date=_day(0)), auto_log=True))

    def test_allows_a_different_ticker(self):
        pm = self.pm()
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True))
        self.assertTrue(pm.log_trade_if_new(_long_call(ticker="MRK"), auto_log=True))
        self.assertEqual(self.n_rows(), 2)
        self.assertEqual(pm.duplicate_rejected, 0)

    def test_allows_the_same_ticker_at_a_different_strike(self):
        pm = self.pm()
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True))
        self.assertTrue(pm.log_trade_if_new(_long_call(strike=265.0), auto_log=True))
        self.assertEqual(self.n_rows(), 2)

    def test_allows_the_same_strike_at_a_different_expiration(self):
        pm = self.pm()
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True))
        self.assertTrue(pm.log_trade_if_new(_long_call(expiration="2026-09-18"), auto_log=True))
        self.assertEqual(self.n_rows(), 2)

    def test_allows_a_different_strategy_on_the_same_contract(self):
        pm = self.pm()
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True))
        self.assertTrue(pm.log_trade_if_new(
            _long_call(strategy_name="Short Call"), auto_log=True))
        self.assertEqual(self.n_rows(), 2)

    def test_manual_log_trade_is_never_refused(self):
        """A deliberate entry is the operator's call, duplicate-looking or not."""
        pm = self.pm()
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True))
        self.assertTrue(pm.log_trade(_long_call()))
        self.assertEqual(self.n_rows(), 2)
        self.assertEqual(pm.duplicate_rejected, 0)

    def test_if_new_without_the_flag_is_not_gated(self):
        """The default is the manual behaviour: only an explicit auto_log arms it."""
        pm = self.pm()
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1))))
        self.assertTrue(pm.log_trade_if_new(_long_call()))
        self.assertEqual(self.n_rows(), 2)

    def test_zero_window_disables_the_guard(self):
        pm = self.pm(dedup_window_days=0)
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True))
        self.assertTrue(pm.log_trade_if_new(_long_call(), auto_log=True))
        self.assertEqual(self.n_rows(), 2)

    def test_window_defaults_to_three_days_when_unconfigured(self):
        pm = self.pm(dedup_window_days=None)
        self.assertEqual(pm._dedup_window_days, 3)
        self.assertTrue(pm.log_trade_if_new(_long_call(date=_day(-2)), auto_log=True))
        self.assertFalse(pm.log_trade_if_new(_long_call(), auto_log=True))

    def test_the_refusal_says_which_row_it_matched(self):
        """A silent refusal is how a feeder looks broken instead of gated."""
        pm = self.pm()
        pm.log_trade_if_new(_long_call(date=_day(-1)), auto_log=True)
        with self.assertLogs("src.paper_manager", level="WARNING") as logged:
            pm.log_trade_if_new(_long_call(), auto_log=True)
        text = "\n".join(logged.output)
        self.assertIn("AUTO-LOG DUPLICATE REFUSED", text)
        self.assertIn("ABBV", text)
        self.assertIn("dedup_window_days", text)

    def test_spreads_and_condors_are_gated_too(self):
        pm = self.pm()
        spread = {
            "date": _day(-1), "ticker": "SPY", "expiration": "2026-07-17",
            "short_strike": 745.0, "long_strike": 750.0, "type": "Bear Call",
            "net_credit": 0.49, "max_profit": 49.0, "max_loss": 451.0,
            "quality_score": 0.6,
        }
        self.assertTrue(pm.log_spread_if_new(spread, auto_log=True))
        self.assertFalse(pm.log_spread_if_new(
            dict(spread, date=_day(0), net_credit=0.51), auto_log=True))

        condor = {
            "date": _day(-1), "ticker": "QQQ", "expiration": "2026-07-17",
            "short_put_strike": 700.0, "long_put_strike": 695.0,
            "short_call_strike": 760.0, "long_call_strike": 765.0,
            "total_credit": 1.10, "max_profit": 110.0, "max_risk": 390.0,
            "quality_score": 0.6,
        }
        self.assertTrue(pm.log_iron_condor_if_new(condor, auto_log=True))
        self.assertFalse(pm.log_iron_condor_if_new(
            dict(condor, date=_day(0), total_credit=1.15), auto_log=True))

    def test_condors_sharing_a_short_put_but_not_a_call_wing_both_log(self):
        """``strike`` holds only the anchor leg — the short put for a condor.

        Two condors on the same ticker and expiration can share it and be
        completely different structures on the call side. Matching on the anchor
        alone would silently refuse the second, which is the same failure class
        as the short-put window starvation (2026-07-30).
        """
        pm = self.pm()
        condor = {
            "date": _day(-1), "ticker": "QQQ", "expiration": "2026-07-17",
            "short_put_strike": 700.0, "long_put_strike": 695.0,
            "short_call_strike": 760.0, "long_call_strike": 765.0,
            "total_credit": 1.10, "max_profit": 110.0, "max_risk": 390.0,
            "quality_score": 0.6,
        }
        self.assertTrue(pm.log_iron_condor_if_new(condor, auto_log=True))
        wider_calls = dict(condor, date=_day(0),
                           short_call_strike=770.0, long_call_strike=775.0)
        self.assertTrue(pm.log_iron_condor_if_new(wider_calls, auto_log=True))
        self.assertEqual(self.n_rows(), 2)
        self.assertEqual(pm.duplicate_rejected, 0)

    def test_condors_differing_only_on_the_put_wing_both_log(self):
        pm = self.pm()
        condor = {
            "date": _day(-1), "ticker": "QQQ", "expiration": "2026-07-17",
            "short_put_strike": 700.0, "long_put_strike": 695.0,
            "short_call_strike": 760.0, "long_call_strike": 765.0,
            "total_credit": 1.10, "max_risk": 390.0, "quality_score": 0.6,
        }
        self.assertTrue(pm.log_iron_condor_if_new(condor, auto_log=True))
        self.assertTrue(pm.log_iron_condor_if_new(
            dict(condor, date=_day(0), long_put_strike=690.0), auto_log=True))
        self.assertEqual(self.n_rows(), 2)

    def test_spreads_at_the_same_short_strike_but_different_widths_both_log(self):
        pm = self.pm()
        spread = {
            "date": _day(-1), "ticker": "SPY", "expiration": "2026-07-17",
            "short_strike": 745.0, "long_strike": 750.0, "type": "Bear Call",
            "net_credit": 0.49, "max_loss": 451.0, "quality_score": 0.6,
        }
        self.assertTrue(pm.log_spread_if_new(spread, auto_log=True))
        self.assertTrue(pm.log_spread_if_new(
            dict(spread, date=_day(0), long_strike=755.0), auto_log=True))
        self.assertEqual(self.n_rows(), 2)
        self.assertEqual(pm.duplicate_rejected, 0)

    def test_an_identical_structure_is_still_refused(self):
        """The wing check must not defeat the guard for a true re-log."""
        pm = self.pm()
        condor = {
            "date": _day(-1), "ticker": "QQQ", "expiration": "2026-07-17",
            "short_put_strike": 700.0, "long_put_strike": 695.0,
            "short_call_strike": 760.0, "long_call_strike": 765.0,
            "total_credit": 1.10, "max_risk": 390.0, "quality_score": 0.6,
        }
        self.assertTrue(pm.log_iron_condor_if_new(condor, auto_log=True))
        self.assertFalse(pm.log_iron_condor_if_new(
            dict(condor, date=_day(0), total_credit=1.15), auto_log=True))
        self.assertEqual(self.n_rows(), 1)

    def test_a_single_leg_does_not_collide_with_a_spread_anchor(self):
        """A Long Call at 745 and a Bear Call anchored at 745 are not the same
        row — the single leg's NULL wings only match other NULL wings."""
        pm = self.pm()
        self.assertTrue(pm.log_spread_if_new({
            "date": _day(-1), "ticker": "SPY", "expiration": "2026-07-17",
            "short_strike": 745.0, "long_strike": 750.0, "type": "Bear Call",
            "net_credit": 0.49, "max_loss": 451.0, "quality_score": 0.6,
        }, auto_log=True))
        self.assertTrue(pm.log_trade_if_new(_long_call(
            ticker="SPY", strike=745.0, expiration="2026-07-17",
            strategy_name="Bear Call", entry_price=0.49), auto_log=True))
        self.assertEqual(self.n_rows(), 2)

    def test_a_zero_wing_and_a_missing_wing_are_the_same_leg(self):
        """Auto-log payloads default an absent wing to 0 (`row.get(k, 0)`) while
        log_trade stores NULL for one never set. Those must not read as two
        different structures, or the guard would never fire on that path."""
        pm = self.pm()
        spread = {
            "date": _day(-1), "ticker": "SPY", "expiration": "2026-07-17",
            "short_strike": 745.0, "long_strike": 750.0, "type": "Bear Call",
            "net_credit": 0.49, "max_loss": 451.0, "quality_score": 0.6,
        }
        self.assertTrue(pm.log_spread_if_new(spread, auto_log=True))
        # Same structure, re-logged with the condor wings explicitly zeroed.
        repeat = dict(spread, date=_day(0), net_credit=0.51,
                      short_call_strike=0, long_call_strike=0,
                      short_put_strike=0, long_put_strike=0)
        self.assertFalse(pm.log_spread_if_new(repeat, auto_log=True))
        self.assertEqual(self.n_rows(), 1)

    def test_manual_condor_logging_is_still_allowed(self):
        """The interactive [L] menu calls log_iron_condor_if_new without the flag."""
        pm = self.pm()
        condor = {
            "date": _day(-1), "ticker": "QQQ", "expiration": "2026-07-17",
            "short_put_strike": 700.0, "long_put_strike": 695.0,
            "short_call_strike": 760.0, "long_call_strike": 765.0,
            "total_credit": 1.10, "max_profit": 110.0, "max_risk": 390.0,
            "quality_score": 0.6,
        }
        self.assertTrue(pm.log_iron_condor_if_new(condor))
        self.assertTrue(pm.log_iron_condor_if_new(dict(condor, date=_day(0))))

    def test_the_flag_does_not_leak_into_the_callers_dict(self):
        pm = self.pm()
        trade = _long_call()
        pm.log_trade_if_new(trade, auto_log=True)
        self.assertNotIn("auto_log", trade)


# ── the audit ───────────────────────────────────────────────────────────────

def _seed(db_path, rows):
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE trades (entry_id INTEGER PRIMARY KEY AUTOINCREMENT, date TEXT, "
        "ticker TEXT, strategy_name TEXT, type TEXT, strike REAL, expiration TEXT, "
        "entry_price REAL, exit_price REAL, exit_date TEXT, pnl_pct REAL, pnl_usd REAL, "
        "status TEXT, quantity REAL, paper_only INTEGER, weight_profile TEXT, "
        "capital_at_risk REAL, exit_reason TEXT)"
    )
    for r in rows:
        cols = ", ".join(r)
        marks = ", ".join("?" for _ in r)
        conn.execute(f"INSERT INTO trades ({cols}) VALUES ({marks})", tuple(r.values()))
    conn.commit()
    conn.close()


def _abbv(date, **over):
    row = {
        "date": date, "ticker": "ABBV", "strategy_name": "Long Call", "type": "call",
        "strike": 260.0, "expiration": "2026-08-21", "entry_price": 8.30,
        "exit_price": 3.75, "exit_date": "2026-07-14 14:06:55", "pnl_pct": -0.6098,
        "pnl_usd": -506.10, "status": "CLOSED", "quantity": 1.0, "weight_profile": "baseline",
    }
    row.update(over)
    return row


class TestAuditFindsCandidates(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")

    def tearDown(self):
        self.tmp.cleanup()

    def groups(self, rows, window_days=3):
        _seed(self.db, rows)
        conn = audit._connect_readonly(self.db)
        try:
            fetched = audit.fetch_rows(conn)
        finally:
            conn.close()
        return audit.find_candidate_duplicates(fetched, window_days=window_days)

    def test_finds_the_abbv_pair(self):
        """The known seed case: 2026-07-07 and 2026-07-08, $8.30 in, $3.75 out."""
        groups = self.groups([_abbv("2026-07-07"), _abbv("2026-07-08")])
        self.assertEqual(len(groups), 1)
        g = groups[0]
        self.assertEqual(g["ticker"], "ABBV")
        self.assertEqual(g["size"], 2)
        self.assertEqual(g["span_days"], 1)
        self.assertEqual(g["n_closed"], 2)
        self.assertTrue(g["identical_exits"])
        self.assertAlmostEqual(g["excess_pnl"], -506.10, places=2)

    def test_ignores_a_different_strike(self):
        self.assertEqual(
            self.groups([_abbv("2026-07-07"), _abbv("2026-07-08", strike=265.0)]), [])

    def test_ignores_a_different_entry_price(self):
        self.assertEqual(
            self.groups([_abbv("2026-07-07"), _abbv("2026-07-08", entry_price=8.55)]), [])

    def test_ignores_entries_outside_the_window(self):
        self.assertEqual(
            self.groups([_abbv("2026-07-07"), _abbv("2026-07-20")]), [])

    def test_chains_a_daily_re_log_into_one_group(self):
        rows = [_abbv(f"2026-07-0{d}") for d in (5, 6, 7, 8)]
        groups = self.groups(rows)
        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0]["size"], 4)

    def test_rows_with_an_unusable_date_are_skipped_not_guessed(self):
        self.assertEqual(self.groups([_abbv("2026-07-07"), _abbv("")]), [])

    def test_report_lists_the_rows_and_says_nothing_was_changed(self):
        groups = self.groups([_abbv("2026-07-07"), _abbv("2026-07-08")])
        md = audit.render_report(groups, total_rows=2, window_days=3,
                                 db_path=self.db, generated="2026-07-31 00:00:00")
        self.assertIn("ABBV Long Call $260", md)
        self.assertIn("2026-07-07", md)
        self.assertIn("2026-07-08", md)
        self.assertIn("identical exits", md)
        self.assertIn("Nothing has been deleted or edited", md)
        self.assertIn("Candidate duplicate groups: **1**", md)

    def test_empty_report_is_still_a_report(self):
        md = audit.render_report([], total_rows=0, window_days=3, db_path=self.db)
        self.assertIn("None found", md)

    def test_the_audit_connection_cannot_write(self):
        """Read-only is enforced by the connection, not by good intentions."""
        _seed(self.db, [_abbv("2026-07-07")])
        conn = audit._connect_readonly(self.db)
        try:
            with self.assertRaises(sqlite3.OperationalError):
                conn.execute("DELETE FROM trades")
        finally:
            conn.close()
        with sqlite3.connect(self.db) as check:
            self.assertEqual(check.execute("SELECT COUNT(*) FROM trades").fetchone()[0], 1)


# ── the checkpoint history append ───────────────────────────────────────────

class TestCheckpointHistoryIsIdempotentPerDay(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.out = os.path.join(self.tmp.name, "reports")

    def tearDown(self):
        self.tmp.cleanup()

    def _result(self, today="2026-07-31", n=64, **over):
        r = {
            "today": today, "phase1_start": "2026-05-27", "weeks_elapsed": 9,
            "n_trades": n, "ic_pearson": 0.0486, "p_pearson": 0.6902,
            "ic_spearman": -0.0204, "p_spearman": 0.8730, "ic_95_ci": [-0.175, 0.248],
            "decision": "GATHERING", "posterior_ic_ge_008": 0.42,
            "max_capital_at_risk": None, "n_affordable": None,
            "ic_pearson_affordable": None, "p_pearson_affordable": None,
            "ic_spearman_affordable": None, "p_spearman_affordable": None,
            "short_premium": None,
        }
        r.update(over)
        return r

    def _write(self, **kw):
        from src.phase1_checkpoint import write_checkpoint
        return write_checkpoint(self._result(**kw), output_dir=self.out)

    def _data_rows(self):
        with open(os.path.join(self.out, "checkpoint_history.tsv")) as f:
            return [ln for ln in f.read().splitlines()
                    if ln.strip() and not ln.startswith("date\t")]

    def test_a_second_identical_run_the_same_day_appends_nothing(self):
        self.assertTrue(self._write()["history_appended"])
        self.assertFalse(self._write()["history_appended"])
        self.assertFalse(self._write()["history_appended"])
        self.assertEqual(len(self._data_rows()), 1)

    def test_the_same_day_at_a_different_n_still_appends(self):
        """A cohort that grew between runs is a new observation, not a repeat."""
        self._write(n=64)
        self.assertTrue(self._write(n=65)["history_appended"])
        rows = self._data_rows()
        self.assertEqual(len(rows), 2)
        self.assertEqual([r.split("\t")[2] for r in rows], ["64", "65"])

    def test_a_new_day_appends(self):
        self._write(today="2026-07-30")
        self.assertTrue(self._write(today="2026-07-31")["history_appended"])
        self.assertEqual(len(self._data_rows()), 2)

    def test_a_six_field_legacy_last_row_is_read_positionally(self):
        """The history holds both widths; date is index 0 and n index 2 in each."""
        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "checkpoint_history.tsv"), "w") as f:
            f.write("date\tweeks\tn\tic\tp\tdecision\n"
                    "2026-07-31\t9\t64\t0.0486\t0.6902\tGATHERING\n")
        self.assertFalse(self._write(today="2026-07-31", n=64)["history_appended"])
        self.assertEqual(len(self._data_rows()), 1)
        self.assertTrue(self._write(today="2026-07-31", n=65)["history_appended"])
        self.assertEqual(len(self._data_rows()), 2)

    def test_the_markdown_is_still_rewritten_on_a_skipped_append(self):
        """Only the history row is suppressed — the report itself stays current.

        Asserted on content, not existence: the file is written before the skip
        logic runs, so an existence check would pass even if the skip short-
        circuited the whole function.
        """
        paths = self._write(ic_pearson=0.0486)
        result = self._write(ic_pearson=0.1234)
        self.assertFalse(result["history_appended"])
        with open(paths["md"]) as f:
            md = f.read()
        self.assertIn("+0.123", md)
        self.assertNotIn("+0.049", md)
        self.assertEqual(len(self._data_rows()), 1)


if __name__ == "__main__":
    unittest.main()
