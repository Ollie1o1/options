"""The operator's own long-option picks, scored at a fixed 1/2/3-session horizon.

Every fixture lives in a temporary directory: these tests build their own
ledger and candidate database and never name the real ones.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

from src import candidate_marks as cm
from src import own_picks as op

# Mon..Fri, then the following Mon..Wed.
_DAYS = ["2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08", "2026-10-09",
         "2026-10-12", "2026-10-13", "2026-10-14"]


def _series(values):
    return dict(zip(_DAYS, values))


def _ledger(path, rows):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trades (entry_id INTEGER PRIMARY KEY, date TEXT, "
        "ticker TEXT, expiration TEXT, strike REAL, type TEXT, "
        "entry_price REAL, entry_price_cross REAL, strategy_name TEXT, "
        "origin TEXT)")
    conn.executemany(
        "INSERT INTO trades (entry_id, date, ticker, expiration, strike, type,"
        " entry_price, entry_price_cross, strategy_name, origin) "
        "VALUES (?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


def _pick(entry_id=1, date="2026-10-05", ticker="AAA", opt_type="call",
          entry_price=2.00, cross=None, strategy=None, origin="manual"):
    strategy = strategy or ("Long Call" if opt_type == "call" else "Long Put")
    return (entry_id, date, ticker, "2026-11-20", 100.0, opt_type,
            entry_price, cross, strategy, origin)


class TestSessionDate(unittest.TestCase):

    def test_counts_sessions_not_calendar_days(self):
        # Friday entry: the first session after it is Monday.
        self.assertEqual(op.session_date(_DAYS, "2026-10-09", 1), "2026-10-12")

    def test_a_session_that_has_not_happened_yet_is_none(self):
        self.assertIsNone(op.session_date(_DAYS, "2026-10-13", 3))

    def test_an_entry_day_with_no_close_is_none(self):
        self.assertIsNone(op.session_date(_DAYS, "2026-10-10", 1))


class TestDirectionReturn(unittest.TestCase):

    def test_call_is_the_stock_move_net_of_spy(self):
        stock = _series([100, 110, 110, 110, 110, 110, 110, 110])
        spy = _series([500, 505, 505, 505, 505, 505, 505, 505])
        got = op.direction_return(stock, spy, "2026-10-05", 1, "call")
        self.assertAlmostEqual(got, 0.10 - 0.01)

    def test_put_flips_the_sign(self):
        stock = _series([100, 110, 110, 110, 110, 110, 110, 110])
        spy = _series([500, 505, 505, 505, 505, 505, 505, 505])
        got = op.direction_return(stock, spy, "2026-10-05", 1, "put")
        self.assertAlmostEqual(got, -(0.10 - 0.01))

    def test_missing_close_is_not_recorded_rather_than_zero(self):
        stock = {"2026-10-05": 100.0}
        spy = _series([500] * 8)
        self.assertIsNone(
            op.direction_return(stock, spy, "2026-10-05", 1, "call"))


class TestOptionReturn(unittest.TestCase):

    def test_buys_at_the_recorded_entry_and_sells_at_the_bid(self):
        self.assertAlmostEqual(op.option_return(2.00, 2.50), 0.25)

    def test_a_zero_bid_is_a_total_loss_not_a_missing_quote(self):
        self.assertEqual(op.option_return(2.00, 0.0), -1.0)

    def test_no_bid_is_not_recorded(self):
        self.assertIsNone(op.option_return(2.00, None))

    def test_no_usable_entry_price_is_not_recorded(self):
        self.assertIsNone(op.option_return(0.0, 1.0))
        self.assertIsNone(op.option_return(None, 1.0))


class TestSummarize(unittest.TestCase):

    def test_same_ticker_same_week_is_one_cluster(self):
        obs = [(op.cluster_key("AAA", "2026-10-05"), 0.10),
               (op.cluster_key("AAA", "2026-10-07"), 0.30),
               (op.cluster_key("BBB", "2026-10-07"), -0.20)]
        got = op.summarize(obs)
        self.assertEqual(got["n_obs"], 3)
        self.assertEqual(got["n_clusters"], 2)
        # Mean of cluster means: (0.20 + -0.20) / 2, not the row mean 0.0667.
        self.assertAlmostEqual(got["mean"], 0.0)
        self.assertAlmostEqual(got["hit_rate"], 0.5)

    def test_same_ticker_next_week_is_a_second_cluster(self):
        self.assertNotEqual(op.cluster_key("AAA", "2026-10-09"),
                            op.cluster_key("AAA", "2026-10-12"))

    def test_no_interval_below_the_minimum_cluster_count(self):
        obs = [(("T%d" % i, 2026, 41), 0.05) for i in range(op.MIN_CLUSTERS - 1)]
        self.assertIsNone(op.summarize(obs)["ci"])

    def test_interval_at_the_minimum_cluster_count(self):
        obs = [(("T%d" % i, 2026, 41), 0.05 if i % 2 else -0.01)
               for i in range(op.MIN_CLUSTERS)]
        lo, hi = op.summarize(obs)["ci"]
        self.assertLess(lo, hi)
        self.assertLessEqual(lo, op.summarize(obs)["mean"])

    def test_nothing_observed_is_empty_not_zero(self):
        got = op.summarize([])
        self.assertEqual(got["n_clusters"], 0)
        self.assertIsNone(got["mean"])
        self.assertIsNone(got["hit_rate"])


class _Dbs(unittest.TestCase):

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ledger = os.path.join(self._tmp.name, "ledger.db")
        self.cand = os.path.join(self._tmp.name, "cand.db")
        cm.connect(self.cand).close()


class TestManualPicks(_Dbs):

    def test_only_manual_long_options_are_picks(self):
        _ledger(self.ledger, [
            _pick(1),
            _pick(2, origin="auto"),
            _pick(3, origin=None),
            _pick(4, strategy="Bull Put"),
            _pick(5, opt_type="put"),
        ])
        self.assertEqual([p["entry_id"] for p in op.manual_picks(self.ledger)],
                         [1, 5])

    def test_a_ledger_without_the_origin_column_has_no_picks(self):
        conn = sqlite3.connect(self.ledger)
        conn.execute("CREATE TABLE trades (entry_id INTEGER PRIMARY KEY)")
        conn.commit()
        conn.close()
        self.assertEqual(op.manual_picks(self.ledger), [])


class TestRecordPickMarks(_Dbs):

    def _marks(self):
        conn = sqlite3.connect(self.cand)
        try:
            return conn.execute(
                "SELECT entry_id, mark_date, bid, ask FROM pick_marks "
                "ORDER BY entry_id, mark_date").fetchall()
        except sqlite3.OperationalError:
            # Nothing was ever due, so the table was never created.
            return []
        finally:
            conn.close()

    def test_marks_a_recent_manual_pick_from_the_chain(self):
        _ledger(self.ledger, [_pick(1, date="2026-10-05")])
        fetch = lambda t, e: {(100.0, "call"): (2.40, 2.60)}
        n = op.record_pick_marks(ledger_path=self.ledger, db_path=self.cand,
                                 today="2026-10-06", fetch=fetch)
        self.assertEqual(n, 1)
        self.assertEqual(self._marks(), [(1, "2026-10-06", 2.40, 2.60)])

    def test_a_zero_bid_is_recorded_as_zero(self):
        _ledger(self.ledger, [_pick(1, date="2026-10-05")])
        fetch = lambda t, e: {(100.0, "call"): (0.0, 0.05)}
        op.record_pick_marks(ledger_path=self.ledger, db_path=self.cand,
                             today="2026-10-06", fetch=fetch)
        self.assertEqual(self._marks(), [(1, "2026-10-06", 0.0, 0.05)])

    def test_a_missing_contract_writes_nothing(self):
        _ledger(self.ledger, [_pick(1, date="2026-10-05")])
        n = op.record_pick_marks(ledger_path=self.ledger, db_path=self.cand,
                                 today="2026-10-06", fetch=lambda t, e: {})
        self.assertEqual(n, 0)
        self.assertEqual(self._marks(), [])

    def test_a_pick_past_the_marking_window_is_left_alone(self):
        _ledger(self.ledger, [_pick(1, date="2026-09-01")])
        calls = []
        fetch = lambda t, e: calls.append(t) or {(100.0, "call"): (1.0, 1.1)}
        op.record_pick_marks(ledger_path=self.ledger, db_path=self.cand,
                             today="2026-10-06", fetch=fetch)
        self.assertEqual(calls, [])
        self.assertEqual(self._marks(), [])

    def test_a_ledger_that_does_not_exist_is_nothing_to_mark(self):
        from src import candidate_record as cr
        before = cr.STATS["errors"]
        n = op.record_pick_marks(
            ledger_path=os.path.join(self._tmp.name, "absent.db"),
            db_path=self.cand, today="2026-10-06", fetch=lambda t, e: {})
        self.assertEqual(n, 0)
        self.assertEqual(cr.STATS["errors"], before)

    def test_running_twice_in_a_day_keeps_one_mark(self):
        _ledger(self.ledger, [_pick(1, date="2026-10-05")])
        fetch = lambda t, e: {(100.0, "call"): (2.40, 2.60)}
        for _ in range(2):
            op.record_pick_marks(ledger_path=self.ledger, db_path=self.cand,
                                 today="2026-10-06", fetch=fetch)
        self.assertEqual(len(self._marks()), 1)


class TestScorecard(_Dbs):

    def _closes(self, table):
        return lambda ticker: table.get(ticker, {})

    def _add_candidate(self, key, symbol, opt_type, ask, day="2026-10-05",
                       scan="s1", gate_passed=1):
        conn = sqlite3.connect(self.cand)
        conn.execute(
            "INSERT INTO candidates (scan_id, ts, board, contract_key, symbol,"
            " strategy_name, expiration, strike, opt_type, bid, ask,"
            " gate_passed) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (scan, day + "T15:00:00+00:00", "AUTO-LOG", key, symbol,
             "Long Call" if opt_type == "call" else "Long Put",
             "2026-11-20", 100.0, opt_type, ask - 0.1, ask, gate_passed))
        conn.commit()
        conn.close()

    def _add_cand_mark(self, key, day, bid):
        conn = sqlite3.connect(self.cand)
        conn.execute("INSERT INTO candidate_marks VALUES (?,?,?,?,?,?)",
                     (key, day, bid, bid + 0.1, bid + 0.05, "live_quote"))
        conn.commit()
        conn.close()

    def test_scores_a_pick_at_each_session(self):
        _ledger(self.ledger, [_pick(1, date="2026-10-05", entry_price=2.00)])
        op.record_pick_marks(
            ledger_path=self.ledger, db_path=self.cand, today="2026-10-06",
            fetch=lambda t, e: {(100.0, "call"): (2.50, 2.70)})
        closes = self._closes({
            "AAA": _series([100, 104, 106, 103, 103, 103, 103, 103]),
            "SPY": _series([500, 500, 505, 505, 505, 505, 505, 505])})
        card = op.build_scorecard(ledger_path=self.ledger, db_path=self.cand,
                                  closes_fn=closes)
        self.assertEqual(card["n_picks"], 1)
        self.assertAlmostEqual(card["picks"]["direction"][1]["mean"], 0.04)
        self.assertAlmostEqual(card["picks"]["direction"][2]["mean"], 0.05)
        self.assertAlmostEqual(card["picks"]["option"][1]["mean"], 0.25)
        # No quote was taken on session 2: not recorded, never zero.
        self.assertEqual(card["picks"]["option"][2]["n_obs"], 0)
        self.assertIsNone(card["picks"]["option"][2]["mean"])

    def test_the_fill_that_crossed_the_spread_is_the_entry_basis(self):
        _ledger(self.ledger, [_pick(1, entry_price=2.00, cross=2.50)])
        op.record_pick_marks(
            ledger_path=self.ledger, db_path=self.cand, today="2026-10-06",
            fetch=lambda t, e: {(100.0, "call"): (2.50, 2.70)})
        closes = self._closes({"AAA": _series([100] * 8),
                               "SPY": _series([500] * 8)})
        card = op.build_scorecard(ledger_path=self.ledger, db_path=self.cand,
                                  closes_fn=closes)
        self.assertAlmostEqual(card["picks"]["option"][1]["mean"], 0.0)

    def test_comparison_is_what_the_board_passed_that_day_and_was_not_picked(self):
        _ledger(self.ledger, [_pick(1, date="2026-10-05")])
        self._add_candidate("AAA|2026-11-20|call|100", "AAA", "call", 2.0)
        self._add_candidate("BBB|2026-11-20|put|100", "BBB", "put", 4.0)
        # Same contract seen by a second scan that day: still one observation.
        self._add_candidate("BBB|2026-11-20|put|100", "BBB", "put", 4.0,
                            scan="s2")
        self._add_candidate("CCC|2026-11-20|call|100", "CCC", "call", 1.0,
                            gate_passed=0)
        self._add_candidate("DDD|2026-11-20|call|100", "DDD", "call", 1.0,
                            day="2026-10-07")
        self._add_cand_mark("BBB|2026-11-20|put|100", "2026-10-06", 3.0)
        closes = self._closes({
            "AAA": _series([100] * 8),
            "BBB": _series([100, 98, 98, 98, 98, 98, 98, 98]),
            "SPY": _series([500] * 8)})
        card = op.build_scorecard(ledger_path=self.ledger, db_path=self.cand,
                                  closes_fn=closes)
        board = card["board"]
        self.assertEqual(board["direction"][1]["n_obs"], 1)
        self.assertAlmostEqual(board["direction"][1]["mean"], 0.02)
        self.assertAlmostEqual(board["option"][1]["mean"], -0.25)

    def test_the_portfolio_panel_scores_picks_without_fetching_the_board(self):
        _ledger(self.ledger, [_pick(1, date="2026-10-05")])
        self._add_candidate("BBB|2026-11-20|put|100", "BBB", "put", 4.0)
        asked = []
        table = {"AAA": _series([100, 104, 104, 104, 104, 104, 104, 104]),
                 "SPY": _series([500] * 8)}

        def closes(ticker):
            asked.append(ticker)
            return table.get(ticker, {})

        text = "\n".join(op.panel_lines(ledger_path=self.ledger,
                                        db_path=self.cand, closes_fn=closes))
        self.assertNotIn("BBB", asked)
        self.assertIn("+4.0%", text)
        self.assertNotIn("Board", text)
        self.assertIn("python -m src.own_picks", text)

    def test_no_picks_yet(self):
        _ledger(self.ledger, [_pick(1, origin=None)])
        card = op.build_scorecard(ledger_path=self.ledger, db_path=self.cand,
                                  closes_fn=self._closes({}))
        self.assertEqual(card["n_picks"], 0)


class TestRender(unittest.TestCase):

    def _card(self, n_clusters, ci=None):
        cell = {"n_obs": n_clusters, "n_clusters": n_clusters, "mean": 0.031,
                "hit_rate": 0.6, "ci": ci}
        empty = {"n_obs": 0, "n_clusters": 0, "mean": None, "hit_rate": None,
                 "ci": None}
        side = {"direction": {1: cell, 2: cell, 3: empty},
                "option": {1: cell, 2: empty, 3: empty}}
        return {"n_picks": n_clusters, "picks": side, "board": side}

    def test_says_too_few_to_judge_below_the_minimum(self):
        text = "\n".join(op.render(self._card(4)))
        self.assertIn("too few to judge", text)
        self.assertIn("4 of %d" % op.MIN_CLUSTERS, text)

    def test_shows_the_interval_once_there_are_enough(self):
        text = "\n".join(op.render(self._card(op.MIN_CLUSTERS,
                                              ci=(-0.012, 0.074))))
        self.assertNotIn("too few to judge", text)
        self.assertIn("-1.2%", text)
        self.assertIn("+7.4%", text)

    def test_an_unobserved_cell_is_shown_as_missing_not_as_zero(self):
        text = "\n".join(op.render(self._card(4)))
        self.assertIn("not recorded", text)
        self.assertNotIn("+0.0%", text)

    def test_empty_ledger_tells_the_operator_how_to_start(self):
        text = "\n".join(op.render({"n_picks": 0, "picks": None,
                                    "board": None}))
        self.assertIn("No manual long-option picks", text)


if __name__ == "__main__":
    unittest.main()


class _FakeManager:
    def __init__(self, accept=True):
        self.accept, self.trades = accept, []

    def log_trade(self, trade):
        self.trades.append(dict(trade))
        return self.accept


_QUOTE = {(250.0, "call"): (4.0, 4.2), (240.0, "put"): (3.0, 3.3)}


class TestLogPick(unittest.TestCase):
    def _log(self, *a, **kw):
        kw.setdefault("fetch", lambda t, e: _QUOTE)
        kw.setdefault("today", "2026-10-14")
        return op.log_pick(*a, **kw)

    def test_a_call_is_a_manual_long_call_at_the_given_fill(self):
        m = _FakeManager()
        ok, msg = self._log("aapl", "2026-11-20", 250, "CALL", price=4.10, manager=m)
        self.assertTrue(ok, msg)
        t = m.trades[0]
        self.assertEqual((t["ticker"], t["strategy_name"], t["type"], t["entry_price"]),
                         ("AAPL", "Long Call", "call", 4.10))
        self.assertIsNone(t["quality_score"])   # no board score, never a fake one
        self.assertNotIn("allow_unaffordable", t)

    def test_without_a_price_it_buys_at_the_live_ask(self):
        m = _FakeManager()
        ok, msg = self._log("AAPL", "2026-11-20", 240, "put", manager=m)
        self.assertTrue(ok)
        self.assertEqual((m.trades[0]["entry_price"], m.trades[0]["strategy_name"]),
                         (3.3, "Long Put"))
        self.assertIn("live ask", msg)

    def test_no_quote_and_no_price_is_refused(self):
        ok, msg = self._log("AAPL", "2026-11-20", 999, "call",
                            manager=_FakeManager())
        self.assertFalse(ok)
        self.assertIn("--price", msg)

    def test_no_quote_with_a_price_logs_but_warns(self):
        ok, msg = self._log("AAPL", "2026-11-20", 999, "call", price=1.0,
                            manager=_FakeManager())
        self.assertTrue(ok)
        self.assertIn("no live quote", msg)

    def test_bad_inputs_are_refused_before_anything_is_written(self):
        m = _FakeManager()
        for args in (("AAPL", "2026-11-20", 250, "straddle"),
                     ("AAPL", "11/20/2026", 250, "call"),
                     ("AAPL", "2026-10-14", 250, "call")):
            ok, _ = self._log(*args, price=1.0, manager=m)
            self.assertFalse(ok, args)
        self.assertEqual(m.trades, [])

    def test_force_bypasses_the_ledger_gates(self):
        m = _FakeManager()
        self._log("AAPL", "2026-11-20", 250, "call", price=4.1, force=True, manager=m)
        self.assertTrue(all(m.trades[0][k] for k in
                            ("allow_untradeable", "allow_unaffordable", "allow_unsized")))

    def test_a_ledger_refusal_points_at_force(self):
        ok, msg = self._log("AAPL", "2026-11-20", 250, "call", price=4.1,
                            manager=_FakeManager(accept=False))
        self.assertFalse(ok)
        self.assertIn("--force", msg)


class TestLogPickEndToEnd(unittest.TestCase):
    """Through the real PaperManager into a temporary ledger, then read back
    by the scorecard's own query."""

    def test_the_logged_pick_is_a_manual_pick(self):
        from datetime import date, timedelta
        exp = (date.today() + timedelta(days=30)).isoformat()
        with tempfile.TemporaryDirectory() as d:
            ledger = os.path.join(d, "ledger.db")
            ok, msg = op.log_pick(
                "AAPL", exp, 250, "call", price=4.1, force=True,
                ledger_path=ledger, fetch=lambda t, e: _QUOTE)
            self.assertTrue(ok, msg)
            picks = op.manual_picks(ledger)
            self.assertEqual(len(picks), 1)
            self.assertEqual((picks[0]["ticker"], picks[0]["strike"],
                              picks[0]["type"], picks[0]["entry_basis"]),
                             ("AAPL", 250.0, "call", 4.1))
