"""Corpus B joins the live ledger to the chain archive. It is small but carries
real bid/ask at every point, so it is the only place a cost assumption can be
checked rather than believed.

Tests build throwaway databases; no test names the real ledger or archive.
"""
import os
import sqlite3
import tempfile
import unittest

from src.policy_lab.paths import load_corpus_b


class TestCorpusBLoader(unittest.TestCase):
    def setUp(self):
        fd, self.ledger = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        fd, self.archive = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        con = sqlite3.connect(self.ledger)
        con.executescript("""
            CREATE TABLE trades (
                entry_id INTEGER PRIMARY KEY, date TEXT, ticker TEXT,
                expiration TEXT, strategy_name TEXT, status TEXT,
                exit_date TEXT, pnl_usd REAL, capital_at_risk REAL,
                net_credit REAL, spread_width REAL, entry_price REAL,
                type TEXT, strike REAL, short_put_strike REAL,
                long_put_strike REAL, long_strike REAL);
        """)
        # Matches the real ledger's shape for every one of its 171 closed
        # Bull Put rows: short_put_strike AND long_put_strike are NULL, the
        # short leg lives in the generic `strike` column and the long leg in
        # `long_strike` instead. A loader that joins on short_put_strike /
        # long_put_strike alone returns zero rows for this fixture.
        con.execute(
            "INSERT INTO trades (entry_id, date, ticker, expiration, "
            "strategy_name, status, exit_date, pnl_usd, capital_at_risk, "
            "net_credit, spread_width, type, strike, short_put_strike, "
            "long_put_strike, long_strike) VALUES "
            "(1,'2026-06-10','AMD','2026-07-17','Bull Put','CLOSED',"
            "'2026-06-18',120.0,400.0,1.0,5.0,'put',100.0,NULL,"
            "NULL,95.0)")
        con.commit(); con.close()

        con = sqlite3.connect(self.archive)
        con.executescript("""
            CREATE TABLE chain_snapshots (
                symbol TEXT, snap_date TEXT, contract TEXT, type TEXT,
                strike REAL, expiration TEXT, bid REAL, ask REAL,
                spot REAL);
        """)
        for d, b, a in (("2026-06-10", 0.95, 1.05), ("2026-06-12", 0.65, 0.75),
                        ("2026-06-16", 0.45, 0.55), ("2026-06-18", 0.35, 0.45)):
            con.execute("INSERT INTO chain_snapshots VALUES "
                        "('AMD',?,'p',?,100.0,'2026-07-17',?,?,150.0)",
                        (d, "put", b, a))
        # The long leg (strike 95, put): bid=ask=0.0 at every date the short
        # leg has a snapshot for -- a worthless, deep-OTM long wing. Because
        # `short - 0.0 == short`, this keeps every OLD assertion in this file
        # (written when Corpus B read the short leg alone) correct under the
        # new two-leg join, while still exercising the join, the dedup and
        # the crossing-direction arithmetic for real (see the dedicated
        # two-leg tests below for a long leg with a non-zero price).
        for d in ("2026-06-10", "2026-06-12", "2026-06-16", "2026-06-18"):
            con.execute("INSERT INTO chain_snapshots VALUES "
                        "('AMD',?,'p2',?,95.0,'2026-07-17',0.0,0.0,150.0)",
                        (d, "put"))
        con.commit(); con.close()

    def tearDown(self):
        os.unlink(self.ledger)
        os.unlink(self.archive)

    def test_loads_a_trade_with_real_bid_ask_points(self):
        paths, _ = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(len(paths), 1)
        p = paths[0]
        self.assertEqual(p.corpus, "B")
        self.assertEqual(p.symbol, "AMD")
        self.assertEqual(len(p.points), 4)
        self.assertNotEqual(p.points[0].bid, p.points[0].ask)

    def test_null_short_put_strike_with_strike_set_still_loads(self):
        """The exact shape of all 163 real closed Bull Put rows: this is the
        regression test for the join-column correction. Without it, a loader
        that joins on `short_put_strike` alone silently returns zero rows for
        every non-Iron-Condor strategy while appearing to run cleanly."""
        con = sqlite3.connect(self.ledger)
        row = con.execute(
            "SELECT short_put_strike, strike FROM trades WHERE entry_id = 1"
        ).fetchone()
        con.close()
        self.assertIsNone(row[0])
        self.assertEqual(row[1], 100.0)

        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(len(paths), 1)
        self.assertEqual(rep.loaded, 1)

    def test_iron_condor_uses_short_put_strike_not_strike(self):
        """Iron Condor is the one strategy where short_put_strike IS
        populated; `strike` may be unrelated (e.g. the call side) or NULL.
        The coalesce must prefer short_put_strike here, not fall through to
        `strike`."""
        con = sqlite3.connect(self.ledger)
        con.execute(
            "INSERT INTO trades (entry_id, date, ticker, expiration, "
            "strategy_name, status, exit_date, pnl_usd, capital_at_risk, "
            "net_credit, spread_width, type, strike, short_put_strike, "
            "long_put_strike) VALUES "
            "(2,'2026-06-10','AMD','2026-07-17','Iron Condor','CLOSED',"
            "'2026-06-18',80.0,400.0,1.0,5.0,'call,put',NULL,100.0,95.0)")
        con.commit(); con.close()

        paths, rep = load_corpus_b(self.ledger, self.archive, "Iron Condor")
        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].symbol, "AMD")

    def test_long_call_with_null_net_credit_and_entry_price_set_loads(self):
        """The exact shape of all 316 real closed Long Call rows: net_credit
        and spread_width are NULL (populated only for the three spread
        strategies) and the opening price lives in entry_price instead.
        Requiring net_credit unconditionally silently drops every
        single-leg trade -- this is the regression test for that."""
        con = sqlite3.connect(self.ledger)
        con.execute(
            "INSERT INTO trades (entry_id, date, ticker, expiration, "
            "strategy_name, status, exit_date, pnl_usd, capital_at_risk, "
            "net_credit, spread_width, entry_price, type, strike, "
            "short_put_strike, long_put_strike) VALUES "
            "(3,'2026-06-10','AMD','2026-07-17','Long Call','CLOSED',"
            "'2026-06-18',-40.0,300.0,NULL,NULL,3.00,'call',110.0,NULL,NULL)")
        con.commit()
        row = con.execute(
            "SELECT net_credit, spread_width, entry_price FROM trades "
            "WHERE entry_id = 3").fetchone()
        con.close()
        self.assertIsNone(row[0])
        self.assertIsNone(row[1])
        self.assertEqual(row[2], 3.00)

        con = sqlite3.connect(self.archive)
        for d, b, a in (("2026-06-10", 2.90, 3.10),
                        ("2026-06-14", 2.60, 2.80),
                        ("2026-06-18", 2.40, 2.60)):
            con.execute("INSERT INTO chain_snapshots VALUES "
                        "('AMD',?,'c',?,110.0,'2026-07-17',?,?,150.0)",
                        (d, "call", b, a))
        con.commit(); con.close()

        paths, rep = load_corpus_b(self.ledger, self.archive, "Long Call")
        self.assertEqual(len(paths), 1)
        self.assertAlmostEqual(paths[0].entry_price, 3.00)
        self.assertEqual(rep.dropped.get("missing_structure_fields", 0), 0)

    def test_type_filter_excludes_the_call_leg_at_the_same_strike(self):
        """chain_snapshots stores one row per (symbol, expiration, strike,
        snap_date, TYPE), and the large majority of strike-days carry both a
        call and a put row at the same strike. Without a type filter, a Bull
        Put's join interleaves call quotes with put quotes at the same
        strike -- meaningless prices, and the extra rows let a position pass
        the two-snapshot floor on an inflated, wrong count."""
        con = sqlite3.connect(self.archive)
        for d, b, a in (("2026-06-10", 9.00, 9.50), ("2026-06-12", 8.50, 9.00),
                        ("2026-06-16", 8.00, 8.50), ("2026-06-18", 7.50, 8.00)):
            con.execute("INSERT INTO chain_snapshots VALUES "
                        "('AMD',?,'c',?,100.0,'2026-07-17',?,?,150.0)",
                        (d, "call", b, a))
        con.commit(); con.close()

        paths, _ = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(len(paths), 1)
        p = paths[0]
        # 4 distinct snap_dates, not 8 (one per row across both legs).
        self.assertEqual(len(p.points), 4)
        # The put's quotes, not the call's.
        self.assertAlmostEqual(p.points[0].bid, 0.95)
        self.assertAlmostEqual(p.points[0].ask, 1.05)
        self.assertAlmostEqual(p.points[-1].bid, 0.35)
        self.assertAlmostEqual(p.points[-1].ask, 0.45)

    def test_duplicate_row_on_same_snap_date_counts_once(self):
        """A second guard against inflated counts: even a genuine duplicate
        row for the same (symbol, expiration, strike, type, snap_date) must
        contribute one point, not two."""
        con = sqlite3.connect(self.archive)
        con.execute("INSERT INTO chain_snapshots VALUES "
                    "('AMD','2026-06-10','p2','put',100.0,'2026-07-17',"
                    "0.96,1.06,150.0)")
        con.commit(); con.close()

        paths, _ = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(len(paths), 1)
        self.assertEqual(len(paths[0].points), 4)

    def test_capital_at_risk_comes_from_the_ledger(self):
        paths, _ = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertAlmostEqual(paths[0].capital_at_risk, 400.0)

    def test_actual_pnl_frac_is_pnl_over_capital_at_risk(self):
        paths, _ = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertAlmostEqual(paths[0].actual_pnl_frac, 120.0 / 400.0)

    def test_points_are_not_marked_imputed(self):
        """Every chain_snapshots row carries a real two-sided quote, so
        Corpus B points must keep spread_imputed=False — unlike Corpus A,
        there is nothing here to impute."""
        paths, _ = load_corpus_b(self.ledger, self.archive, "Bull Put")
        for point in paths[0].points:
            self.assertFalse(point.spread_imputed)

    def test_zero_pnl_is_not_treated_as_missing(self):
        """A pnl_usd of 0.0 is a real result, not a gap. Bare truthiness on
        it would wrongly drop a break-even trade."""
        con = sqlite3.connect(self.ledger)
        con.execute("UPDATE trades SET pnl_usd = 0.0 WHERE entry_id = 1")
        con.commit(); con.close()
        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(len(paths), 1)
        self.assertAlmostEqual(paths[0].actual_pnl_frac, 0.0)

    def test_zero_bid_is_a_real_price_not_a_gap(self):
        """A $0.00 bid is a price, not a missing quote. Bare truthiness on
        bid would wrongly drop or impute a real deep-OTM quote."""
        con = sqlite3.connect(self.archive)
        con.execute(
            "UPDATE chain_snapshots SET bid = 0.0 WHERE snap_date = '2026-06-18'")
        con.commit(); con.close()
        paths, _ = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(len(paths), 1)
        last = paths[0].points[-1]
        self.assertEqual(last.bid, 0.0)
        self.assertFalse(last.spread_imputed)

    def test_trade_with_one_snapshot_is_dropped_and_counted(self):
        con = sqlite3.connect(self.archive)
        con.execute("DELETE FROM chain_snapshots WHERE snap_date <> '2026-06-10'")
        con.commit(); con.close()
        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(paths, [])
        self.assertGreaterEqual(rep.dropped.get("fewer_than_two_snapshots", 0), 1)

    def test_zero_capital_at_risk_is_dropped_not_divided_by(self):
        con = sqlite3.connect(self.ledger)
        con.execute("UPDATE trades SET capital_at_risk = 0")
        con.commit(); con.close()
        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(paths, [])
        self.assertGreaterEqual(rep.dropped.get("no_capital_at_risk", 0), 1)

    def test_null_capital_at_risk_is_dropped_not_divided_by(self):
        con = sqlite3.connect(self.ledger)
        con.execute("UPDATE trades SET capital_at_risk = NULL")
        con.commit(); con.close()
        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(paths, [])
        self.assertGreaterEqual(rep.dropped.get("no_capital_at_risk", 0), 1)

    def test_gap_too_large_between_snapshots_is_dropped_and_counted(self):
        con = sqlite3.connect(self.archive)
        con.execute(
            "UPDATE chain_snapshots SET snap_date = '2026-06-17' "
            "WHERE snap_date = '2026-06-18'")
        con.commit(); con.close()
        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put",
                                    max_gap_days=1)
        self.assertEqual(paths, [])
        self.assertGreaterEqual(rep.dropped.get("gap_too_large", 0), 1)

    def test_wrong_strategy_name_matches_nothing(self):
        paths, rep = load_corpus_b(self.ledger, self.archive, "Bear Call")
        self.assertEqual(paths, [])
        self.assertEqual(rep.loaded, 0)


class TestCorpusBTwoLegSpread(unittest.TestCase):
    """The defect fix: Corpus B's path must come from the SPREAD (both
    legs), not the short leg alone. `entry_price` for the three spread
    strategies is `net_credit` -- the two-leg price -- so a path built from
    one leg describes a different instrument than the price it is checked
    against, and trips `stop_mult` on noise. See paths.py::load_corpus_b.
    """

    def setUp(self):
        fd, self.ledger = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        fd, self.archive = tempfile.mkstemp(suffix=".db")
        os.close(fd)

        con = sqlite3.connect(self.ledger)
        con.executescript("""
            CREATE TABLE trades (
                entry_id INTEGER PRIMARY KEY, date TEXT, ticker TEXT,
                expiration TEXT, strategy_name TEXT, status TEXT,
                exit_date TEXT, pnl_usd REAL, capital_at_risk REAL,
                net_credit REAL, spread_width REAL, entry_price REAL,
                type TEXT, strike REAL, short_put_strike REAL,
                long_put_strike REAL, long_strike REAL);
        """)
        con.commit(); con.close()

        con = sqlite3.connect(self.archive)
        con.executescript("""
            CREATE TABLE chain_snapshots (
                symbol TEXT, snap_date TEXT, contract TEXT, type TEXT,
                strike REAL, expiration TEXT, bid REAL, ask REAL,
                spot REAL);
        """)
        con.commit(); con.close()

    def tearDown(self):
        os.unlink(self.ledger)
        os.unlink(self.archive)

    def _insert_trade(self, entry_id, ticker, entry_date, exit_date,
                       short_strike, long_strike, net_credit=1.5,
                       spread_width=5.0, capital_at_risk=350.0,
                       pnl_usd=50.0, strategy="Bull Put", opt_type="put"):
        con = sqlite3.connect(self.ledger)
        con.execute(
            "INSERT INTO trades (entry_id, date, ticker, expiration, "
            "strategy_name, status, exit_date, pnl_usd, capital_at_risk, "
            "net_credit, spread_width, type, strike, short_put_strike, "
            "long_put_strike, long_strike) VALUES "
            "(?,?,?,'2026-08-21',?,'CLOSED',?,?,?,?,?,?,?,NULL,NULL,?)",
            (entry_id, entry_date, ticker, strategy, exit_date, pnl_usd,
             capital_at_risk, net_credit, spread_width, opt_type,
             short_strike, long_strike))
        con.commit(); con.close()

    def _insert_snap(self, ticker, snap_date, strike, bid, ask,
                      opt_type="put"):
        con = sqlite3.connect(self.archive)
        con.execute(
            "INSERT INTO chain_snapshots VALUES (?,?,'x',?,?,"
            "'2026-08-21',?,?,150.0)",
            (ticker, snap_date, opt_type, strike, bid, ask))
        con.commit(); con.close()

    def test_mid_is_short_minus_long_and_crossing_uses_the_right_legs(self):
        """Both legs present at the same symbol/expiration/date: `mid` must
        equal `short_mid - long_mid`, and `ask`/`bid` must use the crossing
        combination that matches closing a credit spread (buy back the
        short at its ask, sell the long at its bid) -- NOT
        `short_ask - long_ask`, which is a different, wrong number here."""
        self._insert_trade(1, "NVDA", "2026-07-01", "2026-07-05",
                            short_strike=50.0, long_strike=45.0)
        self._insert_snap("NVDA", "2026-07-01", 50.0, 2.00, 2.20)
        self._insert_snap("NVDA", "2026-07-01", 45.0, 0.80, 1.00)
        self._insert_snap("NVDA", "2026-07-05", 50.0, 1.50, 1.70)
        self._insert_snap("NVDA", "2026-07-05", 45.0, 0.50, 0.70)

        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put")
        self.assertEqual(len(paths), 1)
        p0 = paths[0].points[0]

        self.assertAlmostEqual(p0.mid, 2.10 - 0.90)          # 1.20
        self.assertAlmostEqual(p0.ask, 2.20 - 0.80)           # 1.40: pay the
        # short's ask, take the long's bid.
        self.assertAlmostEqual(p0.bid, 2.00 - 1.00)           # 1.00: take
        # the short's bid, pay the long's ask.

        # Reject the wrong (same-side) combination explicitly: it would
        # produce different, incorrect numbers on this fixture.
        self.assertNotAlmostEqual(p0.ask, 2.20 - 1.00)        # short_ask - long_ask
        self.assertNotAlmostEqual(p0.bid, 2.00 - 0.80)        # short_bid - long_bid
        self.assertFalse(p0.spread_imputed)

    def test_long_leg_missing_on_one_date_is_dropped_and_counted(self):
        """A date present on the short leg but absent on the long leg must
        be dropped from the path and counted, not paired with a stale long
        quote from a different date. A trade left with fewer than two
        surviving common dates is dropped entirely."""
        # MSFT: short leg has 3 dates, long leg is missing the middle one.
        # Two common dates survive, so the trade still loads.
        self._insert_trade(2, "MSFT", "2026-07-01", "2026-07-05",
                            short_strike=50.0, long_strike=45.0)
        self._insert_snap("MSFT", "2026-07-01", 50.0, 2.00, 2.20)
        self._insert_snap("MSFT", "2026-07-03", 50.0, 1.80, 2.00)
        self._insert_snap("MSFT", "2026-07-05", 50.0, 1.50, 1.70)
        self._insert_snap("MSFT", "2026-07-01", 45.0, 0.80, 1.00)
        # (no long-leg snapshot for 2026-07-03 -- the gap under test)
        self._insert_snap("MSFT", "2026-07-05", 45.0, 0.50, 0.70)

        # TSLA: short leg has 2 dates, long leg only has 1 of them. Only one
        # common date survives, so the whole trade is dropped.
        self._insert_trade(3, "TSLA", "2026-07-01", "2026-07-03",
                            short_strike=50.0, long_strike=45.0)
        self._insert_snap("TSLA", "2026-07-01", 50.0, 2.00, 2.20)
        self._insert_snap("TSLA", "2026-07-03", 50.0, 1.80, 2.00)
        self._insert_snap("TSLA", "2026-07-01", 45.0, 0.80, 1.00)
        # (no long-leg snapshot for 2026-07-03 -- TSLA is left with 1 date)

        paths, rep = load_corpus_b(self.ledger, self.archive, "Bull Put")

        self.assertEqual(len(paths), 1)
        self.assertEqual(paths[0].symbol, "MSFT")
        self.assertEqual(len(paths[0].points), 2)
        self.assertEqual({pt.date for pt in paths[0].points},
                          {"2026-07-01", "2026-07-05"})

        # 2 dropped dates counted: MSFT's 07-03 and TSLA's 07-03.
        self.assertGreaterEqual(rep.dropped.get("leg_snapshot_missing", 0), 2)
        # TSLA never made it into a path at all.
        self.assertGreaterEqual(rep.dropped.get("fewer_than_two_snapshots", 0), 1)

    def test_single_leg_strategy_still_loads_from_one_leg_unnetted(self):
        """Long Call, Long Put and Short Put are genuinely one leg. Their
        `entry_price` must still come from `trades.entry_price` (never
        `net_credit`, which is NULL for them), and their path points must be
        the leg's own mid -- NOT netted against any other strike."""
        con = sqlite3.connect(self.ledger)
        con.execute(
            "INSERT INTO trades (entry_id, date, ticker, expiration, "
            "strategy_name, status, exit_date, pnl_usd, capital_at_risk, "
            "net_credit, spread_width, entry_price, type, strike, "
            "short_put_strike, long_put_strike, long_strike) VALUES "
            "(4,'2026-07-01','AAPL','2026-08-21','Long Call','CLOSED',"
            "'2026-07-05',-30.0,300.0,NULL,NULL,3.20,'call',110.0,NULL,"
            "NULL,NULL)")
        con.commit(); con.close()
        self._insert_snap("AAPL", "2026-07-01", 110.0, 3.00, 3.20, "call")
        self._insert_snap("AAPL", "2026-07-05", 110.0, 2.60, 2.80, "call")

        paths, rep = load_corpus_b(self.ledger, self.archive, "Long Call")

        self.assertEqual(len(paths), 1)
        self.assertEqual(rep.dropped.get("missing_structure_fields", 0), 0)
        self.assertEqual(rep.dropped.get("leg_snapshot_missing", 0), 0)
        p = paths[0]
        self.assertAlmostEqual(p.entry_price, 3.20)   # from entry_price,
        # not net_credit (which is NULL here).
        self.assertEqual(len(p.points), 2)
        self.assertAlmostEqual(p.points[0].mid, (3.00 + 3.20) / 2.0)
        self.assertAlmostEqual(p.points[1].mid, (2.60 + 2.80) / 2.0)


if __name__ == "__main__":
    unittest.main()
