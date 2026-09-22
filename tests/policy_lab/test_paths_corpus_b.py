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
                long_put_strike REAL);
        """)
        # Matches the real ledger's shape for every one of its 163 closed
        # Bull Put rows: short_put_strike is NULL, the short leg lives in
        # the generic `strike` column instead. A loader that joins on
        # short_put_strike alone returns zero rows for this fixture.
        con.execute(
            "INSERT INTO trades (entry_id, date, ticker, expiration, "
            "strategy_name, status, exit_date, pnl_usd, capital_at_risk, "
            "net_credit, spread_width, type, strike, short_put_strike, "
            "long_put_strike) VALUES "
            "(1,'2026-06-10','AMD','2026-07-17','Bull Put','CLOSED',"
            "'2026-06-18',120.0,400.0,1.0,5.0,'put',100.0,NULL,95.0)")
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


if __name__ == "__main__":
    unittest.main()
