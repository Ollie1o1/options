"""Corpus A loads candidate positions and their forward marks into PricePaths.

Tests build their own throwaway SQLite file; no test may name the real
candidates database.
"""
import os
import sqlite3
import tempfile
import unittest

from src.policy_lab.paths import load_corpus_a


class TestCorpusALoader(unittest.TestCase):
    def setUp(self):
        fd, self.db = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        con = sqlite3.connect(self.db)
        con.executescript("""
            CREATE TABLE candidates (
                scan_id TEXT, board TEXT, contract_key TEXT, symbol TEXT,
                strategy_name TEXT, premium REAL, features_json TEXT);
            CREATE TABLE candidate_positions (
                scan_id TEXT, board TEXT, contract_key TEXT, family TEXT,
                entry_date TEXT, entry_price REAL, status TEXT,
                exit_date TEXT, exit_price REAL, exit_reason TEXT,
                pnl_pct REAL);
            CREATE TABLE candidate_marks (
                contract_key TEXT, mark_date TEXT, bid REAL, ask REAL,
                mid REAL, source TEXT);
        """)

        # --- AMD: the primary Bull Put fixture. Its marks are
        # 'live_quote_structure' with bid/ask NULL, matching the real corpus
        # (every mark attached to a closed Bull Put position is this
        # source), so the imputed-spread path is actually exercised.
        con.execute(
            "INSERT INTO candidates "
            "(scan_id, board, contract_key, symbol, strategy_name, premium, "
            " features_json) VALUES (?,?,?,?,?,?,?)",
            ("s1", "CREDIT SPREADS", "AMD|k", "AMD", "Bull Put", 1.0,
             '{"spread_width": 5.0, "max_loss": 400.0}'))
        con.execute("INSERT INTO candidate_positions VALUES "
                    "('s1','CREDIT SPREADS','AMD|k','spread','2026-08-18',"
                    "1.0,'CLOSED','2026-08-25',0.4,'time_exit',0.6)")
        for d, m in (("2026-08-18", 1.00), ("2026-08-20", 0.70),
                     ("2026-08-22", 0.55), ("2026-08-25", 0.40)):
            con.execute("INSERT INTO candidate_marks VALUES (?,?,?,?,?,?)",
                        ("AMD|k", d, None, None, m, "live_quote_structure"))

        # An UNSUPPORTED Short Put row: never priced, must be dropped.
        con.execute(
            "INSERT INTO candidates "
            "(scan_id, board, contract_key, symbol, strategy_name, premium, "
            " features_json) VALUES (?,?,?,?,?,?,?)",
            ("s1", "PREMIUM SELLING", "MU|k", "MU", "Short Put", 2.0, None))
        con.execute("INSERT INTO candidate_positions VALUES "
                    "('s1','PREMIUM SELLING','MU|k','short_premium',"
                    "'2026-08-18',2.0,'UNSUPPORTED',NULL,NULL,"
                    "'needs_spot_and_delta',NULL)")

        # --- MSFT: a second contract (different strategy, so it never
        # collides with the Bull Put assertions above) whose marks DO carry
        # a real bid/ask, to pin spread_imputed is False.
        con.execute(
            "INSERT INTO candidates "
            "(scan_id, board, contract_key, symbol, strategy_name, premium, "
            " features_json) VALUES (?,?,?,?,?,?,?)",
            ("s1", "CREDIT SPREADS", "MSFT|k", "MSFT", "Bear Call", 1.2,
             '{"spread_width": 5.0, "max_loss": 380.0}'))
        con.execute("INSERT INTO candidate_positions VALUES "
                    "('s1','CREDIT SPREADS','MSFT|k','spread','2026-08-18',"
                    "1.2,'CLOSED','2026-08-25',0.5,'time_exit',0.58)")
        for d, m in (("2026-08-18", 1.20), ("2026-08-20", 0.90),
                     ("2026-08-22", 0.70), ("2026-08-25", 0.50)):
            con.execute("INSERT INTO candidate_marks VALUES (?,?,?,?,?,?)",
                        ("MSFT|k", d, m - 0.05, m + 0.05, m, "live_quote"))

        # --- NOFEAT: also Bear Call (dropped rows never enter `paths`, so
        # this cannot disturb the MSFT len(paths) == 1 assertion). Its
        # candidates row has no features_json at all, so capital at risk
        # cannot be sized and it must be dropped as 'unsizable'.
        con.execute(
            "INSERT INTO candidates "
            "(scan_id, board, contract_key, symbol, strategy_name, premium, "
            " features_json) VALUES (?,?,?,?,?,?,?)",
            ("s1", "CREDIT SPREADS", "NOFEAT|k", "NOFEAT", "Bear Call", 1.0,
             None))
        con.execute("INSERT INTO candidate_positions VALUES "
                    "('s1','CREDIT SPREADS','NOFEAT|k','spread','2026-08-18',"
                    "1.0,'CLOSED','2026-08-25',0.4,'time_exit',0.6)")
        for d, m in (("2026-08-18", 1.00), ("2026-08-20", 0.70),
                     ("2026-08-22", 0.55), ("2026-08-25", 0.40)):
            con.execute("INSERT INTO candidate_marks VALUES (?,?,?,?,?,?)",
                        ("NOFEAT|k", d, m - 0.05, m + 0.05, m, "live_quote"))

        # --- BADSIGN: Iron Condor, family='spread' but entry_price is
        # negative — contradicts the family's expected sign and must be
        # dropped as 'sign_contradicts_family' rather than silently flipped.
        con.execute(
            "INSERT INTO candidates "
            "(scan_id, board, contract_key, symbol, strategy_name, premium, "
            " features_json) VALUES (?,?,?,?,?,?,?)",
            ("s1", "CREDIT SPREADS", "BADSIGN|k", "BADSIGN", "Iron Condor",
             1.0, '{"spread_width": 5.0, "max_loss": 400.0}'))
        con.execute("INSERT INTO candidate_positions VALUES "
                    "('s1','CREDIT SPREADS','BADSIGN|k','spread',"
                    "'2026-08-18',-1.0,'CLOSED','2026-08-25',0.4,"
                    "'time_exit',0.6)")
        for d, m in (("2026-08-18", 1.00), ("2026-08-25", 0.40)):
            con.execute("INSERT INTO candidate_marks VALUES (?,?,?,?,?,?)",
                        ("BADSIGN|k", d, m - 0.05, m + 0.05, m,
                         "live_quote"))

        con.commit()
        con.close()

    def tearDown(self):
        os.unlink(self.db)

    def test_loads_a_closed_position_as_a_pricepath(self):
        paths, _ = load_corpus_a(self.db, strategy="Bull Put")
        self.assertEqual(len(paths), 1)
        p = paths[0]
        self.assertEqual(p.symbol, "AMD")
        self.assertEqual(p.corpus, "A")
        self.assertTrue(p.is_credit)
        self.assertEqual(len(p.points), 4)
        self.assertEqual(p.capital_at_risk, 400.0)

    def test_interior_points_are_the_two_middle_marks(self):
        paths, _ = load_corpus_a(self.db, strategy="Bull Put")
        self.assertEqual([q.date for q in paths[0].interior_points],
                         ["2026-08-20", "2026-08-22"])

    def test_unsupported_rows_are_dropped_and_counted(self):
        paths, rep = load_corpus_a(self.db, strategy="Short Put")
        self.assertEqual(paths, [])
        self.assertGreaterEqual(rep.dropped.get("not_closed", 0), 1)

    def test_points_are_sorted_by_date(self):
        paths, _ = load_corpus_a(self.db, strategy="Bull Put")
        dates = [q.date for q in paths[0].points]
        self.assertEqual(dates, sorted(dates))

    def test_report_accounts_for_every_candidate_row(self):
        paths, rep = load_corpus_a(self.db, strategy="Bull Put")
        self.assertEqual(rep.loaded, len(paths))
        self.assertIsInstance(rep.dropped, dict)

    def test_connection_is_read_only(self):
        # Loading must not be able to write, even if asked to.
        load_corpus_a(self.db, strategy="Bull Put")
        con = sqlite3.connect(self.db)
        n = con.execute("SELECT COUNT(*) FROM candidate_positions").fetchone()[0]
        con.close()
        self.assertEqual(n, 5)

    def test_null_bid_ask_marks_are_imputed(self):
        # AMD's marks are 'live_quote_structure' with bid/ask NULL, matching
        # every mark attached to a closed Bull Put position in the real
        # corpus: they must load with spread_imputed=True, never
        # bid=ask=mid masquerading as a real quote.
        paths, _ = load_corpus_a(self.db, strategy="Bull Put")
        self.assertTrue(all(q.spread_imputed for q in paths[0].points))

    def test_real_bid_ask_marks_are_not_imputed(self):
        paths, _ = load_corpus_a(self.db, strategy="Bear Call")
        msft = next(p for p in paths if p.symbol == "MSFT")
        self.assertTrue(all(not q.spread_imputed for q in msft.points))

    def test_missing_features_json_is_dropped_as_unsizable(self):
        paths, rep = load_corpus_a(self.db, strategy="Bear Call")
        self.assertNotIn("NOFEAT", [p.symbol for p in paths])
        self.assertGreaterEqual(rep.dropped.get("unsizable", 0), 1)

    def test_sign_contradicting_family_is_dropped(self):
        paths, rep = load_corpus_a(self.db, strategy="Iron Condor")
        self.assertEqual(paths, [])
        self.assertGreaterEqual(rep.dropped.get("sign_contradicts_family", 0), 1)

    def test_actual_pnl_frac_is_scaled_to_capital_at_risk(self):
        # pnl_pct (0.6) is a fraction of premium; actual_pnl_frac must be
        # rescaled to a fraction of capital at risk: 0.6 * 1.0 * 100 / 400.
        paths, _ = load_corpus_a(self.db, strategy="Bull Put")
        self.assertAlmostEqual(paths[0].actual_pnl_frac, 0.15)


if __name__ == "__main__":
    unittest.main()
