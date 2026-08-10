"""Tests for src/optionsdx.py — the loader that breaks the DTE 10-67 wall.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_optionsdx -v

The fixture reproduces the real optionsDX EOD export: bracketed, space-padded
headers, one row carrying BOTH the call and the put, and `C_SIZE`/`P_SIZE` as
"bid x ask" depth strings.

The loader is being written before the real files are in hand, so these tests
carry the format contract. If a downloaded file disagrees with them, the file
is right and this is wrong — `parse_rows` raises on a missing required column
rather than guessing, so that disagreement surfaces loudly instead of loading
a month of mis-parsed rows.
"""
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from src import optionsdx


HEADER = ("[QUOTE_UNIXTIME], [QUOTE_READTIME], [QUOTE_DATE], [QUOTE_TIME_HOURS], "
          "[UNDERLYING_LAST], [EXPIRE_DATE], [EXPIRE_UNIX], [DTE], "
          "[C_DELTA], [C_GAMMA], [C_VEGA], [C_THETA], [C_RHO], [C_IV], "
          "[C_VOLUME], [C_LAST], [C_SIZE], [C_BID], [C_ASK], [STRIKE], "
          "[P_BID], [P_ASK], [P_SIZE], [P_LAST], [P_DELTA], [P_GAMMA], "
          "[P_VEGA], [P_THETA], [P_RHO], [P_IV], [P_VOLUME], "
          "[STRIKE_DISTANCE], [STRIKE_DISTANCE_PCT]")

# One quote date, one expiry ~90 DTE (past the Dolt cache's 67-day ceiling),
# two strikes.
ROW_1 = ("1672876800, 2023-01-05 16:00, 2023-01-05, 16.0, 3808.10, "
         "2023-04-06, 1680811200, 91.0, "
         "0.5512, 0.0021, 4.9100, -0.4210, 2.1100, 0.1820, "
         "1250, 118.50, 12 x 30, 117.90, 119.10, 3800.0, "
         "104.20, 105.60, 45 x 8, 104.90, -0.4488, 0.0021, "
         "4.9100, -0.3900, -1.9800, 0.1901, 980, 8.10, 0.0021")
ROW_2 = ("1672876800, 2023-01-05 16:00, 2023-01-05, 16.0, 3808.10, "
         "2023-04-06, 1680811200, 91.0, "
         "0.4210, 0.0022, 4.8800, -0.4300, 1.9900, 0.1795, "
         "800, 92.40, 5 x 5, 91.80, 93.00, 3850.0, "
         "128.00, 129.40, 20 x 22, 128.70, -0.5790, 0.0022, "
         "4.8800, -0.4010, -2.2000, 0.1877, 610, 41.90, 0.0110")


def _write_csv(path, rows=(ROW_1, ROW_2), header=HEADER):
    Path(path).write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return path


class HeaderNormalisationTest(unittest.TestCase):
    """Real exports bracket and pad their headers."""

    def test_brackets_and_padding_are_stripped(self):
        for raw in ("[C_BID]", " [C_BID] ", "C_BID", "c_bid"):
            self.assertEqual(optionsdx._norm_header(raw), "c_bid")

    def test_a_non_optionsdx_file_is_refused_loudly(self):
        with self.assertRaises(ValueError) as ctx:
            list(optionsdx.parse_rows(["symbol,close\n", "SPY,400\n"], "SPY"))
        self.assertIn("not an optionsDX chain export", str(ctx.exception))


class SizeParsingTest(unittest.TestCase):
    """`C_SIZE` is the column H2 depends on; a silent mis-parse answers the
    liquidity question with noise."""

    def test_depth_splits_into_bid_and_ask(self):
        self.assertEqual(optionsdx._sizes("12 x 30"), (12.0, 30.0))

    def test_spacing_and_case_are_tolerated(self):
        for raw in ("12x30", " 12 X 30 ", "12  x  30"):
            self.assertEqual(optionsdx._sizes(raw), (12.0, 30.0))

    def test_an_unparseable_size_is_none_not_zero(self):
        """Zero depth is a claim about the market; None is 'not recorded'."""
        for raw in ("", None, "n/a", "12"):
            self.assertEqual(optionsdx._sizes(raw), (None, None))


class RowExpansionTest(unittest.TestCase):
    """Each input row carries a call and a put."""

    def setUp(self):
        self.rows = list(optionsdx.parse_rows(
            [HEADER, ROW_1, ROW_2], "SPX"))

    def test_two_input_rows_become_four_contracts(self):
        self.assertEqual(len(self.rows), 4)

    def test_both_sides_are_emitted(self):
        self.assertEqual({r["type"] for r in self.rows}, {"call", "put"})

    def test_call_fields_come_from_the_c_columns(self):
        call = next(r for r in self.rows
                    if r["type"] == "call" and r["strike"] == 3800.0)
        self.assertAlmostEqual(call["bid"], 117.90)
        self.assertAlmostEqual(call["ask"], 119.10)
        self.assertAlmostEqual(call["delta"], 0.5512)
        self.assertAlmostEqual(call["iv"], 0.1820)
        self.assertEqual(call["volume"], 1250)
        self.assertEqual((call["bid_size"], call["ask_size"]), (12.0, 30.0))

    def test_put_fields_come_from_the_p_columns(self):
        put = next(r for r in self.rows
                   if r["type"] == "put" and r["strike"] == 3800.0)
        self.assertAlmostEqual(put["bid"], 104.20)
        self.assertAlmostEqual(put["delta"], -0.4488)
        self.assertEqual((put["bid_size"], put["ask_size"]), (45.0, 8.0))

    def test_mid_is_derived_not_read(self):
        call = next(r for r in self.rows
                    if r["type"] == "call" and r["strike"] == 3800.0)
        self.assertAlmostEqual(call["mid"], (117.90 + 119.10) / 2)

    def test_shared_columns_are_carried_onto_both_sides(self):
        for r in self.rows:
            self.assertEqual(r["date"], "2023-01-05")
            self.assertEqual(r["expiration"], "2023-04-06")
            self.assertAlmostEqual(r["underlying"], 3808.10)
            self.assertAlmostEqual(r["dte"], 91.0)

    def test_a_row_without_a_strike_is_rejected_not_guessed(self):
        broken = ROW_1.replace(" 3800.0,", " ,")
        rep = optionsdx.LoadReport()
        rows = list(optionsdx.parse_rows([HEADER, broken], "SPX", rep))
        self.assertEqual(rows, [])
        self.assertEqual(rep.rejected.get("missing date/expiry/strike"), 1)


class CacheTest(unittest.TestCase):

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "odx.db")
        self.csv = _write_csv(os.path.join(self.dir.name, "spx_eod_202301.txt"))

    def tearDown(self):
        self.dir.cleanup()

    def _count(self):
        with sqlite3.connect(self.db) as c:
            return c.execute("select count(*) from odx_chain").fetchone()[0]

    def test_a_file_loads_its_rows(self):
        rep = optionsdx.load_file(self.csv, db_path=self.db)
        self.assertEqual(rep.rows_written, 4)
        self.assertEqual(self._count(), 4)

    def test_loading_twice_is_a_no_op(self):
        """Backfills in this repo are always resumable."""
        optionsdx.load_file(self.csv, db_path=self.db)
        rep = optionsdx.load_file(self.csv, db_path=self.db)
        self.assertEqual(rep.rows_written, 0)
        self.assertIn("spx_eod_202301.txt", rep.skipped_files)
        self.assertEqual(self._count(), 4)

    def test_the_symbol_is_taken_from_the_filename(self):
        optionsdx.load_file(self.csv, db_path=self.db)
        with sqlite3.connect(self.db) as c:
            self.assertEqual(
                c.execute("select distinct symbol from odx_chain").fetchone()[0],
                "SPX")

    def test_an_unusable_filename_refuses_rather_than_guessing(self):
        odd = _write_csv(os.path.join(self.dir.name, "12345678.csv"))
        with self.assertRaises(ValueError):
            optionsdx.load_file(odd, db_path=self.db)

    def test_an_explicit_symbol_overrides_the_filename(self):
        odd = _write_csv(os.path.join(self.dir.name, "12345678.csv"))
        optionsdx.load_file(odd, db_path=self.db, symbol="QQQ")
        with sqlite3.connect(self.db) as c:
            self.assertEqual(
                c.execute("select distinct symbol from odx_chain").fetchone()[0],
                "QQQ")

    def test_the_cache_is_separate_from_the_dolt_cache(self):
        """Blending sources with different DTE coverage would make tenor look
        like signal. The table name is the guard."""
        optionsdx.load_file(self.csv, db_path=self.db)
        with sqlite3.connect(self.db) as c:
            tables = {r[0] for r in c.execute(
                "select name from sqlite_master where type='table'")}
        self.assertIn("odx_chain", tables)
        self.assertNotIn("dolt_chain", tables)


class CoverageTest(unittest.TestCase):
    """The load is only useful if it actually clears the two walls."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.dir.name, "odx.db")
        optionsdx.load_file(
            _write_csv(os.path.join(self.dir.name, "spx_eod_202301.txt")),
            db_path=self.db)

    def tearDown(self):
        self.dir.cleanup()

    def test_coverage_reports_the_dte_reach(self):
        cov = optionsdx.coverage(self.db)[0]
        self.assertEqual(cov["symbol"], "SPX")
        self.assertEqual(cov["n"], 4)
        self.assertAlmostEqual(cov["max_dte"], 91.0)

    def test_the_dte_ceiling_exceeds_the_dolt_wall(self):
        """67 days is the Dolt cache's ceiling and the reason for this loader."""
        self.assertGreater(optionsdx.coverage(self.db)[0]["max_dte"], 67)

    def test_expiries_per_day_is_measurable(self):
        self.assertEqual(optionsdx.expiries_per_day(self.db, "SPX"), 1.0)

    def test_coverage_on_an_empty_cache_is_empty_not_an_error(self):
        empty = os.path.join(self.dir.name, "empty.db")
        self.assertEqual(optionsdx.coverage(empty), [])


if __name__ == "__main__":
    unittest.main()
