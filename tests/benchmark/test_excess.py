"""Tests for per-trade/daily excess and the cumulative dollar comparison.

Every ledger here is a throwaway file built in `setUp`; no test names or
opens the real `paper_trades.db`.
"""
import math
import os
import sqlite3
import tempfile
import unittest

from src.benchmark.excess import (
    NULL_CASH,
    NULL_SPY,
    Trade,
    build_paired_frame,
    concurrent_exposure_daily,
    concurrent_stats,
    cumulative_dollar_comparison,
    daily_aggregate,
    daily_frame,
    load_closed_trades,
    n_needed_for_power,
    paired_aggregate,
)

_TRADES_SCHEMA = """
CREATE TABLE trades (
    entry_id INTEGER PRIMARY KEY, date TEXT, ticker TEXT, status TEXT,
    exit_date TEXT, pnl_usd REAL, capital_at_risk REAL
);
"""


def _make_ledger(rows):
    """rows: (entry_id, date, ticker, status, exit_date, pnl_usd, car)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    con = sqlite3.connect(path)
    con.executescript(_TRADES_SCHEMA)
    con.executemany(
        "INSERT INTO trades (entry_id, date, ticker, status, exit_date, "
        "pnl_usd, capital_at_risk) VALUES (?,?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return path


class TestLoadClosedTrades(unittest.TestCase):
    def setUp(self):
        self.path = _make_ledger([
            (1, "2026-01-01", "AAA", "CLOSED", "2026-01-05", 50.0, 500.0),
            (2, "2026-01-02", "BBB", "OPEN", None, None, 300.0),
            (3, "2026-01-03", "CCC", "CLOSED", None, 10.0, 300.0),        # exit_date missing
            (4, "2026-01-04", "DDD", "CLOSED", "2026-01-06", 5.0, 0.0),   # car zero
            (5, "2026-01-05", "EEE", "CLOSED", "2026-01-07", 5.0, None),  # car missing
            (6, "2026-01-06", "FFF", "CLOSED", "2026-01-08", None, 400.0),  # pnl missing
            (7, "2026-01-07", "GGG", "CLOSED", "2026-01-09", 0.0, 400.0),  # pnl == 0.0, KEEP
        ])

    def tearDown(self):
        os.remove(self.path)

    def test_keeps_only_fully_usable_closed_rows(self):
        trades, report = load_closed_trades(self.path)
        self.assertEqual(report.loaded, 2)
        symbols = {t.symbol for t in trades}
        self.assertEqual(symbols, {"AAA", "GGG"})

    def test_zero_pnl_is_kept_not_dropped(self):
        # Regression guard: `pnl_usd == 0.0` must not be treated as missing
        # (bare truthiness on a legitimate zero is the exact bug class
        # `mark_open` hit three times in this repo).
        trades, _ = load_closed_trades(self.path)
        ggg = [t for t in trades if t.symbol == "GGG"][0]
        self.assertEqual(ggg.pnl_usd, 0.0)

    def test_drop_reasons_counted(self):
        _, report = load_closed_trades(self.path)
        self.assertEqual(report.dropped.get("exit_date_missing"), 1)
        self.assertEqual(report.dropped.get("capital_at_risk_missing_or_zero"), 2)
        self.assertEqual(report.dropped.get("pnl_usd_missing"), 1)


class TestBuildPairedFrame(unittest.TestCase):
    def test_book_ret_and_spy_ret_and_excess(self):
        trades = [Trade("AAA|2026-01-01|1", "AAA", "2026-01-01", "2026-01-05",
                         pnl_usd=50.0, capital_at_risk=500.0)]
        spy = {"2026-01-01": 400.0, "2026-01-05": 404.0}
        frames, match = build_paired_frame(trades, spy)
        self.assertEqual(match.exact, 1)
        self.assertEqual(match.tolerant, 0)
        self.assertEqual(match.missing, 0)

        spy_df = frames[NULL_SPY]
        self.assertEqual(len(spy_df), 1)
        row = spy_df.iloc[0]
        book_ret = 50.0 / 500.0
        spy_ret = 404.0 / 400.0 - 1.0
        self.assertAlmostEqual(row["r_alt"], book_ret)
        self.assertAlmostEqual(row["r_base"], spy_ret)
        self.assertAlmostEqual(row["d"], book_ret - spy_ret)

        cash_df = frames[NULL_CASH]
        row = cash_df.iloc[0]
        self.assertEqual(row["r_base"], 0.0)
        self.assertAlmostEqual(row["d"], book_ret)  # excess vs cash == book_ret

    def test_same_day_round_trip_spy_ret_is_zero_not_missing(self):
        trades = [Trade("AAA|2026-01-01|1", "AAA", "2026-01-01", "2026-01-01",
                         pnl_usd=10.0, capital_at_risk=100.0)]
        spy = {"2026-01-01": 400.0}
        frames, match = build_paired_frame(trades, spy)
        self.assertEqual(match.exact, 1)
        row = frames[NULL_SPY].iloc[0]
        self.assertEqual(row["r_base"], 0.0)  # a real 0.0% SPY move, not dropped

    def test_missing_when_no_spy_price_within_tolerance(self):
        trades = [Trade("AAA|2026-01-01|1", "AAA", "2026-01-01", "2026-06-01",
                         pnl_usd=10.0, capital_at_risk=100.0)]
        spy = {"2026-01-01": 400.0}  # no price anywhere near exit
        frames, match = build_paired_frame(trades, spy, tolerance_days=4)
        self.assertEqual(match.missing, 1)
        self.assertEqual(len(frames[NULL_SPY]), 0)

    def test_tolerant_match_counted(self):
        trades = [Trade("AAA|2026-01-01|1", "AAA", "2026-01-01", "2026-01-05",
                         pnl_usd=10.0, capital_at_risk=100.0)]
        # No exact price on the exit date; nearest is 2 days off, within tol.
        spy = {"2026-01-01": 400.0, "2026-01-07": 410.0}
        frames, match = build_paired_frame(trades, spy, tolerance_days=4)
        self.assertEqual(match.exact, 0)
        self.assertEqual(match.tolerant, 1)

    def test_clustering_unit_is_symbol_and_entry_date(self):
        trades = [Trade("AAA|2026-01-01|1", "AAA", "2026-01-01", "2026-01-05",
                         10.0, 100.0),
                  Trade("AAA|2026-01-01|2", "AAA", "2026-01-01", "2026-01-05",
                         20.0, 200.0)]
        spy = {"2026-01-01": 400.0, "2026-01-05": 404.0}
        frames, _ = build_paired_frame(trades, spy)
        df = frames[NULL_SPY]
        self.assertEqual(df["cluster"].nunique(), 1)  # same symbol|entry_date


class TestPairedAggregate(unittest.TestCase):
    def test_empty_frame(self):
        trades = []
        frames, _ = build_paired_frame(trades, {})
        result = paired_aggregate(frames[NULL_SPY])
        self.assertEqual(result.n_rows, 0)
        self.assertIsNone(result.ci_lo)

    def test_ci_reported_with_enough_clusters(self):
        trades = [
            Trade(f"AAA{i}|2026-01-0{i}|{i}", "AAA", f"2026-01-0{i}",
                  f"2026-01-0{i+1}", pnl_usd=float(i * 10 - 25),
                  capital_at_risk=100.0)
            for i in range(1, 9)
        ]
        spy = {f"2026-01-0{i}": 400.0 + i for i in range(1, 10)}
        frames, _ = build_paired_frame(trades, spy)
        result = paired_aggregate(frames[NULL_SPY], n_boot=200, seed=1)
        self.assertGreaterEqual(result.n_clusters, 2)
        self.assertIsNotNone(result.ci_lo)
        self.assertIsNotNone(result.ci_hi)
        self.assertLessEqual(result.ci_lo, result.ci_hi)

    def test_variance_ratio_is_against_book_not_against_null(self):
        # Regression guard: the ratio must divide by sd(book_ret), not
        # sd(null_ret). A null with near-zero variance (SPY's small window
        # move) must NOT produce a huge, nonsensical ratio the way dividing
        # by policy_lab.stats.variance_reduction's `r_base` would (that
        # function's `r_base` is the NULL here, not the book).
        trades = [
            Trade(f"AAA{i}|2026-01-0{i}|{i}", "AAA", f"2026-01-0{i}",
                  f"2026-01-0{i + 1}",
                  pnl_usd=float((i - 4) * 100),  # highly variable book P&L
                  capital_at_risk=100.0)
            for i in range(1, 9)
        ]
        # SPY barely moves day to day: tiny sd(r_base).
        spy = {f"2026-01-0{i}": 400.0 + 0.01 * i for i in range(1, 10)}
        frames, _ = build_paired_frame(trades, spy)
        result = paired_aggregate(frames[NULL_SPY], n_boot=50, seed=1)
        # sd(book_ret) here is large (pnl swings +-100 on car=100 -> returns
        # of +-1.0), so a ratio against it stays near 1, never blowing up
        # into the tens the way dividing by SPY's near-zero sd would.
        self.assertLess(result.variance_ratio, 5.0)


class TestConcurrentExposure(unittest.TestCase):
    def test_peak_and_mean_hand_computed(self):
        # A: open [01-01, 01-04)  -> days 01,02,03
        # B: open [01-02, 01-03)  -> day 02 only (overlaps A)
        # Peak is day 02: 100 (A) + 50 (B) = 150.
        # Series spans the full range min(entry)..max(exit) = 01-01..01-04
        # (4 days, inclusive): [100, 150, 100, 0] -- 01-04 is A's exit day,
        # excluded from A's own exposure but still part of the series index.
        # mean = 350/4.
        trades = [
            Trade("A|2026-01-01|1", "A", "2026-01-01", "2026-01-04", 0.0, 100.0),
            Trade("B|2026-01-02|2", "B", "2026-01-02", "2026-01-03", 0.0, 50.0),
        ]
        series = concurrent_exposure_daily(trades)
        self.assertEqual(series["2026-01-01"], 100.0)
        self.assertEqual(series["2026-01-02"], 150.0)
        self.assertEqual(series["2026-01-03"], 100.0)
        self.assertEqual(series["2026-01-04"], 0.0)
        stats = concurrent_stats(trades)
        self.assertEqual(stats.peak_deployed, 150.0)
        self.assertAlmostEqual(stats.mean_deployed, 350.0 / 4.0)

    def test_exit_day_excluded(self):
        # A single position open [01-01, 01-02): only 01-01 carries exposure.
        trades = [Trade("A|2026-01-01|1", "A", "2026-01-01", "2026-01-02",
                         0.0, 100.0)]
        series = concurrent_exposure_daily(trades)
        self.assertEqual(list(series.index), ["2026-01-01", "2026-01-02"])
        self.assertEqual(series["2026-01-01"], 100.0)
        self.assertEqual(series["2026-01-02"], 0.0)

    def test_empty_trades(self):
        stats = concurrent_stats([])
        self.assertEqual(stats.mean_deployed, 0.0)
        self.assertEqual(stats.peak_deployed, 0.0)
        self.assertEqual(stats.n_days, 0)


class TestDailyFrame(unittest.TestCase):
    def test_book_ret_split_across_open_and_exit_intervals(self):
        trades = [Trade("A|2026-01-01|1", "A", "2026-01-01", "2026-01-03",
                         pnl_usd=30.0, capital_at_risk=100.0)]
        spy = {"2026-01-01": 400.0, "2026-01-02": 404.0, "2026-01-03": 408.0}
        df = daily_frame(trades, spy)
        # Exposure: 01-01 -> 100, 01-02 -> 100, 01-03 -> 0 (exit-day excluded
        # from A's OWN exposure). Each interval's return is measured against
        # the capital deployed at the START of that interval (prev_d), so
        # both intervals divide by 100 -- including the one ending on the
        # exit day, which is where the realised P&L lands.
        self.assertEqual(len(df), 2)
        row0, row1 = df.iloc[0], df.iloc[1]
        self.assertEqual(row0["trade_date"], "2026-01-02")
        self.assertEqual(row0["book_ret"], 0.0)       # nothing realised yet
        self.assertEqual(row1["trade_date"], "2026-01-03")
        self.assertAlmostEqual(row1["book_ret"], 30.0 / 100.0)  # realised here

    def test_realised_pnl_attributed_to_next_trading_day(self):
        trades = [Trade("A|2026-01-01|1", "A", "2026-01-01", "2026-01-02",
                         pnl_usd=30.0, capital_at_risk=100.0)]
        spy = {"2026-01-01": 400.0, "2026-01-03": 408.0}  # 01-02 not a trading day
        df = daily_frame(trades, spy)
        self.assertEqual(len(df), 1)
        row = df.iloc[0]
        self.assertEqual(row["trade_date"], "2026-01-03")
        self.assertAlmostEqual(row["book_ret"], 30.0 / 100.0)

    def test_empty_trades(self):
        df = daily_frame([], {"2026-01-01": 400.0})
        self.assertEqual(len(df), 0)


class TestDailyAggregate(unittest.TestCase):
    def test_empty(self):
        import pandas as pd
        result = daily_aggregate(pd.DataFrame(columns=["trade_date", "book_ret",
                                                          "spy_ret", "deployed"]))
        self.assertEqual(result.n_days, 0)
        self.assertIsNone(result.t_stat)

    def test_basic_stats(self):
        import pandas as pd
        df = pd.DataFrame({
            "trade_date": ["2026-01-01", "2026-01-02", "2026-01-03"],
            "book_ret": [0.01, -0.02, 0.03],
            "spy_ret": [0.005, 0.002, -0.001],
            "deployed": [100.0, 100.0, 100.0],
        })
        result = daily_aggregate(df)
        self.assertEqual(result.n_days, 3)
        self.assertAlmostEqual(result.mean_book, (0.01 - 0.02 + 0.03) / 3)
        self.assertIsNotNone(result.t_stat)
        self.assertIsNotNone(result.ci_lo)
        self.assertLessEqual(result.ci_lo, result.ci_hi)


class TestCumulativeDollarComparison(unittest.TestCase):
    def test_basic(self):
        trades = [
            Trade("A|2026-01-01|1", "A", "2026-01-01", "2026-01-05", 50.0, 500.0),
            Trade("B|2026-01-02|2", "B", "2026-01-02", "2026-01-06", -20.0, 300.0),
        ]
        spy = {"2026-01-01": 400.0, "2026-01-05": 404.0,
               "2026-01-02": 401.0, "2026-01-06": 405.0}
        cmp = cumulative_dollar_comparison(trades, spy)
        self.assertEqual(cmp.n_trades, 2)
        self.assertAlmostEqual(cmp.book_pnl_usd, 30.0)
        self.assertAlmostEqual(cmp.total_capital_at_risk_usd, 800.0)
        expected_primary = 500.0 * (404.0 / 400.0 - 1.0) + 300.0 * (405.0 / 401.0 - 1.0)
        self.assertAlmostEqual(cmp.spy_equiv_primary_usd, expected_primary)
        self.assertIsNotNone(cmp.spy_span_return)
        self.assertIsNotNone(cmp.spy_equiv_secondary_usd)

    def test_empty_trades(self):
        cmp = cumulative_dollar_comparison([], {})
        self.assertEqual(cmp.n_trades, 0)
        self.assertEqual(cmp.book_pnl_usd, 0.0)
        self.assertIsNone(cmp.spy_span_return)


class TestNNeededForPower(unittest.TestCase):
    def test_zero_effect_is_infinite(self):
        self.assertEqual(n_needed_for_power(0.0, 0.5), float("inf"))

    def test_known_value(self):
        # n = ((1.959964 + 0.841621)^2 * sd^2) / effect^2
        n = n_needed_for_power(effect=0.01, sd=0.5)
        z_sum = 1.959963984540054 + 0.8416212335729143
        expected = (z_sum ** 2) * (0.5 ** 2) / (0.01 ** 2)
        self.assertAlmostEqual(n, expected, places=2)

    def test_larger_sd_needs_more_n(self):
        small = n_needed_for_power(0.01, 0.1)
        large = n_needed_for_power(0.01, 0.5)
        self.assertLess(small, large)


if __name__ == "__main__":
    unittest.main()
