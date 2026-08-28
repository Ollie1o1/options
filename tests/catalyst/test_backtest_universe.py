"""The swept universe is pinned, so two runs share one population.

The sweep was the last live input to the study's POPULATION. CT.gov gains and
edits trials, and `universe.market_caps` applies the band with TODAY'S cap, so
re-running silently re-drew the sample — it moved H3's arms 755 -> 736 between
two runs a day apart with no code change. A study whose population shifts
under it cannot be compared with itself.

It is also the slow part: `market_caps` makes one uncached yfinance call per
resolved ticker, serially.
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import pit_cache
from src.catalyst.backtest import __main__ as cli


class UniverseCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = pit_cache.connect(os.path.join(self._dir.name, "pit.db"))
        self.addCleanup(self.conn.close)


class TestUniverseKey(unittest.TestCase):
    def test_a_different_window_is_a_different_universe(self):
        # Reusing one window's pinned population for another would be the
        # worst possible outcome of caching: silently the wrong sample.
        self.assertNotEqual(cli._universe_key("2023-01-01", "2025-10-01"),
                            cli._universe_key("2023-01-01", "2026-10-01"))
        self.assertNotEqual(cli._universe_key("2022-01-01", "2025-10-01"),
                            cli._universe_key("2023-01-01", "2025-10-01"))

    def test_the_same_window_is_the_same_key(self):
        self.assertEqual(cli._universe_key("2023-01-01", "2025-10-01"),
                         cli._universe_key("2023-01-01", "2025-10-01"))


class TestSweepIsPinned(UniverseCase):
    def test_a_pinned_universe_is_reused_without_sweeping(self):
        pit_cache.put_universe(self.conn, cli._universe_key("A", "B"),
                               "2026-08-27", ["NCT1", "NCT2"])
        with mock.patch.object(cli, "_fresh_sweep") as fresh:
            ncts, pinned_at = cli._sweep_ncts("A", "B", conn=self.conn)
        fresh.assert_not_called()
        self.assertEqual(ncts, ["NCT1", "NCT2"])
        self.assertEqual(pinned_at, "2026-08-27")

    def test_a_miss_sweeps_once_and_pins_the_result(self):
        with mock.patch.object(cli, "_fresh_sweep",
                               return_value=["NCT9"]) as fresh:
            ncts, _ = cli._sweep_ncts("A", "B", conn=self.conn,
                                      today="2026-08-27")
        fresh.assert_called_once()
        self.assertEqual(ncts, ["NCT9"])
        # And the NEXT run reuses it rather than sweeping again.
        with mock.patch.object(cli, "_fresh_sweep") as again:
            ncts2, pinned_at = cli._sweep_ncts("A", "B", conn=self.conn)
        again.assert_not_called()
        self.assertEqual(ncts2, ["NCT9"])
        self.assertEqual(pinned_at, "2026-08-27")

    def test_refresh_re_sweeps_and_repins(self):
        pit_cache.put_universe(self.conn, cli._universe_key("A", "B"),
                               "2026-01-01", ["OLD"])
        with mock.patch.object(cli, "_fresh_sweep", return_value=["NEW"]):
            ncts, pinned_at = cli._sweep_ncts("A", "B", conn=self.conn,
                                              refresh=True,
                                              today="2026-08-27")
        self.assertEqual(ncts, ["NEW"])
        self.assertEqual(pinned_at, "2026-08-27")
        self.assertEqual(pit_cache.get_universe(
            self.conn, cli._universe_key("A", "B")), ("2026-08-27", ["NEW"]))

    def test_an_empty_pinned_sweep_is_honoured_not_re_swept(self):
        # [] is a real answer — "swept, nothing matched". Re-sweeping on it
        # would make an empty universe permanently expensive and unstable.
        pit_cache.put_universe(self.conn, cli._universe_key("A", "B"),
                               "2026-08-27", [])
        with mock.patch.object(cli, "_fresh_sweep") as fresh:
            ncts, pinned_at = cli._sweep_ncts("A", "B", conn=self.conn)
        fresh.assert_not_called()
        self.assertEqual(ncts, [])
        self.assertEqual(pinned_at, "2026-08-27")

    def test_without_a_connection_it_still_works_and_does_not_pin(self):
        # The calendar imports this path too; no cache must not mean no sweep.
        with mock.patch.object(cli, "_fresh_sweep", return_value=["NCT1"]):
            ncts, pinned_at = cli._sweep_ncts("A", "B", conn=None)
        self.assertEqual(ncts, ["NCT1"])
        self.assertIsNone(pinned_at)


class TestTheReportSaysWhichUniverseRan(unittest.TestCase):
    def test_it_states_the_pin_date(self):
        from src.catalyst.backtest import report
        import src.formatting as fmt
        fmt._COLOR_ENABLED = False
        out = report.render([], horizon_counts={}, dropped_delisted=0,
                            prereg_ok=True, universe_pinned_at="2026-08-27",
                            universe_n=832)
        self.assertIn("2026-08-27", out)
        self.assertIn("832", out)
        self.assertIn("pinned", out.lower())

    def test_a_fresh_sweep_is_not_described_as_pinned(self):
        from src.catalyst.backtest import report
        import src.formatting as fmt
        fmt._COLOR_ENABLED = False
        out = report.render([], horizon_counts={}, dropped_delisted=0,
                            prereg_ok=True, universe_pinned_at=None,
                            universe_n=832)
        self.assertNotIn("pinned", out.lower())


class TestPricesAreCached(UniverseCase):
    """Per-ticker price fetches dominated the run: one uncached, serial
    yfinance call per ticker, ~270 of them.

    Correctness first: a CLOSED window is safe forever, an OPEN one only for
    the day it was taken. The study asks for `end=today`, so a same-day re-run
    is instant and tomorrow's run refetches — which is what "renews when it
    should" means here.
    """

    def test_a_second_call_the_same_day_does_not_refetch(self):
        with mock.patch.object(cli, "_fetch_prices",
                               return_value={"2026-01-02": 10.0}) as fetch:
            a = cli._prices("ABC", "2025-01-01", "2026-08-27", conn=self.conn,
                            today="2026-08-27")
            b = cli._prices("ABC", "2025-01-01", "2026-08-27", conn=self.conn,
                            today="2026-08-27")
        fetch.assert_called_once()
        self.assertEqual(a, b)

    def test_the_next_day_refetches_an_open_window(self):
        with mock.patch.object(cli, "_fetch_prices",
                               return_value={"2026-01-02": 10.0}):
            cli._prices("ABC", "2025-01-01", "2026-08-27", conn=self.conn,
                        today="2026-08-27")
        with mock.patch.object(cli, "_fetch_prices",
                               return_value={"2026-01-02": 10.0}) as fetch:
            cli._prices("ABC", "2025-01-01", "2026-08-27", conn=self.conn,
                        today="2026-08-28")
        fetch.assert_called_once()

    def test_a_closed_window_survives_the_day_boundary(self):
        with mock.patch.object(cli, "_fetch_prices",
                               return_value={"2026-01-02": 10.0}):
            cli._prices("ABC", "2024-01-01", "2024-12-31", conn=self.conn,
                        today="2026-08-27")
        with mock.patch.object(cli, "_fetch_prices") as fetch:
            got = cli._prices("ABC", "2024-01-01", "2024-12-31",
                              conn=self.conn, today="2027-01-01")
        fetch.assert_not_called()
        self.assertEqual(got, {"2026-01-02": 10.0})

    def test_an_EMPTY_series_is_never_cached(self):
        """`_fetch_prices` returns {} on ANY exception, so an empty series is
        indistinguishable from a rate-limit or a network blip. Caching it
        poisons every later run: on 2026-08-28 a rate-limited run cached 145
        empty series and the NEXT run returned n=0 for every hypothesis — the
        whole study evaluated to nothing, silently.

        This is the `_fetch_chain_quotes` defect exactly: a fetcher that
        swallows its own exceptions cannot report failure, so its caller must
        not treat a falsy result as an answer. Refetching a genuinely dead
        ticker is far cheaper than a study that quietly returns nothing."""
        with mock.patch.object(cli, "_fetch_prices", return_value={}) as fetch:
            cli._prices("ZZZ", "a", "2026-08-27", conn=self.conn,
                        today="2026-08-27")
            cli._prices("ZZZ", "a", "2026-08-27", conn=self.conn,
                        today="2026-08-27")
        self.assertEqual(fetch.call_count, 2)
        self.assertIsNone(pit_cache.get_prices(self.conn, "ZZZ", "a",
                                               "2026-08-27",
                                               today="2026-08-27"))

    def test_a_non_empty_series_is_still_cached(self):
        with mock.patch.object(cli, "_fetch_prices",
                               return_value={"2026-01-02": 10.0}) as fetch:
            cli._prices("ABC", "a", "2026-08-27", conn=self.conn,
                        today="2026-08-27")
            cli._prices("ABC", "a", "2026-08-27", conn=self.conn,
                        today="2026-08-27")
        fetch.assert_called_once()

    def test_without_a_connection_it_still_fetches(self):
        with mock.patch.object(cli, "_fetch_prices",
                               return_value={"2026-01-02": 10.0}) as fetch:
            got = cli._prices("ABC", "a", "b", conn=None)
        fetch.assert_called_once()
        self.assertEqual(got, {"2026-01-02": 10.0})


class TestAFailedRunRefusesInsteadOfReportingZeros(unittest.TestCase):
    """n=0 printed as UNDERPOWERED reads as a finding about the data.

    Observed 2026-08-28: yfinance rate-limited the benchmark fetch, XBI came
    back empty, every XBI-relative outcome was None, and the report printed
    "n = 0 vs 0 ... UNDERPOWERED" for all three hypotheses under a heading
    that looked exactly like a real result. UNDERPOWERED means "we measured
    and the arm was small". Zero observations everywhere means the RUN FAILED,
    and the two must never render the same way.
    """

    def test_an_empty_benchmark_refuses_and_exits_non_zero(self):
        import tempfile as tf
        from src.catalyst.backtest import prereg
        with tf.TemporaryDirectory() as d:
            path = os.path.join(d, "prereg.md")
            prereg.write(path)
            with mock.patch.object(cli, "_prices", return_value={}), \
                    mock.patch.object(cli, "_sweep_ncts",
                                      return_value=([], "2026-08-28")):
                rc = cli.main(["--prereg", path,
                               "--db", os.path.join(d, "pit.db"),
                               "--today", "2026-08-28"])
        self.assertNotEqual(rc, 0)

    def test_the_report_names_a_failed_run_as_a_failure(self):
        from src.catalyst.backtest import report
        from src.catalyst.backtest.study import Result
        import src.formatting as fmt
        fmt._COLOR_ENABLED = False
        zeros = [Result(key="H1", label="x", n_true=0, n_false=0,
                        mean_true=0.0, mean_false=0.0, diff=0.0, ci_lo=0.0,
                        ci_hi=0.0, verdict="UNDERPOWERED")]
        out = report.render(zeros, horizon_counts={}, dropped_delisted=0,
                            prereg_ok=True, run_failed="benchmark unavailable")
        self.assertIn("FAILED", out.upper())
        self.assertIn("benchmark unavailable", out)
        # And it must not present the empty arms as a measurement: no result
        # rows at all. (The prose may still use the word UNDERPOWERED to say
        # what this is NOT, which is why the check is on the rendered rows.)
        self.assertNotIn("n = 0 vs 0", out)
        self.assertNotIn("95% CI", out)
        self.assertNotIn("H1", out)


class TestFetchPricesReportsFailure(UniverseCase):
    """A fetcher that swallows its own exceptions cannot report failure.

    `_fetch_prices` returned {} on ANY exception, which made three separate
    bugs possible in one day: empty series cached as answers, a rate-limited
    benchmark rendering as UNDERPOWERED, and single tickers silently dropping
    out so the sample shrank between runs (1544 vs 1545 rows, and a moved CI).

    THREE outcomes must stay distinguishable:
      * a series            — data
      * an empty series     — we looked, this ticker has no bars (delisted)
      * PriceFetchError     — we could not look
    """

    def test_a_transport_failure_raises_instead_of_returning_empty(self):
        with mock.patch.object(cli, "_history",
                               side_effect=RuntimeError("429 rate limited")):
            with self.assertRaises(cli.PriceFetchError):
                cli._fetch_prices("ABC", "a", "b")

    def test_an_empty_frame_is_an_ANSWER_not_a_failure(self):
        # A genuinely delisted name returns no bars and no exception. That is
        # data, and must not be confused with a source outage.
        with mock.patch.object(cli, "_history", return_value=[]):
            self.assertEqual(cli._fetch_prices("ABC", "a", "b"), {})

    def test_prices_retries_a_failing_fetch(self):
        calls = []

        def flaky(ticker, start, end):
            calls.append(ticker)
            if len(calls) == 1:
                raise cli.PriceFetchError("boom")
            return {"2026-01-02": 10.0}

        with mock.patch.object(cli, "_fetch_prices", side_effect=flaky):
            got = cli._prices("ABC", "a", "2026-08-27", conn=self.conn,
                              today="2026-08-27")
        self.assertEqual(got, {"2026-01-02": 10.0})
        self.assertEqual(len(calls), 2)

    def test_prices_raises_once_the_retry_budget_is_spent(self):
        with mock.patch.object(cli, "_fetch_prices",
                               side_effect=cli.PriceFetchError("boom")) as f:
            with self.assertRaises(cli.PriceFetchError):
                cli._prices("ABC", "a", "2026-08-27", conn=self.conn,
                            today="2026-08-27")
        self.assertEqual(f.call_count, cli._PRICE_ATTEMPTS)

    def test_a_failed_fetch_is_never_cached_as_an_empty_series(self):
        with mock.patch.object(cli, "_fetch_prices",
                               side_effect=cli.PriceFetchError("boom")):
            with self.assertRaises(cli.PriceFetchError):
                cli._prices("ABC", "a", "2026-08-27", conn=self.conn,
                            today="2026-08-27")
        self.assertIsNone(pit_cache.get_prices(self.conn, "ABC", "a",
                                               "2026-08-27",
                                               today="2026-08-27"))


class TestTheRunRefusesWhenPricesFailMaterially(unittest.TestCase):
    """Dropping tickers quietly is how the sample shrank between runs."""

    def test_the_report_names_failed_fetches_separately_from_delistings(self):
        from src.catalyst.backtest import report
        import src.formatting as fmt
        fmt._COLOR_ENABLED = False
        out = report.render([a_result_stub()], horizon_counts={6: 10},
                            dropped_delisted=3, prereg_ok=True,
                            failed_fetches=7)
        # "delisted" and "we could not look" are different claims.
        self.assertIn("7", out)
        self.assertIn("failed", out.lower())

    def test_no_failures_says_nothing(self):
        from src.catalyst.backtest import report
        import src.formatting as fmt
        fmt._COLOR_ENABLED = False
        out = report.render([a_result_stub()], horizon_counts={6: 10},
                            dropped_delisted=3, prereg_ok=True,
                            failed_fetches=0)
        self.assertNotIn("failed price", out.lower())


def a_result_stub():
    from src.catalyst.backtest.study import Result
    return Result(key="H1", label="x", n_true=40, n_false=22, mean_true=0.0,
                  mean_false=0.0, diff=0.0, ci_lo=-0.1, ci_hi=0.1,
                  verdict="NO EVIDENCE")


class TestPartialSeriesAreNeverCached(UniverseCase):
    """A truncated fetch is non-empty, so it slipped past the empty guard.

    Measured 2026-08-28 on two runs against one pinned universe: identical
    TICKER counts (183/62, 99/203, 109/179) but different ROW counts
    (1544->1538, 534->532, 736->732), and H2's difference moved +0.007 ->
    +0.029. Whole tickers were not dropping; individual (ticker, vintage)
    observations were, because throttled responses came back SHORT. Being
    non-empty, those partial series were cached and persisted.

    Raising on exceptions cannot catch this — a truncated response never
    raises. The series has to be checked against the benchmark, which defines
    the trading calendar for the window.
    """

    BENCH_LAST = "2026-08-27"

    def test_a_series_reaching_the_benchmark_is_cached(self):
        full = {"2026-08-26": 9.0, "2026-08-27": 10.0}
        with mock.patch.object(cli, "_fetch_prices", return_value=full):
            cli._prices("ABC", "a", "2026-08-28", conn=self.conn,
                        today="2026-08-28", expect_through=self.BENCH_LAST)
        self.assertIsNotNone(pit_cache.get_prices(
            self.conn, "ABC", "a", "2026-08-28", today="2026-08-28"))

    def test_a_TRUNCATED_series_is_not_cached(self):
        short = {"2026-01-02": 9.0}
        with mock.patch.object(cli, "_fetch_prices", return_value=short) as f:
            first = cli._prices("ABC", "a", "2026-08-28", conn=self.conn,
                                today="2026-08-28",
                                expect_through=self.BENCH_LAST)
            cli._prices("ABC", "a", "2026-08-28", conn=self.conn,
                        today="2026-08-28", expect_through=self.BENCH_LAST)
        self.assertEqual(first, short)      # still returned, not discarded
        self.assertEqual(f.call_count, 2)   # but refetched, never served
        self.assertIsNone(pit_cache.get_prices(
            self.conn, "ABC", "a", "2026-08-28", today="2026-08-28"))

    def test_without_an_expectation_nothing_changes(self):
        short = {"2026-01-02": 9.0}
        with mock.patch.object(cli, "_fetch_prices", return_value=short):
            cli._prices("ABC", "a", "2026-08-28", conn=self.conn,
                        today="2026-08-28")
        self.assertIsNotNone(pit_cache.get_prices(
            self.conn, "ABC", "a", "2026-08-28", today="2026-08-28"))

    def test_is_complete_compares_the_last_bar(self):
        self.assertTrue(cli._is_complete({"2026-08-27": 1.0}, "2026-08-27"))
        self.assertTrue(cli._is_complete({"2026-08-28": 1.0}, "2026-08-27"))
        self.assertFalse(cli._is_complete({"2026-06-01": 1.0}, "2026-08-27"))
        self.assertFalse(cli._is_complete({}, "2026-08-27"))


if __name__ == "__main__":
    unittest.main()
