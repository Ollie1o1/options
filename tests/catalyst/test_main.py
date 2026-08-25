"""CLI wiring. Every network boundary is mocked."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import src.formatting as fmt
from src.catalyst import __main__ as cli
from src.catalyst.models import Trial

fmt._COLOR_ENABLED = False


def trials():
    def t(nct, sponsor, date, phase="PHASE3"):
        return Trial(nct_id=nct, sponsor_name=sponsor, brief_title="A Study",
                     phase=phase, event_date=date, date_precision="day",
                     date_type="ESTIMATED", status="RECRUITING",
                     enrollment=400, allocation="RANDOMIZED",
                     masking="DOUBLE", primary_outcome="OS",
                     conditions=("Breast Cancer",))
    return [
        t("NCT1", "Annexon, Inc.", "2026-10-31"),
        t("NCT2", "Pfizer", "2026-12-30"),
        t("NCT3", "Qilu Pharmaceutical Co., Ltd.", "2027-03-01"),
    ]


def _patch_all():
    return mock.patch.multiple(
        cli,
        _sweep=mock.DEFAULT, _name_index=mock.DEFAULT,
        _aliases=mock.DEFAULT, _market_caps=mock.DEFAULT,
        _amendments=mock.DEFAULT, _runway=mock.DEFAULT,
        _implied=mock.DEFAULT)


class TestBuildRows(unittest.TestCase):
    def test_drops_unresolved_and_out_of_band_and_counts_both(self):
        with _patch_all() as m:
            m["_sweep"].return_value = trials()
            m["_name_index"].return_value = {"annexon": "ANNX", "pfizer": "PFE"}
            m["_aliases"].return_value = {}
            m["_market_caps"].return_value = {"ANNX": 976_332_558.0,
                                              "PFE": 150_000_000_000.0}
            m["_amendments"].return_value = cli.Amendments()
            m["_runway"].return_value = cli.Runway()
            m["_implied"].return_value = cli.ImpliedMove()
            rows, coverage = cli.build_rows("2026-09-01", "2027-03-01")

        self.assertEqual([r.event.ticker for r in rows], ["ANNX"])
        self.assertEqual(coverage.swept, 3)
        self.assertEqual(coverage.resolved, 2)
        self.assertEqual(coverage.dropped_unresolved, 1)
        self.assertEqual(coverage.dropped_out_of_band, 1)

    def test_deep_tier_failure_increments_the_counter_not_an_exception(self):
        with _patch_all() as m:
            m["_sweep"].return_value = trials()[:1]
            m["_name_index"].return_value = {"annexon": "ANNX"}
            m["_aliases"].return_value = {}
            m["_market_caps"].return_value = {"ANNX": 976_332_558.0}
            m["_amendments"].side_effect = OSError("boom")
            m["_runway"].return_value = cli.Runway()
            m["_implied"].return_value = cli.ImpliedMove()
            rows, coverage = cli.build_rows("2026-09-01", "2027-03-01")

        self.assertEqual(len(rows), 1)
        self.assertGreaterEqual(coverage.deep_failures, 1)

    def test_funded_only_filters_out_underfunded_names(self):
        with _patch_all() as m:
            m["_sweep"].return_value = trials()[:1]
            m["_name_index"].return_value = {"annexon": "ANNX"}
            m["_aliases"].return_value = {}
            m["_market_caps"].return_value = {"ANNX": 976_332_558.0}
            m["_amendments"].return_value = cli.Amendments()
            m["_runway"].return_value = cli.Runway(cash=1.0, funded_through=False)
            m["_implied"].return_value = cli.ImpliedMove()
            rows, _ = cli.build_rows("2026-09-01", "2027-03-01", funded_only=True)
        self.assertEqual(rows, [])


class TestWindow(unittest.TestCase):
    def test_parses_months(self):
        start, end = cli.window("6m", today="2026-08-25")
        self.assertEqual(start, "2026-08-25")
        self.assertEqual(end, "2027-02-21")

    def test_parses_days(self):
        _, end = cli.window("90d", today="2026-08-25")
        self.assertEqual(end, "2026-11-23")

    def test_rejects_garbage(self):
        with self.assertRaises(ValueError):
            cli.window("soon", today="2026-08-25")


class TestMain(unittest.TestCase):
    def test_board_path_writes_events_and_prints(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "catalysts.db")
            with mock.patch.object(cli, "build_rows") as build:
                build.return_value = ([], cli.Coverage(swept=0))
                rc = cli.main(["--window", "6m", "--db", db])
        self.assertEqual(rc, 0)

    def test_mark_path_returns_zero_with_nothing_outstanding(self):
        with tempfile.TemporaryDirectory() as d:
            db = os.path.join(d, "catalysts.db")
            rc = cli.main(["--mark", "--db", db])
        self.assertEqual(rc, 0)

    def test_bad_window_returns_two_not_a_traceback(self):
        with tempfile.TemporaryDirectory() as d:
            rc = cli.main(["--window", "soon",
                           "--db", os.path.join(d, "c.db")])
        self.assertEqual(rc, 2)


class TestDetail(unittest.TestCase):
    def test_shows_every_in_window_event_for_one_ticker(self):
        with _patch_all() as m:
            m["_sweep"].return_value = trials()
            m["_name_index"].return_value = {"annexon": "ANNX", "pfizer": "PFE"}
            m["_aliases"].return_value = {}
            m["_market_caps"].return_value = {"ANNX": 976_332_558.0,
                                              "PFE": 150_000_000_000.0}
            m["_amendments"].return_value = cli.Amendments()
            m["_runway"].return_value = cli.Runway()
            m["_implied"].return_value = cli.ImpliedMove()
            rows, _ = cli.detail_rows("ANNX", "2026-09-01", "2027-03-01")
        self.assertEqual([r.event.ticker for r in rows], ["ANNX"])

    def test_detail_ignores_the_market_cap_band(self):
        # You asked for this name explicitly; the band exists to shorten the
        # board, not to refuse a direct question.
        with _patch_all() as m:
            m["_sweep"].return_value = trials()
            m["_name_index"].return_value = {"pfizer": "PFE"}
            m["_aliases"].return_value = {}
            m["_market_caps"].return_value = {"PFE": 150_000_000_000.0}
            m["_amendments"].return_value = cli.Amendments()
            m["_runway"].return_value = cli.Runway()
            m["_implied"].return_value = cli.ImpliedMove()
            rows, _ = cli.detail_rows("PFE", "2026-09-01", "2027-03-01")
        self.assertEqual(len(rows), 1)

    def test_unknown_ticker_yields_no_rows(self):
        with _patch_all() as m:
            m["_sweep"].return_value = trials()
            m["_name_index"].return_value = {"annexon": "ANNX"}
            m["_aliases"].return_value = {}
            m["_market_caps"].return_value = {"ANNX": 976_332_558.0}
            m["_amendments"].return_value = cli.Amendments()
            m["_runway"].return_value = cli.Runway()
            m["_implied"].return_value = cli.ImpliedMove()
            rows, _ = cli.detail_rows("ZZZZ", "2026-09-01", "2027-03-01")
        self.assertEqual(rows, [])

    def test_main_routes_a_positional_ticker_to_detail(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.object(cli, "detail_rows") as detail:
                detail.return_value = ([], cli.Coverage(swept=0))
                rc = cli.main(["ANNX", "--db", os.path.join(d, "c.db")])
        self.assertEqual(rc, 0)
        detail.assert_called_once()


if __name__ == "__main__":
    unittest.main()
