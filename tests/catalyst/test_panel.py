"""Vintage list and feature panel."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import pit_cache
from src.catalyst.backtest import panel
from src.catalyst.design import Amendments
from src.catalyst.models import CatalystEvent, Coverage, Trial
from src.catalyst.runway import Runway


def a_trial(nct="NCT1", date="2025-06-30"):
    return Trial(nct_id=nct, sponsor_name="Annexon, Inc.", brief_title="S",
                 phase="PHASE3", event_date=date, date_precision="day",
                 date_type="ESTIMATED", status="RECRUITING", enrollment=400,
                 allocation="RANDOMIZED", masking="QUADRUPLE",
                 primary_outcome="OS", conditions=("Geographic Atrophy",),
                 phases=("PHASE3",))


class TestVintages(unittest.TestCase):
    def test_quarter_starts_across_three_years(self):
        v = panel.vintages("2023-01-01", "2025-10-01")
        self.assertEqual(len(v), 12)
        self.assertEqual(v[0], "2023-01-01")
        self.assertEqual(v[-1], "2025-10-01")

    def test_every_vintage_is_a_quarter_start(self):
        for d in panel.vintages("2023-01-01", "2025-10-01"):
            self.assertIn(d[5:], ("01-01", "04-01", "07-01", "10-01"))

    def test_a_single_quarter_range(self):
        self.assertEqual(panel.vintages("2024-04-01", "2024-04-01"),
                         ["2024-04-01"])


class PanelCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = pit_cache.connect(os.path.join(self._dir.name, "pit.db"))
        self.addCleanup(self.conn.close)

    def _patch(self):
        return mock.patch.multiple(
            panel, _board=mock.DEFAULT, _runway=mock.DEFAULT,
            _amendments=mock.DEFAULT, _cik=mock.DEFAULT)


class TestBuild(PanelCase):
    def test_one_row_per_event_with_features_attached(self):
        ev = CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=1e9)
        with self._patch() as m:
            m["_board"].return_value = ([ev], Coverage(swept=1, resolved=1))
            m["_cik"].return_value = 111
            m["_runway"].return_value = Runway(cash=1e8, funded_through=True)
            m["_amendments"].return_value = Amendments(
                versions=9, outcomes_updated=3, available=True)
            rows, cov = panel.build("2025-01-01", ["NCT1"], self.conn)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r.vintage, "2025-01-01")
        self.assertEqual(r.ticker, "ANNX")
        self.assertTrue(r.funded_through)
        self.assertTrue(r.amended)

    def test_unknown_funded_state_is_none_not_false(self):
        ev = CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=1e9)
        with self._patch() as m:
            m["_board"].return_value = ([ev], Coverage())
            m["_cik"].return_value = 111
            m["_runway"].return_value = Runway()
            m["_amendments"].return_value = Amendments()
            rows, _ = panel.build("2025-01-01", ["NCT1"], self.conn)
        self.assertIsNone(rows[0].funded_through)

    def test_unavailable_amendments_give_none_not_false(self):
        ev = CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=1e9)
        with self._patch() as m:
            m["_board"].return_value = ([ev], Coverage())
            m["_cik"].return_value = 111
            m["_runway"].return_value = Runway(cash=1.0)
            m["_amendments"].return_value = Amendments(available=False)
            rows, _ = panel.build("2025-01-01", ["NCT1"], self.conn)
        self.assertIsNone(rows[0].amended)

    def test_a_ticker_with_no_cik_still_produces_a_row(self):
        ev = CatalystEvent(trial=a_trial(), ticker="ANNX", mcap=1e9)
        with self._patch() as m:
            m["_board"].return_value = ([ev], Coverage())
            m["_cik"].return_value = None
            m["_amendments"].return_value = Amendments()
            rows, _ = panel.build("2025-01-01", ["NCT1"], self.conn)
        self.assertEqual(len(rows), 1)
        self.assertIsNone(rows[0].funded_through)

    def test_phase_uses_the_furthest_along_registration(self):
        trial = Trial(nct_id="NCT1", sponsor_name="Annexon, Inc.",
                      brief_title="S", phase="PHASE2", event_date="2025-06-30",
                      date_precision="day", date_type="ESTIMATED",
                      status="RECRUITING", enrollment=100,
                      allocation="RANDOMIZED", masking="NONE",
                      primary_outcome="OS", conditions=(),
                      phases=("PHASE2", "PHASE3"))
        ev = CatalystEvent(trial=trial, ticker="ANNX", mcap=1e9)
        with self._patch() as m:
            m["_board"].return_value = ([ev], Coverage())
            m["_cik"].return_value = None
            m["_amendments"].return_value = Amendments()
            rows, _ = panel.build("2025-01-01", ["NCT1"], self.conn)
        self.assertEqual(rows[0].phase, "PHASE3")


if __name__ == "__main__":
    unittest.main()
