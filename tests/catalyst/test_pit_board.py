"""Rewinding the whole board to a past vantage date."""
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import pit, pit_cache
from src.catalyst.models import Trial


def a_trial(nct="NCT1", sponsor="Annexon, Inc.", date="2026-10-31"):
    return Trial(nct_id=nct, sponsor_name=sponsor, brief_title="S",
                 phase="PHASE3", event_date=date, date_precision="day",
                 date_type="ESTIMATED", status="RECRUITING", enrollment=400,
                 allocation="RANDOMIZED", masking="QUADRUPLE",
                 primary_outcome="OS", conditions=("Geographic Atrophy",),
                 phases=("PHASE3",))


class BoardCase(unittest.TestCase):
    def setUp(self):
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        self.conn = pit_cache.connect(os.path.join(self._dir.name, "pit.db"))
        self.addCleanup(self.conn.close)


class TestBoardAsOf(BoardCase):
    def test_keeps_only_events_inside_the_forward_horizon(self):
        trials = {"NCT1": a_trial("NCT1", date="2025-03-01"),
                  "NCT2": a_trial("NCT2", date="2030-01-01")}
        with mock.patch.object(pit, "trial_as_of",
                               side_effect=lambda n, d, c: trials.get(n)), \
             mock.patch.object(pit, "_caps", return_value={"ANNX": 1e9}), \
             mock.patch.object(pit, "_index", return_value={"annexon": "ANNX"}), \
             mock.patch.object(pit, "_aliases", return_value={}):
            events, cov = pit.board_as_of("2025-01-01", ["NCT1", "NCT2"],
                                          self.conn, horizon_days=365)
        self.assertEqual([e.trial.nct_id for e in events], ["NCT1"])

    def test_excludes_events_already_in_the_past(self):
        trials = {"NCT1": a_trial("NCT1", date="2024-06-01")}
        with mock.patch.object(pit, "trial_as_of",
                               side_effect=lambda n, d, c: trials.get(n)), \
             mock.patch.object(pit, "_caps", return_value={"ANNX": 1e9}), \
             mock.patch.object(pit, "_index", return_value={"annexon": "ANNX"}), \
             mock.patch.object(pit, "_aliases", return_value={}):
            events, _ = pit.board_as_of("2025-01-01", ["NCT1"], self.conn)
        self.assertEqual(events, [])

    def test_unresolved_sponsor_is_dropped_and_counted(self):
        # Date must sit INSIDE the 365d horizon, or the row is filtered before
        # it ever reaches the resolver and nothing is counted.
        trials = {"NCT1": a_trial("NCT1", sponsor="Qilu Pharmaceutical Co., Ltd.",
                                  date="2025-06-30")}
        with mock.patch.object(pit, "trial_as_of",
                               side_effect=lambda n, d, c: trials.get(n)), \
             mock.patch.object(pit, "_caps", return_value={}), \
             mock.patch.object(pit, "_index", return_value={"annexon": "ANNX"}), \
             mock.patch.object(pit, "_aliases", return_value={}):
            events, cov = pit.board_as_of("2025-01-01", ["NCT1"], self.conn)
        self.assertEqual(events, [])
        self.assertEqual(cov.dropped_unresolved, 1)

    def test_out_of_band_cap_is_dropped_and_counted(self):
        trials = {"NCT1": a_trial("NCT1", date="2025-06-30")}
        with mock.patch.object(pit, "trial_as_of",
                               side_effect=lambda n, d, c: trials.get(n)), \
             mock.patch.object(pit, "_caps", return_value={"ANNX": 5e11}), \
             mock.patch.object(pit, "_index", return_value={"annexon": "ANNX"}), \
             mock.patch.object(pit, "_aliases", return_value={}):
            events, cov = pit.board_as_of("2025-01-01", ["NCT1"], self.conn)
        self.assertEqual(events, [])
        self.assertEqual(cov.dropped_out_of_band, 1)

    def test_swept_counts_only_rows_resolution_is_attempted_on(self):
        # swept and resolved must share a denominator, or the printed
        # percentage mixes two populations.
        trials = {"NCT1": a_trial("NCT1", date="2025-06-30"),
                  "NCT2": a_trial("NCT2", date="2030-01-01")}
        with mock.patch.object(pit, "trial_as_of",
                               side_effect=lambda n, d, c: trials.get(n)), \
             mock.patch.object(pit, "_caps", return_value={"ANNX": 1e9}), \
             mock.patch.object(pit, "_index", return_value={"annexon": "ANNX"}), \
             mock.patch.object(pit, "_aliases", return_value={}):
            _, cov = pit.board_as_of("2025-01-01", ["NCT1", "NCT2"], self.conn)
        self.assertEqual(cov.swept, 1)      # NCT2 is outside the horizon
        self.assertEqual(cov.resolved, 1)

    def test_a_trial_not_yet_registered_is_simply_absent(self):
        with mock.patch.object(pit, "trial_as_of", return_value=None), \
             mock.patch.object(pit, "_caps", return_value={}), \
             mock.patch.object(pit, "_index", return_value={}), \
             mock.patch.object(pit, "_aliases", return_value={}):
            events, cov = pit.board_as_of("2020-01-01", ["NCT1"], self.conn)
        self.assertEqual(events, [])
        self.assertEqual(cov.swept, 0)


if __name__ == "__main__":
    unittest.main()
