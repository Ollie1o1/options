"""Parser tests for the ClinicalTrials.gov v2 sweep. No network."""
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import ctgov

FIXTURE = os.path.join(os.path.dirname(__file__), "..", "fixtures", "catalyst",
                       "ctgov_sweep.json")


def payload():
    with open(FIXTURE) as f:
        return json.load(f)


class TestParseStudies(unittest.TestCase):
    def setUp(self):
        self.trials = ctgov.parse_studies(payload())
        self.by_id = {t.nct_id: t for t in self.trials}

    def test_parses_every_study(self):
        self.assertEqual(len(self.trials), 5)

    def test_day_precision_date(self):
        t = self.by_id["NCT06510816"]
        self.assertEqual(t.event_date, "2026-10-31")
        self.assertEqual(t.date_precision, "day")
        self.assertEqual(t.date_type, "ESTIMATED")

    def test_month_precision_date(self):
        t = self.by_id["NCT06880276"]
        self.assertEqual(t.event_date, "2027-03")
        self.assertEqual(t.date_precision, "month")

    def test_actual_date_type_is_preserved(self):
        self.assertEqual(self.by_id["NCT06000003"].date_type, "ACTUAL")

    def test_missing_enrollment_is_none_not_zero(self):
        self.assertIsNone(self.by_id["NCT06000002"].enrollment)
        self.assertIsNone(self.by_id["NCT06000003"].enrollment)

    def test_missing_primary_outcome_is_none(self):
        self.assertIsNone(self.by_id["NCT06000003"].primary_outcome)

    def test_sponsor_name_is_verbatim(self):
        self.assertEqual(self.by_id["NCT06880276"].sponsor_name,
                         "Qilu Pharmaceutical Co., Ltd.")

    def test_conditions_captured_as_tuple(self):
        self.assertEqual(self.by_id["NCT06510816"].conditions,
                         ("Geographic Atrophy", "Macular Degeneration"))

    def test_design_fields(self):
        t = self.by_id["NCT06510816"]
        self.assertEqual(t.allocation, "RANDOMIZED")
        self.assertEqual(t.masking, "QUADRUPLE")

    def test_study_missing_date_is_skipped_not_defaulted(self):
        p = payload()
        del p["studies"][0]["protocolSection"]["statusModule"]["primaryCompletionDateStruct"]
        self.assertEqual(len(ctgov.parse_studies(p)), 4)


class TestSweep(unittest.TestCase):
    def test_follows_page_tokens_and_stops(self):
        first = payload()
        second = {"studies": [], "totalCount": 599}
        with mock.patch.object(ctgov, "_get_json",
                               side_effect=[first, second]) as g:
            trials = ctgov.sweep("2026-09-01", "2027-03-01", phases=("PHASE3",))
        self.assertEqual(len(trials), 5)
        self.assertEqual(g.call_count, 2)

    def test_queries_each_phase_separately(self):
        with mock.patch.object(ctgov, "_get_json",
                               return_value={"studies": []}) as g:
            ctgov.sweep("2026-09-01", "2027-03-01", phases=("PHASE2", "PHASE3"))
        urls = " ".join(str(c) for c in g.call_args_list)
        self.assertIn("PHASE2", urls)
        self.assertIn("PHASE3", urls)

    def test_dedupes_a_trial_returned_under_two_phases(self):
        with mock.patch.object(ctgov, "_get_json",
                               side_effect=[payload(), payload()]):
            trials = ctgov.sweep("2026-09-01", "2027-03-01",
                                 phases=("PHASE2", "PHASE3"), max_pages=1)
        self.assertEqual(len({t.nct_id for t in trials}), len(trials))

    def test_network_failure_returns_empty_not_raise(self):
        with mock.patch.object(ctgov, "_get_json", side_effect=OSError("boom")):
            self.assertEqual(ctgov.sweep("2026-09-01", "2027-03-01"), [])

    def test_max_pages_is_honoured(self):
        with mock.patch.object(ctgov, "_get_json", return_value=payload()) as g:
            ctgov.sweep("2026-09-01", "2027-03-01", phases=("PHASE3",), max_pages=3)
        self.assertEqual(g.call_count, 3)


if __name__ == "__main__":
    unittest.main()
