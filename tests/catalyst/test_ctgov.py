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

    def test_single_phase_registration(self):
        t = self.by_id["NCT06510816"]
        self.assertEqual(t.phases, ("PHASE3",))
        self.assertEqual(t.phase, "PHASE3")

    def test_multi_phase_registration_keeps_every_phase(self):
        p = payload()
        p["studies"][0]["protocolSection"]["designModule"]["phases"] = \
            ["PHASE1", "PHASE2"]
        t = {x.nct_id: x for x in ctgov.parse_studies(p)}["NCT06880276"]
        self.assertEqual(t.phases, ("PHASE1", "PHASE2"))

    def test_phase_is_the_lowest_registered_not_merely_the_first(self):
        # The prior must never claim more maturity than the trial has.
        p = payload()
        p["studies"][0]["protocolSection"]["designModule"]["phases"] = \
            ["PHASE2", "PHASE1"]
        t = {x.nct_id: x for x in ctgov.parse_studies(p)}["NCT06880276"]
        self.assertEqual(t.phase, "PHASE1")

    def test_study_missing_date_is_skipped_not_defaulted(self):
        p = payload()
        del p["studies"][0]["protocolSection"]["statusModule"]["primaryCompletionDateStruct"]
        self.assertEqual(len(ctgov.parse_studies(p)), 4)


class TestIsEvent(unittest.TestCase):
    """A long-term extension is not a catalyst.

    Correcting an overstatement made while investigating this: OPEN-LABEL is
    not the same as not-a-catalyst. RGNX's single-arm open-label gene-therapy
    trial in Duchenne carried a 38% implied move — single-arm is standard in
    rare disease and oncology. Only studies that structurally cannot surprise
    are excluded: extensions, rollovers, expanded access.
    """

    def _t(self, title, masking="QUADRUPLE", allocation="RANDOMIZED"):
        from src.catalyst.models import Trial
        return Trial(nct_id="N", sponsor_name="S", brief_title=title,
                     phase="PHASE3", event_date="2026-10-31",
                     date_precision="day", date_type="ESTIMATED",
                     status="RECRUITING", enrollment=100,
                     allocation=allocation, masking=masking,
                     primary_outcome="OS", conditions=(), phases=("PHASE3",))

    def test_a_long_term_extension_is_not_an_event(self):
        self.assertFalse(ctgov.is_event(self._t(
            "Open-label Study to Evaluate Long-term Safety of SPN-812",
            masking="NONE", allocation=None)))

    def test_an_extension_study_is_not_an_event(self):
        self.assertFalse(ctgov.is_event(self._t("An Extension Study of X")))

    def test_a_rollover_is_not_an_event(self):
        self.assertFalse(ctgov.is_event(self._t("Rollover Study for Subjects")))

    def test_expanded_access_is_not_an_event(self):
        self.assertFalse(ctgov.is_event(self._t("Expanded Access Protocol")))

    def test_a_blinded_rct_is_an_event(self):
        self.assertTrue(ctgov.is_event(self._t(
            "Efficacy and Safety of BEM/RZR vs SOF/VEL in Chronic HCV")))

    def test_a_single_arm_open_label_trial_IS_still_an_event(self):
        # The RGNX case. Open-label does not mean non-binary.
        self.assertTrue(ctgov.is_event(self._t(
            "AFFINITY DUCHENNE: RGX-202 Gene Therapy in Participants With DMD",
            masking="NONE", allocation="NON_RANDOMIZED")))

    def test_extension_matching_is_case_insensitive(self):
        self.assertFalse(ctgov.is_event(self._t("LONG-TERM SAFETY EXTENSION")))

    def test_a_missing_title_is_left_as_an_event(self):
        # Unknown is not disqualifying; a false exclusion is worse here.
        self.assertTrue(ctgov.is_event(self._t("")))


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
