"""PDUFA extraction from 8-K full text. No network in tests."""
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import pdufa

# Real EDGAR full-text-search hit shape, recorded 2026-08-26.
HIT = {
    "_id": "0001104659-26-090086:hrmy-20260804xex99d1.htm",
    "_source": {
        "display_names": ["Harmony Biosciences Holdings, Inc.  (HRMY)  (CIK 0001802665)"],
        "file_date": "2026-08-04",
        "form": "8-K",
    },
}

# Real sentence from that filing.
TEXT = ("Pitolisant GR NDA Accepted in July; Target PDUFA Date April 1, 2027 "
        "Pitolisant HD On Track for Phase 3 Topline Data in 2027 and Target "
        "PDUFA Date in 2028")


class TestParseHit(unittest.TestCase):
    def test_pulls_ticker_from_display_names(self):
        self.assertEqual(pdufa.parse_hit(HIT).ticker, "HRMY")

    def test_pulls_cik_without_leading_zeros(self):
        self.assertEqual(pdufa.parse_hit(HIT).cik, 1802665)

    def test_builds_the_document_url(self):
        url = pdufa.parse_hit(HIT).doc_url
        self.assertIn("/Archives/edgar/data/1802665/000110465926090086/", url)
        self.assertTrue(url.endswith("hrmy-20260804xex99d1.htm"))

    def test_keeps_the_filing_date(self):
        self.assertEqual(pdufa.parse_hit(HIT).filed, "2026-08-04")

    def test_a_hit_with_no_ticker_is_none(self):
        bad = {"_id": "x:y.htm",
               "_source": {"display_names": ["Some Private Co (CIK 0000000123)"],
                           "file_date": "2026-08-04", "form": "8-K"}}
        self.assertIsNone(pdufa.parse_hit(bad))


class TestExtractDates(unittest.TestCase):
    def test_finds_a_full_date_after_pdufa(self):
        self.assertIn("2027-04-01", pdufa.extract_dates(TEXT))

    def test_ignores_a_year_only_mention(self):
        # "Target PDUFA Date in 2028" is not a date. Inventing 2028-01-01
        # would fabricate a precision the filing does not state.
        self.assertNotIn("2028-01-01", pdufa.extract_dates(TEXT))

    def test_handles_the_goal_date_phrasing(self):
        t = "The FDA has set a PDUFA goal date of December 15, 2026 for the NDA."
        self.assertEqual(pdufa.extract_dates(t), ["2026-12-15"])

    def test_handles_date_of_phrasing(self):
        t = "a PDUFA date of March 3, 2027"
        self.assertEqual(pdufa.extract_dates(t), ["2027-03-03"])

    def test_no_pdufa_mention_yields_nothing(self):
        self.assertEqual(pdufa.extract_dates("An ordinary press release."), [])

    def test_a_date_far_from_the_mention_is_not_captured(self):
        t = "PDUFA " + ("filler " * 60) + "January 5, 2027"
        self.assertEqual(pdufa.extract_dates(t), [])

    def test_deduplicates_repeated_dates(self):
        t = "PDUFA date of May 1, 2027 ... reiterating the PDUFA date of May 1, 2027"
        self.assertEqual(pdufa.extract_dates(t), ["2027-05-01"])


class TestFetchEvents(unittest.TestCase):
    def test_network_failure_returns_empty_not_raise(self):
        with mock.patch.object(pdufa, "_search", side_effect=OSError("boom")):
            self.assertEqual(pdufa.pdufa_events("2026-06-01", "2026-08-26"), [])

    def test_builds_one_event_per_extracted_date(self):
        with mock.patch.object(pdufa, "_search", return_value=[HIT]), \
             mock.patch.object(pdufa, "_document_text", return_value=TEXT):
            events = pdufa.pdufa_events("2026-06-01", "2026-08-26")
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].ticker, "HRMY")
        self.assertEqual(events[0].event_date, "2027-04-01")

    def test_a_document_that_will_not_load_is_skipped(self):
        with mock.patch.object(pdufa, "_search", return_value=[HIT]), \
             mock.patch.object(pdufa, "_document_text", return_value=None):
            self.assertEqual(pdufa.pdufa_events("2026-06-01", "2026-08-26"), [])


if __name__ == "__main__":
    unittest.main()
