"""Sponsor-name resolution. Every case here was observed in live CT.gov data."""
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import resolve

# Real SEC company_tickers.json titles, verified 2026-08-25.
SEC_TITLES = {
    "Annexon, Inc.": "ANNX",
    "Sarepta Therapeutics, Inc.": "SRPT",
    "PFIZER INC": "PFE",
    "Arcus Biosciences, Inc.": "RCUS",
    "AbbVie Inc.": "ABBV",
}


def index():
    return resolve.build_index(SEC_TITLES)


class TestNormalize(unittest.TestCase):
    def test_strips_corporate_suffixes(self):
        self.assertEqual(resolve.normalize("Annexon, Inc."), "annexon")
        self.assertEqual(resolve.normalize("PFIZER INC"), "pfizer")
        self.assertEqual(resolve.normalize("Qilu Pharmaceutical Co., Ltd."),
                         "qilu pharmaceutical")

    def test_strips_stacked_suffixes(self):
        self.assertEqual(resolve.normalize("Foo Holdings Group Inc."), "foo")

    def test_keeps_meaningful_words(self):
        self.assertEqual(resolve.normalize("Sarepta Therapeutics, Inc."),
                         "sarepta therapeutics")

    def test_expands_ampersand(self):
        self.assertEqual(resolve.normalize("A & B Inc"), "a and b")

    def test_is_case_and_punctuation_insensitive(self):
        self.assertEqual(resolve.normalize("ARCUS BIOSCIENCES, INC."),
                         resolve.normalize("Arcus Biosciences, Inc."))


class TestResolve(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(resolve.resolve("Annexon, Inc.", index(), {}), "ANNX")

    def test_suffix_normalised_match(self):
        # CT.gov says "Pfizer"; SEC says "PFIZER INC".
        self.assertEqual(resolve.resolve("Pfizer", index(), {}), "PFE")

    def test_private_sponsor_resolves_to_none(self):
        self.assertIsNone(resolve.resolve("Qilu Pharmaceutical Co., Ltd.",
                                          index(), {}))

    def test_foreign_sponsor_resolves_to_none(self):
        self.assertIsNone(resolve.resolve("NaviFUS Corporation", index(), {}))

    def test_subsidiary_resolves_only_via_alias(self):
        self.assertIsNone(resolve.resolve("Acerta Pharma BV", index(), {}))
        aliases = {"acerta pharma": "AZN"}
        self.assertEqual(resolve.resolve("Acerta Pharma BV", index(), aliases), "AZN")

    def test_alias_beats_index(self):
        aliases = {"annexon": "XXXX"}
        self.assertEqual(resolve.resolve("Annexon, Inc.", index(), aliases), "XXXX")

    def test_empty_sponsor_is_none(self):
        self.assertIsNone(resolve.resolve("", index(), {}))


class TestBuildIndex(unittest.TestCase):
    def test_ambiguous_normalised_name_is_dropped_not_guessed(self):
        titles = {"Acme Inc": "AAA", "ACME CORP": "BBB"}
        idx = resolve.build_index(titles)
        self.assertNotIn("acme", idx)


class TestLoadAliases(unittest.TestCase):
    def test_reads_and_normalises_keys(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "a.json")
            with open(path, "w") as f:
                json.dump({"Acerta Pharma BV": "AZN"}, f)
            self.assertEqual(resolve.load_aliases(path), {"acerta pharma": "AZN"})

    def test_missing_file_is_empty_not_error(self):
        self.assertEqual(resolve.load_aliases("/nonexistent/nope.json"), {})


class TestNameIndex(unittest.TestCase):
    def test_network_failure_returns_empty(self):
        with mock.patch.object(resolve, "_fetch_sec_titles",
                               side_effect=OSError("boom")):
            self.assertEqual(resolve.name_index(cache_path="/nonexistent/x.json"), {})


if __name__ == "__main__":
    unittest.main()
