"""Phase-transition priors. Pure data, no network."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst import baserates


class TestAreaFor(unittest.TestCase):
    def test_maps_oncology_conditions(self):
        self.assertEqual(baserates.area_for(["Breast Cancer"]), "oncology")
        self.assertEqual(baserates.area_for(["Glioblastoma"]), "oncology")

    def test_maps_ophthalmology(self):
        self.assertEqual(
            baserates.area_for(["Geographic Atrophy", "Macular Degeneration"]),
            "ophthalmology")

    def test_maps_neurology(self):
        self.assertEqual(baserates.area_for(["Duchenne Muscular Dystrophy"]),
                         "neurology")

    def test_unmapped_condition_is_none_not_a_guess(self):
        self.assertIsNone(baserates.area_for(["Atopic Dermatitis Subtype Q"]))

    def test_empty_is_none(self):
        self.assertIsNone(baserates.area_for([]))

    def test_short_substrings_do_not_produce_false_matches(self):
        # "als" once lived in the neurology list and fired on words like these.
        for condition in ["Clinical Trials Registry Cohort", "False Positive Study",
                          "Vital Signals Monitoring"]:
            self.assertIsNone(baserates.area_for([condition]), condition)


class TestPrior(unittest.TestCase):
    def test_area_specific_rate_when_known(self):
        self.assertEqual(baserates.prior("PHASE3", "oncology"),
                         baserates.RATES[("PHASE3", "oncology")])

    def test_falls_back_to_all_areas_when_area_unknown(self):
        self.assertEqual(baserates.prior("PHASE3", None),
                         baserates.RATES[("PHASE3", "all")])

    def test_falls_back_when_the_area_has_no_specific_rate(self):
        self.assertEqual(baserates.prior("PHASE3", "dermatology"),
                         baserates.RATES[("PHASE3", "all")])

    def test_unknown_phase_is_none(self):
        self.assertIsNone(baserates.prior("PHASE1", "oncology"))

    def test_phase2_is_lower_than_phase3(self):
        p2 = baserates.prior("PHASE2", "all")
        p3 = baserates.prior("PHASE3", "all")
        assert p2 is not None and p3 is not None
        self.assertLess(p2, p3)


class TestDescribe(unittest.TestCase):
    def test_always_carries_the_not_this_drug_caveat(self):
        text = baserates.describe("PHASE3", "oncology")
        assert text is not None
        self.assertIn("other drugs", text.lower())

    def test_none_for_an_unknown_phase(self):
        self.assertIsNone(baserates.describe("PHASE1", None))

    def test_names_the_area_it_used(self):
        text = baserates.describe("PHASE3", "oncology")
        assert text is not None
        self.assertIn("oncology", text)

    def test_says_all_areas_when_falling_back(self):
        text = baserates.describe("PHASE3", None)
        assert text is not None
        self.assertIn("all areas", text)


class TestCitation(unittest.TestCase):
    def test_citation_is_present_and_non_empty(self):
        self.assertTrue(baserates.CITATION.strip())


if __name__ == "__main__":
    unittest.main()
