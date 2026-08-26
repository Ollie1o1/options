"""The prereg guard. Changing a hypothesis must break the hash."""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.catalyst.backtest import prereg


class TestHypotheses(unittest.TestCase):
    def test_exactly_one_primary(self):
        primaries = [h for h in prereg.HYPOTHESES if h.primary]
        self.assertEqual(len(primaries), 1)

    def test_the_primary_is_funded_through_at_six_months(self):
        h = [h for h in prereg.HYPOTHESES if h.primary][0]
        self.assertEqual(h.key, "H1")
        self.assertEqual(h.horizon_months, 6)
        self.assertIn("funded", h.statement.lower())

    def test_every_secondary_is_labelled_exploratory(self):
        for h in prereg.HYPOTHESES:
            if not h.primary:
                self.assertTrue(h.exploratory)


class TestRenderAndVerify(unittest.TestCase):
    def test_render_names_every_hypothesis(self):
        text = prereg.render()
        for h in prereg.HYPOTHESES:
            self.assertIn(h.key, text)

    def test_render_marks_exploratory_ones_in_the_text(self):
        self.assertIn("EXPLORATORY", prereg.render().upper())

    def test_write_then_verify_passes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "PREREG.md")
            prereg.write(p)
            self.assertTrue(prereg.verify(p))

    def test_a_tampered_file_fails_verification(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "PREREG.md")
            prereg.write(p)
            with open(p, "a") as f:
                f.write("\nH5: something I thought of afterwards\n")
            self.assertFalse(prereg.verify(p))

    def test_a_missing_file_fails_verification(self):
        self.assertFalse(prereg.verify("/nonexistent/PREREG.md"))

    def test_hash_of_a_missing_file_is_none_not_empty_string(self):
        self.assertIsNone(prereg.file_hash("/nonexistent/PREREG.md"))


if __name__ == "__main__":
    unittest.main()
