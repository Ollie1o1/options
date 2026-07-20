"""Tests for the HOLDINGS guided-menu's pure input helpers."""
import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import wizard as W


class TestParseLevels(unittest.TestCase):
    def test_comma_separated(self):
        self.assertEqual(W.parse_levels("750, 650, 550"), [750.0, 650.0, 550.0])

    def test_slash_separated(self):
        self.assertEqual(W.parse_levels("750/650/550"), [750.0, 650.0, 550.0])

    def test_space_separated(self):
        self.assertEqual(W.parse_levels("750 650 550"), [750.0, 650.0, 550.0])

    def test_sorts_descending_regardless_of_typed_order(self):
        self.assertEqual(W.parse_levels("550, 750, 650"), [750.0, 650.0, 550.0])

    def test_single_level(self):
        self.assertEqual(W.parse_levels("750"), [750.0])

    def test_empty_raises(self):
        with self.assertRaises(ValueError):
            W.parse_levels("")

    def test_garbage_raises(self):
        with self.assertRaises(ValueError):
            W.parse_levels("cheap, cheaper")


if __name__ == "__main__":
    unittest.main()
