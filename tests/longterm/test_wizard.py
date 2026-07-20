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


class TestBuildCommands(unittest.TestCase):
    def test_add(self):
        self.assertEqual(W.build_add_command("mu", [750.0, 650.0]), "ADD MU 750/650")

    def test_edit(self):
        self.assertEqual(W.build_edit_command("mu", [800.0, 700.0]), "EDIT MU 800/700")

    def test_remove(self):
        self.assertEqual(W.build_remove_command("mu"), "REMOVE MU")

    def test_cash(self):
        self.assertEqual(W.build_cash_command(6000), "CASH 6000")

    def test_fill(self):
        self.assertEqual(W.build_fill_command("mu", 750.0, 2.5, 748.20),
                         "FILL MU 750 2.5 748.2")

    def test_cash_large_number_no_scientific_notation(self):
        self.assertEqual(W.build_cash_command(1234567), "CASH 1234567")

    def test_fill_preserves_precision_with_existing_case(self):
        self.assertEqual(W.build_fill_command("mu", 750.0, 2.5, 748.20),
                         "FILL MU 750 2.5 748.2")


if __name__ == "__main__":
    unittest.main()
