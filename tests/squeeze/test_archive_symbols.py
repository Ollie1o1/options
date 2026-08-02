"""Per-run archive symbol list: config plus an explicit extra cohort."""
import unittest

from src import chain_archive


class SymbolsForRunTest(unittest.TestCase):
    CFG = {"data_archive": {"symbols": ["SPY", "AAPL"]}}

    def test_config_only_is_unchanged(self):
        self.assertEqual(chain_archive.symbols_for_run(self.CFG), ["SPY", "AAPL"])

    def test_extra_symbols_are_appended(self):
        got = chain_archive.symbols_for_run(self.CFG, ["NBIS", "SMCI"])
        self.assertEqual(got, ["SPY", "AAPL", "NBIS", "SMCI"])

    def test_duplicates_collapse_and_config_order_wins(self):
        got = chain_archive.symbols_for_run(self.CFG, ["aapl", "NBIS", "NBIS"])
        self.assertEqual(got, ["SPY", "AAPL", "NBIS"])

    def test_blank_entries_are_dropped(self):
        got = chain_archive.symbols_for_run(self.CFG, ["", "  ", "NBIS"])
        self.assertEqual(got, ["SPY", "AAPL", "NBIS"])

    def test_a_missing_config_section_still_returns_the_extras(self):
        self.assertEqual(chain_archive.symbols_for_run({}, ["NBIS"]), ["NBIS"])
