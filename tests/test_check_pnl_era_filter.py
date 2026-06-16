"""Portfolio viewer era filter: show only finalized (today-onward) or pre-data."""
import unittest

from src import check_pnl


def _rows():
    return [
        {"entry_id": 1, "era": "pre_data", "status": "CLOSED"},
        {"entry_id": 2, "era": "finalized", "status": "OPEN"},
        {"entry_id": 3, "status": "OPEN"},  # missing era → treated as pre_data
    ]


class EraFilterTest(unittest.TestCase):
    def test_finalized_only(self):
        out = check_pnl._filter_by_era(_rows(), "finalized")
        self.assertEqual([r["entry_id"] for r in out], [2])

    def test_pre_data_includes_missing_era(self):
        out = check_pnl._filter_by_era(_rows(), "pre_data")
        self.assertEqual(sorted(r["entry_id"] for r in out), [1, 3])

    def test_none_returns_all(self):
        self.assertEqual(len(check_pnl._filter_by_era(_rows(), None)), 3)

    def test_unknown_era_returns_all(self):
        self.assertEqual(len(check_pnl._filter_by_era(_rows(), "bogus")), 3)


if __name__ == "__main__":
    unittest.main()
