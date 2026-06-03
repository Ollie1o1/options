import unittest
from src.leverage.__main__ import _coerce_equity


class TestCoerceEquity(unittest.TestCase):
    def test_valid_number(self):
        val, err = _coerce_equity("2500", 1500.0)
        self.assertEqual(val, 2500.0)
        self.assertIsNone(err)

    def test_non_number_falls_back(self):
        val, err = _coerce_equity("abc", 1500.0)
        self.assertEqual(val, 1500.0)
        self.assertIsNotNone(err)

    def test_zero_rejected(self):
        val, err = _coerce_equity("0", 1500.0)
        self.assertEqual(val, 1500.0)
        self.assertIn("positive", err.lower())

    def test_negative_rejected(self):
        val, err = _coerce_equity("-500", 1500.0)
        self.assertEqual(val, 1500.0)
        self.assertIn("positive", err.lower())

    def test_implausibly_large_rejected(self):
        val, err = _coerce_equity("1e12", 1500.0)
        self.assertEqual(val, 1500.0)
        self.assertIsNotNone(err)


if __name__ == "__main__":
    unittest.main()
