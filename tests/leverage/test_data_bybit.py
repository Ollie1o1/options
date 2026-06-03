import unittest
from src.leverage import data as D


class TestBybit(unittest.TestCase):
    def test_bybit_rows_to_frame(self):
        # Bybit returns newest-first: [start, open, high, low, close, volume, turnover]
        rows = [["1780414500000", "67212.7", "67284.6", "67030.4", "67098.7",
                 "1708.097", "1.1e8"],
                ["1780414200000", "67420.4", "67475.6", "67126.2", "67207.9",
                 "5316.7", "3.5e8"]]
        df = D._bybit_to_frame(rows)
        self.assertTrue(df.index.is_monotonic_increasing)  # sorted oldest-first
        self.assertAlmostEqual(df["close"].iloc[0], 67207.9)


if __name__ == "__main__":
    unittest.main()
