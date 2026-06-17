import unittest

from src.crypto.volbacktest.data import parse_dvol, parse_funding


class TestParsers(unittest.TestCase):
    def test_parse_dvol_to_date_close(self):
        payload = {"result": {"data": [[1700000000000, 50, 55, 49, 52.5],
                                        [1700086400000, 52, 56, 51, 53.0]]}}
        df = parse_dvol(payload)
        self.assertIn("dvol", df.columns)
        self.assertIn("Date", df.columns)
        self.assertAlmostEqual(df["dvol"].iloc[-1], 53.0)

    def test_parse_funding_rows(self):
        payload = [{"fundingTime": 1700000000000, "fundingRate": "0.0001"},
                   {"fundingTime": 1700028800000, "fundingRate": "-0.00005"}]
        df = parse_funding(payload)
        self.assertEqual(len(df), 2)
        self.assertAlmostEqual(df["funding_rate"].iloc[0], 0.0001)


if __name__ == "__main__":
    unittest.main()
