import unittest
from src.leverage import universe as U


class TestUniverse(unittest.TestCase):
    def test_default_symbols_include_majors(self):
        syms = U.symbols()
        self.assertIn("BTC", syms)
        self.assertIn("ETH", syms)
        self.assertIn("SOL", syms)

    def test_perp_symbol_mapping(self):
        self.assertEqual(U.perp_symbol("BTC"), "BTCUSDT")
        self.assertEqual(U.perp_symbol("SOL"), "SOLUSDT")

    def test_cost_frac_is_bps_over_10000(self):
        # BTC tightest, SOL widest; all returned as fractions
        self.assertAlmostEqual(U.cost_frac("BTC"), 0.0013, places=6)
        self.assertGreater(U.cost_frac("SOL"), U.cost_frac("BTC"))
        self.assertLess(U.cost_frac("SOL"), 0.01)

    def test_config_override(self):
        cfg = {"leverage_universe": {"BTC": {"symbol": "BTCUSDT", "cost_bps": 20}}}
        u = U.default_universe(config=cfg)
        self.assertEqual(u["BTC"]["cost_bps"], 20)

    def test_unknown_key_raises(self):
        with self.assertRaises(KeyError):
            U.perp_symbol("DOGE")


if __name__ == "__main__":
    unittest.main()
