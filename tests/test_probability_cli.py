import unittest
import numpy as np
from src.probability_lab.rnd import rnd_from_smile
from src.probability_lab.cli import parse_drift, render_report


class TestCli(unittest.TestCase):
    def test_parse_drift(self):
        self.assertAlmostEqual(parse_drift("+3%"), 0.03)
        self.assertAlmostEqual(parse_drift("-2.5%"), -0.025)
        self.assertAlmostEqual(parse_drift("0"), 0.0)
        self.assertAlmostEqual(parse_drift("5"), 0.05)

    def test_render_report_contains_sections(self):
        S, T, r = 100.0, 0.25, 0.04
        strikes = np.linspace(60, 160, 41)
        market = rnd_from_smile(strikes, np.full_like(strikes, 0.30), T, S, r)
        ctx = {
            "ticker": "TEST", "spot": S, "expiry": "2026-09-18", "dte": 91,
            "r": r, "confidence": {"source": "svi", "fit_quality": 0.99},
            "market": market, "view": market, "drift": 0.0, "vol_mult": 1.0,
            "ranked": [{"name": "Long 100 call", "strikes": "100", "entry": 6.0,
                        "ev_view": 12.0, "pop_view": 0.44, "ev_market": 0.5}],
            "levels": [95, 100, 105],
        }
        out = "\n".join(render_report(ctx))
        self.assertIn("TEST", out)
        self.assertIn("Long 100 call", out)
        self.assertIn("Market", out)
        self.assertIn("E[S_T]", out)


if __name__ == "__main__":
    unittest.main()
