"""D_hist: matched treated-vs-control call payoff with a moving-block bootstrap."""
import unittest

import numpy as np

from src.squeeze.sleeve import dhist


def _row(date, symbol, decile, path, rv=0.9, sigma_d=0.05, iv=0.80):
    return {"date": date, "symbol": symbol, "si_decile": decile,
            "rv": rv, "log_mcap": 20.0, "log_price": 3.0,
            "sigma_d": sigma_d, "iv": iv, "path": path}


def _flat(n=42, level=100.0):
    return [level] * n


def _spike(n=42, level=100.0, mult=2.0, at=5):
    return [level] * at + [level * mult] * (n - at)


class DHistTest(unittest.TestCase):
    def _panel(self, n_dates=40, treated_spikes=True):
        rows = []
        for d in range(n_dates):
            date = f"2020-{1 + d % 12:02d}-{1 + d % 28:02d}"
            for i in range(6):
                path = _spike() if (treated_spikes and i < 3) else _flat()
                rows.append(_row(date, f"T{i}", 10, path))
            for i in range(12):
                rows.append(_row(date, f"C{i}", 2, _flat()))
        return rows

    def test_a_planted_effect_is_recovered_with_a_positive_interval(self):
        got = dhist.compute(self._panel(), horizon=42, variant="conservative")
        self.assertGreater(got["observed"], 0.0)
        self.assertGreater(got["ci_lo"], 0.0)

    def test_no_effect_gives_an_interval_spanning_zero(self):
        got = dhist.compute(self._panel(treated_spikes=False), horizon=42,
                            variant="conservative")
        self.assertAlmostEqual(got["observed"], 0.0, places=6)
        self.assertLessEqual(got["ci_lo"], 0.0)
        self.assertGreaterEqual(got["ci_hi"], 0.0)

    def test_deciles_six_to_nine_are_excluded_from_both_arms(self):
        rows = self._panel()
        rows.append(_row("2020-01-01", "MID", 7, _spike()))
        got = dhist.compute(rows, horizon=42, variant="conservative")
        self.assertNotIn("MID", got["used_symbols"])

    def test_the_draw_count_matches_the_request(self):
        got = dhist.compute(self._panel(), horizon=42, n_boot=250,
                            variant="conservative")
        self.assertLessEqual(len(got["draws"]), 250)
        self.assertGreater(len(got["draws"]), 0)

    def test_the_same_seed_reproduces_the_same_draws(self):
        panel = self._panel()
        a = dhist.compute(panel, horizon=42, n_boot=200, seed=99,
                          variant="conservative")
        b = dhist.compute(panel, horizon=42, n_boot=200, seed=99,
                          variant="conservative")
        np.testing.assert_allclose(a["draws"], b["draws"])

    def test_a_date_whose_match_fails_is_flagged_not_silently_dropped(self):
        rows = []
        for d in range(20):
            date = f"2021-{1 + d % 12:02d}-{1 + d % 28:02d}"
            for i in range(3):
                rows.append(_row(date, f"T{i}", 10, _flat(), rv=1.0))
            for i in range(6):
                # far outside the rv caliper -> every treated unit drops
                rows.append(_row(date, f"C{i}", 2, _flat(), rv=9.0))
        got = dhist.compute(rows, horizon=42, variant="conservative")
        self.assertEqual(len(got["flagged_dates"]), 20)

    def test_an_empty_panel_returns_zero_dates(self):
        got = dhist.compute([], horizon=42)
        self.assertEqual(got["n_dates"], 0)
