"""The D_hist report: what it states, and that it re-renders offline."""
import unittest

from src.squeeze.sleeve import dhist_report


def _result(observed, lo, hi, n_dates=40):
    return {"observed": observed, "ci_lo": lo, "ci_hi": hi,
            "n_dates": n_dates, "treat_n": 120, "control_n": 360,
            "flagged_dates": ["2020-01-01"], "used_symbols": ["A", "B"]}


class SummariseTest(unittest.TestCase):
    STATS = {"ungradeable": 12, "short_path": 3,
             "treated": 120, "control": 360, "excluded": 400}

    def _payload(self, observed=0.05):
        results = {(h, v): _result(observed, observed - 0.1, observed + 0.1)
                   for h in dhist_report.HORIZONS
                   for v in dhist_report.VARIANTS}
        return dhist_report.summarise(results, self.STATS)

    def test_every_horizon_and_variant_reaches_the_payload(self):
        got = self._payload()
        keys = {(c["horizon"], c["variant"]) for c in got["cells"]}
        self.assertEqual(len(keys), 4)
        self.assertIn((42, "conservative"), keys)

    def test_the_stats_are_carried_through(self):
        self.assertEqual(self._payload()["stats"]["ungradeable"], 12)

    def test_flagged_dates_are_counted_not_dumped(self):
        cell = self._payload()["cells"][0]
        self.assertEqual(cell["flagged"], 1)
        self.assertNotIn("flagged_dates", cell)


class RenderTest(unittest.TestCase):
    def _payload(self, observed):
        results = {(h, v): _result(observed, observed - 0.1, observed + 0.1)
                   for h in dhist_report.HORIZONS
                   for v in dhist_report.VARIANTS}
        return dhist_report.summarise(results, {"ungradeable": 0,
                                                "short_path": 0, "treated": 1,
                                                "control": 3, "excluded": 0})

    def test_render_is_a_pure_function_of_the_payload(self):
        payload = self._payload(0.05)
        self.assertEqual(dhist_report.render(payload),
                         dhist_report.render(payload))

    def test_the_report_states_what_the_number_is_not(self):
        text = dhist_report.render(self._payload(0.05))
        self.assertIn("not yet dead", text)
        self.assertIn("P_live", text)

    def test_a_negative_result_is_reported_plainly(self):
        text = dhist_report.render(self._payload(-0.08))
        self.assertIn("-8.00%", text)
