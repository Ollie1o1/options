"""The D_hist report: what it states, and that it re-renders offline."""
import unittest

from src.squeeze.sleeve import dhist_report


def _result(observed, lo, hi, n_dates=40):
    return {"observed": observed, "ci_lo": lo, "ci_hi": hi,
            "n_dates": n_dates, "treat_n": 120, "control_n": 360,
            "flagged_dates": ["2020-01-01"], "used_symbols": ["A", "B"]}


def _invalid_result(n_flagged=198, n_dates=2):
    # The shape run() actually produced: nearly every date flagged, a point
    # estimate from the surviving sliver, and a NaN (refused) interval.
    return {"observed": 0.1011, "ci_lo": float("nan"), "ci_hi": float("nan"),
            "n_dates": n_dates, "treat_n": 4, "control_n": 8,
            "flagged_dates": [f"d{i}" for i in range(n_flagged)],
            "used_symbols": []}


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

    def test_a_majority_flagged_cell_is_ruled_invalid(self):
        got = dhist_report.summarise({(21, "central"): _invalid_result()},
                                     self.STATS)
        self.assertEqual(got["cells"][0]["verdict"], "INVALID")

    def test_a_majority_surviving_cell_is_ruled_valid(self):
        self.assertTrue(all(c["verdict"] == "VALID"
                            for c in self._payload()["cells"]))

    def test_a_flagged_tie_is_not_a_majority(self):
        # The tripwire is a MAJORITY of flagged cycles: flagged must exceed
        # the survivors, not merely equal them.
        got = dhist_report.summarise(
            {(21, "central"): _invalid_result(n_flagged=5, n_dates=5)},
            self.STATS)
        self.assertEqual(got["cells"][0]["verdict"], "VALID")


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


def _selection(coverage=0.42, matched_rv=0.90, dropped_rv=1.60):
    return {"treated_eligible": 1000, "treated_matched": int(1000 * coverage),
            "coverage": coverage, "dates_over_drop_bar": 7,
            "median_drop_rate": 0.58, "n_dropped": 580,
            "matched_mean": {"rv": matched_rv, "ret_5d": 0.18,
                             "log_mcap": 20.1, "log_price": 2.9},
            "dropped_mean": {"rv": dropped_rv, "ret_5d": 0.31,
                             "log_mcap": 19.2, "log_price": 2.1}}


class SelectionSectionTest(unittest.TestCase):
    STATS = {"ungradeable": 0, "short_path": 0, "treated": 1,
             "control": 3, "excluded": 0}

    def _payload(self):
        results = {}
        for h in dhist_report.HORIZONS:
            for v in dhist_report.VARIANTS:
                r = _result(0.10, 0.03, 0.18)
                r["selection"] = _selection()
                results[(h, v)] = r
        return dhist_report.summarise(results, self.STATS)

    def test_selection_is_keyed_by_horizon_not_duplicated_per_variant(self):
        # Matching does not depend on the payoff variant, so two variants of
        # the same horizon must not print two identical selection blocks.
        got = self._payload()
        self.assertEqual(sorted(got["selection"]), ["21", "42"])

    def test_the_report_states_the_estimand_and_its_coverage(self):
        text = dhist_report.render(self._payload())
        self.assertIn("matchable subsample", text)
        self.assertIn("42.0%", text)

    def test_the_dropped_units_covariates_are_printed_beside_the_kept_ones(self):
        text = dhist_report.render(self._payload())
        self.assertIn("dropped treated", text)
        self.assertIn("1.600", text)   # dropped mean rv
        self.assertIn("0.900", text)   # matched mean rv

    def test_the_report_says_the_number_is_not_the_uniform_treated_effect(self):
        text = dhist_report.render(self._payload())
        self.assertIn("not the effect on a uniformly-drawn treated name", text)

    def test_a_payload_without_selection_still_renders(self):
        # Older JSON sidecars carry no selection block; --from must not break.
        results = {(21, "central"): _result(0.05, 0.0, 0.1)}
        payload = dhist_report.summarise(results, self.STATS)
        self.assertEqual(payload["selection"], {})
        self.assertIn("D_hist", dhist_report.render(payload))
