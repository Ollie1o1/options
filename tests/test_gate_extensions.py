"""The extension clock — the half of the bounded EXTEND that was missing.

These tests pin the behaviour that was absent on 2026-08-01: a gate sitting at
EXTEND reprinted "extension 1 of 2" every day forever, because nothing recorded
when the window opened and so nothing could tell when it had elapsed.
"""
import json
import os
import tempfile
import unittest

from src.gate_extensions import (EXTENSION_DAYS, LONG_CALL, SHORT_PREMIUM,
                                 ExtensionState, apply_verdict, describe, load,
                                 resolve, save)


class TestResolve(unittest.TestCase):
    def test_no_window_is_a_no_op(self):
        st = resolve({"extensions_used": 0, "window_opened": None}, "2026-08-01")
        self.assertEqual(st.extensions_used, 0)
        self.assertIsNone(st.window_opened)
        self.assertEqual(st.expired_now, 0)

    def test_open_window_inside_two_weeks_does_not_count(self):
        st = resolve({"extensions_used": 0, "window_opened": "2026-08-01"},
                     "2026-08-10")
        self.assertEqual(st.extensions_used, 0)
        self.assertEqual(st.window_opened, "2026-08-01")
        self.assertEqual(st.days_remaining("2026-08-10"), 5)

    def test_window_counts_when_it_expires_not_when_it_opens(self):
        """The regression this module exists for: the counter must advance."""
        st = resolve({"extensions_used": 0, "window_opened": "2026-08-01"},
                     "2026-08-15")
        self.assertEqual(st.extensions_used, 1)
        self.assertEqual(st.expired_now, 1)
        # Window two starts where window one ended.
        self.assertEqual(st.window_opened, "2026-08-15")

    def test_allowance_exhausts_after_two_windows(self):
        st = resolve({"extensions_used": 0, "window_opened": "2026-08-01"},
                     "2026-08-29")
        self.assertEqual(st.extensions_used, 2)
        self.assertEqual(st.expired_now, 2)
        self.assertIsNone(st.window_opened)

    def test_a_long_gap_cannot_gift_back_missed_extensions(self):
        """Checkpoints are not guaranteed to run — this repo's schedulers have
        been dead since 2026-06-15. Six months of silence must still exhaust
        the allowance, not leave it at one."""
        st = resolve({"extensions_used": 0, "window_opened": "2026-02-01"},
                     "2026-08-01")
        self.assertEqual(st.extensions_used, 2)
        self.assertIsNone(st.window_opened)

    def test_counting_never_exceeds_the_allowance(self):
        st = resolve({"extensions_used": 2, "window_opened": "2026-01-01"},
                     "2026-08-01")
        self.assertEqual(st.extensions_used, 2)
        self.assertIsNone(st.window_opened)

    def test_expiry_is_inclusive_of_the_closing_day(self):
        st = resolve({"extensions_used": 0, "window_opened": "2026-08-01"},
                     "2026-08-14")
        self.assertEqual(st.extensions_used, 0, "day 13 is still inside")
        st = resolve({"extensions_used": 0, "window_opened": "2026-08-01"},
                     "2026-08-15")
        self.assertEqual(st.extensions_used, 1, "day 14 closes it")


class TestApplyVerdict(unittest.TestCase):
    def test_extend_opens_a_window_when_none_is_running(self):
        st = apply_verdict(ExtensionState(0, None), "EXTEND", "2026-08-01")
        self.assertEqual(st.window_opened, "2026-08-01")

    def test_extend_does_not_restart_a_running_window(self):
        """The daily-reprint bug: re-deciding EXTEND must not reset the clock."""
        st = apply_verdict(ExtensionState(0, "2026-08-01"), "EXTEND", "2026-08-09")
        self.assertEqual(st.window_opened, "2026-08-01")

    def test_ready_closes_the_window(self):
        st = apply_verdict(ExtensionState(1, "2026-08-01"), "READY", "2026-08-05")
        self.assertIsNone(st.window_opened)
        self.assertEqual(st.extensions_used, 1)

    def test_stop_closes_the_window(self):
        st = apply_verdict(ExtensionState(1, "2026-08-01"), "STOP", "2026-08-05")
        self.assertIsNone(st.window_opened)

    def test_gathering_closes_the_window(self):
        st = apply_verdict(ExtensionState(0, "2026-08-01"), "GATHERING", "2026-08-05")
        self.assertIsNone(st.window_opened)


class TestRoundTrip(unittest.TestCase):
    def test_two_extensions_then_forced_resolution(self):
        """End to end: the terminal condition fires without anyone editing a
        config field."""
        entry = {"extensions_used": 0, "window_opened": None}
        st = resolve(entry, "2026-08-01")
        st = apply_verdict(st, "EXTEND", "2026-08-01")
        self.assertEqual(st.extensions_used, 0)

        # Two weeks on, still unresolved.
        st = resolve(st.as_dict(), "2026-08-15")
        self.assertEqual(st.extensions_used, 1)
        st = apply_verdict(st, "EXTEND", "2026-08-15")
        self.assertEqual(st.window_opened, "2026-08-15")

        # Two weeks more: the allowance is gone and the decision rule, which
        # reads extensions_used, can no longer return EXTEND.
        st = resolve(st.as_dict(), "2026-08-29")
        self.assertEqual(st.extensions_used, 2)
        self.assertIsNone(st.window_opened)


class TestTerminalConditionActuallyFires(unittest.TestCase):
    """The point of the whole module: an unresolved gate must eventually STOP
    on its own, with nobody editing config.json."""

    def _arm_a(self, used):
        from src.short_premium_gate import decide_arm_a
        # Posterior parked between the bands, where EXTEND lives.
        return decide_arm_a(n_eff=105.0, posterior=0.54, extensions_used=used,
                            median_ror=0.28, capital_weighted_ror=0.19)

    def _v2(self, used):
        from src.phase1_checkpoint import decide_v2
        return decide_v2(n_eff=105.0, ic_rank=0.05, ic_pearson=0.04,
                         extensions_used=used)

    def test_short_premium_arm_a_stops_after_the_allowance_elapses(self):
        entry = {"extensions_used": 0, "window_opened": "2026-08-01"}
        self.assertEqual(self._arm_a(resolve(entry, "2026-08-01").extensions_used)[0],
                         "EXTEND")
        self.assertEqual(self._arm_a(resolve(entry, "2026-08-15").extensions_used)[0],
                         "EXTEND", "one window gone, one left")
        verdict, reason = self._arm_a(resolve(entry, "2026-08-29").extensions_used)
        self.assertEqual(verdict, "STOP")
        self.assertIn("allowance exhausted", reason)

    def test_long_call_v2_stops_after_the_allowance_elapses(self):
        entry = {"extensions_used": 0, "window_opened": "2026-08-01"}
        self.assertEqual(self._v2(resolve(entry, "2026-08-01").extensions_used)[0],
                         "EXTEND")
        verdict, reason = self._v2(resolve(entry, "2026-08-29").extensions_used)
        self.assertEqual(verdict, "STOP")
        self.assertIn("allowance exhausted", reason)

    def test_a_daily_checkpoint_does_not_burn_the_allowance(self):
        """The mirror of the bug: re-running the gate every day for a week must
        leave the allowance untouched. Only the calendar spends it."""
        entry = {"extensions_used": 0, "window_opened": "2026-08-01"}
        for day in range(1, 15):
            st = resolve(entry, f"2026-08-{day:02d}")
            st = apply_verdict(st, "EXTEND", f"2026-08-{day:02d}")
            entry = st.as_dict()
        self.assertEqual(entry["extensions_used"], 0)
        self.assertEqual(entry["window_opened"], "2026-08-01")


class TestPersistence(unittest.TestCase):
    def test_load_seeds_from_config_counters_on_first_run(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate_extensions.json")
            data = load(path, seed={LONG_CALL: 1, SHORT_PREMIUM: 0})
            self.assertEqual(data[LONG_CALL]["extensions_used"], 1)
            self.assertEqual(data[SHORT_PREMIUM]["extensions_used"], 0)
            self.assertIsNone(data[SHORT_PREMIUM]["window_opened"])

    def test_save_then_load_round_trips(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "gate_extensions.json")
            save({LONG_CALL: {"extensions_used": 2, "window_opened": None},
                  SHORT_PREMIUM: {"extensions_used": 1,
                                  "window_opened": "2026-08-01"}}, path)
            data = load(path)
            self.assertEqual(data[SHORT_PREMIUM]["window_opened"], "2026-08-01")
            self.assertEqual(data[LONG_CALL]["extensions_used"], 2)

    def test_save_writes_a_do_not_hand_edit_note(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate_extensions.json")
            save(load(path), path)
            with open(path) as f:
                self.assertIn("do NOT hand-edit", json.load(f)["_note"])

    def test_corrupt_state_file_falls_back_to_seed(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "gate_extensions.json")
            with open(path, "w") as f:
                f.write("{not json")
            data = load(path, seed={SHORT_PREMIUM: 1})
            self.assertEqual(data[SHORT_PREMIUM]["extensions_used"], 1)


class TestDescribe(unittest.TestCase):
    def test_running_window_names_its_closing_date(self):
        line = describe(ExtensionState(0, "2026-08-01"), "2026-08-10")
        self.assertIn("Extension 1 of 2", line)
        self.assertIn("2026-08-15", line)
        self.assertIn("5 days left", line)

    def test_exhausted_allowance_says_the_gate_must_resolve(self):
        self.assertIn("must resolve", describe(ExtensionState(2, None), "2026-08-10"))

    def test_no_window_reports_the_count(self):
        self.assertIn("No extension running",
                      describe(ExtensionState(0, None), "2026-08-10"))

    def test_singular_day_is_not_pluralised(self):
        line = describe(ExtensionState(0, "2026-08-01"), "2026-08-14")
        self.assertIn("1 day left", line)


class TestConstants(unittest.TestCase):
    def test_a_window_is_two_weeks(self):
        self.assertEqual(EXTENSION_DAYS, 14)


if __name__ == "__main__":
    unittest.main()
