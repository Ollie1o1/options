"""config.json resolves against the repo root, not the process CWD.

`load_config` swallows FileNotFoundError and returns a hardcoded fallback, so a
run started from the wrong directory did not fail — it silently scored against
DIFFERENT WEIGHTS. Measured 2026-08-07 from /tmp: the fallback carries 9
composite weights with pop=0.18 and no vrp/iv_velocity/term_structure at all,
against the live config's 27 weights with pop=0.0354 and vrp=0.1755 — the three
largest live weights simply absent.

The same helper guards `auto_log_budget_cap` (drops the per-position budget when
it reads nothing) and `apply_auto_log_allowlist` (drops the Phase 1 cohort
quarantine). Both fail open, which is why silence was expensive.
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from src import options_screener as S

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestRepoPathHelper(unittest.TestCase):

    def test_relative_paths_anchor_to_the_repo_root(self):
        self.assertEqual(Path(S._repo_path("config.json")),
                         REPO_ROOT / "config.json")

    def test_absolute_paths_pass_through_untouched(self):
        """Every caller that injects its own config must keep working —
        the doctor's temp fixtures, --config, the calibration harnesses."""
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "config.json")
            self.assertEqual(S._repo_path(p), p)

    def test_result_is_always_absolute(self):
        for candidate in ("config.json", "a/b/config.json", "./config.json"):
            self.assertTrue(os.path.isabs(S._repo_path(candidate)), candidate)


class TestLoadConfigIsCwdIndependent(unittest.TestCase):

    def _live(self):
        with open(REPO_ROOT / "config.json") as fh:
            return json.load(fh)

    def test_reads_the_real_config_from_a_foreign_cwd(self):
        live = self._live()
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                got = S.load_config()
            finally:
                os.chdir(cwd)
        self.assertEqual(got["composite_weights"], live["composite_weights"],
                         "load_config fell back to the hardcoded defaults "
                         "instead of reading the repo's config")

    def test_the_fallback_is_distinguishable_from_the_real_config(self):
        """Guards the test above: if the fallback ever happened to match the
        live config, that test would pass without proving anything."""
        live = self._live()["composite_weights"]
        self.assertIn("vrp", live)
        self.assertGreater(len(live), 12)

    def test_an_injected_config_still_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "custom.json")
            with open(p, "w") as fh:
                json.dump({"composite_weights": {"pop": 1.0}}, fh)
            self.assertEqual(S.load_config(p)["composite_weights"], {"pop": 1.0})

    def test_a_missing_injected_config_still_falls_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            got = S.load_config(os.path.join(tmp, "nope.json"))
        self.assertIn("composite_weights", got)


class TestAutoLogGuardsAreCwdIndependent(unittest.TestCase):
    """Both of these fail OPEN, so a wrong-directory read removed a safety rail
    rather than raising."""

    def _from_tmp(self, fn):
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                return fn()
            finally:
                os.chdir(cwd)

    def test_budget_cap_survives_a_foreign_cwd(self):
        here = S.auto_log_budget_cap()
        there = self._from_tmp(S.auto_log_budget_cap)
        self.assertEqual(here, there)

    def test_allowlist_survives_a_foreign_cwd(self):
        trade = {"strategy_name": "Long Call"}
        here = S.apply_auto_log_allowlist(dict(trade))
        there = self._from_tmp(lambda: S.apply_auto_log_allowlist(dict(trade)))
        self.assertEqual(here, there)


class TestEveryConfigReaderIsCwdIndependent(unittest.TestCase):
    """The sweep across the remaining ~20 modules.

    Each reader is called from the repo root and again from a temp directory;
    the two answers must match. Every one of these swallowed its error and
    returned a fallback, so a mismatch is a silently different answer rather
    than a crash — which is why none of them was noticed.
    """

    def _both_cwds(self, fn):
        here = fn()
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chdir(tmp)
                there = fn()
            finally:
                os.chdir(cwd)
        return here, there

    def test_readers_agree_across_directories(self):
        from src.doctor import check_config
        from src.lottery.selector import load_lottery_config
        from src.maintenance import _cohort_min_dte, _max_capital_at_risk
        from src.structure.express import load_costs

        cases = {
            "maintenance._max_capital_at_risk": _max_capital_at_risk,
            "maintenance._cohort_min_dte": _cohort_min_dte,
            "structure.express.load_costs": load_costs,
            "lottery.selector.load_lottery_config": load_lottery_config,
            "doctor.check_config": lambda: check_config().status,
        }
        for name, fn in cases.items():
            with self.subTest(reader=name):
                here, there = self._both_cwds(fn)
                self.assertEqual(here, there,
                                 f"{name} gave a different answer from another "
                                 f"directory: {here!r} vs {there!r}")

    def test_doctor_guard_and_read_agree_on_the_same_file(self):
        """check_config tested the RAW path for existence and opened the
        RESOLVED one, so from elsewhere it reported the config missing without
        ever opening the file it would have read."""
        from src.doctor import check_config
        _, there = self._both_cwds(lambda: check_config().status)
        self.assertEqual(there, "PASS")

    def test_a_missing_injected_config_still_fails_the_doctor(self):
        """The guard must not be softened into always finding the repo's."""
        from src.doctor import check_config
        with tempfile.TemporaryDirectory() as tmp:
            result = check_config(os.path.join(tmp, "config.json"))
        self.assertEqual(result.status, "FAIL")


if __name__ == "__main__":
    unittest.main()
