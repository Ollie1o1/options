"""Tests for src/doctor.py — read-only first-run environment self-check.

Every check is a pure function over injected inputs (mirrors
tests/test_preflight.py). Network checks are always given a fake `fetch`;
nothing here touches a real socket. The fresh-clone simulation points every
path at an empty tmp dir (never the real repo config/db/lock/.env) and asserts
every non-passing check carries an actionable one-line fix.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_doctor -v
"""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from src import doctor
from src import formatting as fmt


class PythonVersionTest(unittest.TestCase):
    def test_current_interpreter_passes_floor(self):
        # This suite only runs on the project venv, which is well above 3.11.
        c = doctor.check_python_version()
        self.assertEqual(c.status, "PASS")

    def test_old_version_fails_with_fix(self):
        c = doctor.check_python_version((3, 9, 0))
        self.assertEqual(c.status, "FAIL")
        self.assertIn("3.9.0", c.detail)
        self.assertTrue(c.fix)

    def test_floor_version_passes(self):
        c = doctor.check_python_version((3, 11, 0))
        self.assertEqual(c.status, "PASS")


class ParseLockFileTest(unittest.TestCase):
    def test_parses_pinned_versions(self):
        text = "# comment\n\nnumpy==2.4.3\npandas==2.3.3\n"
        pins = doctor.parse_lock_file(text)
        self.assertEqual(pins["numpy"], "2.4.3")
        self.assertEqual(pins["pandas"], "2.3.3")

    def test_ignores_lines_without_pin(self):
        pins = doctor.parse_lock_file("-e .\nsome-editable-thing\n")
        self.assertEqual(pins, {})


class DependenciesCheckTest(unittest.TestCase):
    LOCK = "numpy==2.4.3\npandas==2.3.3\nrequests==2.32.5\n"

    def test_missing_lock_file_fails(self):
        c = doctor.check_dependencies(None)
        self.assertEqual(c.status, "FAIL")
        self.assertIn("requirements-lock.txt", c.detail)
        self.assertTrue(c.fix)

    def test_all_matched_passes(self):
        def fake_version(pkg):
            return {"numpy": "2.4.3", "pandas": "2.3.3", "requests": "2.32.5"}[pkg]
        c = doctor.check_dependencies(self.LOCK, get_version=fake_version,
                                      packages=["numpy", "pandas", "requests"])
        self.assertEqual(c.status, "PASS")

    def test_drift_warns_never_installs(self):
        def fake_version(pkg):
            return {"numpy": "2.4.1", "pandas": "2.3.3", "requests": "2.32.5"}[pkg]
        c = doctor.check_dependencies(self.LOCK, get_version=fake_version,
                                      packages=["numpy", "pandas", "requests"])
        self.assertEqual(c.status, "WARN")
        self.assertIn("numpy 2.4.1!=2.4.3", c.detail)
        self.assertIn("pip install -r requirements-lock.txt", c.fix)

    def test_missing_package_fails(self):
        def fake_version(pkg):
            return None if pkg == "requests" else "2.4.3"
        c = doctor.check_dependencies(self.LOCK, get_version=fake_version,
                                      packages=["numpy", "requests"])
        self.assertEqual(c.status, "FAIL")
        self.assertIn("requests", c.detail)

    def test_package_not_in_lock_is_skipped(self):
        # numpy matches its pin; the unpinned package has nothing to compare
        # against and must not be reported as drift or as missing.
        def fake_version(pkg):
            return "2.4.3" if pkg == "numpy" else "9.9.9"
        c = doctor.check_dependencies("numpy==2.4.3\n", get_version=fake_version,
                                      packages=["numpy", "some-unpinned-thing"])
        self.assertEqual(c.status, "PASS")
        self.assertNotIn("some-unpinned-thing", c.detail)


class NetworkClassifierTest(unittest.TestCase):
    def test_200_passes(self):
        status, detail, fix = doctor.classify_network_result(None, 200)
        self.assertEqual(status, "PASS")
        self.assertFalse(fix)

    def test_429_is_rate_limited_not_down(self):
        status, detail, fix = doctor.classify_network_result(None, 429)
        self.assertEqual(status, "WARN")
        self.assertIn("rate-limited", detail)

    def test_connection_error_is_down(self):
        status, detail, fix = doctor.classify_network_result(ConnectionError("boom"), None)
        self.assertEqual(status, "FAIL")
        self.assertIn("unreachable", detail)

    def test_timeout_is_down_not_a_crash(self):
        class Timeout(Exception):
            pass
        status, detail, fix = doctor.classify_network_result(Timeout("slow"), None)
        self.assertEqual(status, "FAIL")
        self.assertIn("timed out", detail)

    def test_999_blocked_distinguished_from_outage(self):
        status, detail, fix = doctor.classify_network_result(None, 999)
        self.assertEqual(status, "WARN")
        self.assertIn("blocked", detail)


class NetworkCheckMockedTest(unittest.TestCase):
    """Mocked fetch only — must never hit the network in the test suite."""

    def test_yahoo_reachable(self):
        class Resp:
            status_code = 200
        c = doctor.check_yahoo_network(fetch=lambda: Resp())
        self.assertEqual(c.name, "network: yahoo")
        self.assertEqual(c.status, "PASS")

    def test_cboe_unreachable_degrades_gracefully(self):
        def raise_conn_error():
            raise ConnectionError("no route to host")
        c = doctor.check_cboe_network(fetch=raise_conn_error)
        self.assertEqual(c.name, "network: cboe")
        self.assertEqual(c.status, "FAIL")
        self.assertTrue(c.fix)

    def test_probe_never_raises_out_of_the_function(self):
        def blows_up():
            raise RuntimeError("anything at all")
        # Must not propagate — the doctor must never hang or crash on a dead
        # network path.
        c = doctor.check_yahoo_network(fetch=blows_up)
        self.assertEqual(c.status, "FAIL")


class DirWritableTest(unittest.TestCase):
    def test_existing_writable_dir_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = doctor.check_dir_writable("reports", tmp)
            self.assertEqual(c.status, "PASS")

    def test_missing_dir_with_writable_parent_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "reports")
            c = doctor.check_dir_writable("reports", target)
            self.assertEqual(c.status, "PASS")
            self.assertIn("does not exist yet", c.detail)

    def test_unwritable_dir_fails_with_fix(self):
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "locked")
            os.mkdir(target)
            os.chmod(target, 0o500)
            try:
                if os.access(target, os.W_OK):
                    self.skipTest("running as root or FS ignores chmod; can't force unwritable")
                c = doctor.check_dir_writable("logs", target)
                self.assertEqual(c.status, "FAIL")
                self.assertTrue(c.fix)
            finally:
                os.chmod(target, 0o700)


class DbWritableTest(unittest.TestCase):
    def test_missing_db_with_writable_parent_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = doctor.check_db_writable(os.path.join(tmp, "paper_trades.db"))
            self.assertEqual(c.status, "PASS")

    def test_existing_db_reports_schema_version_read_only(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "paper_trades.db")
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA user_version = 16")
            conn.commit()
            conn.close()
            c = doctor.check_db_writable(db_path, expected_schema=16)
            self.assertEqual(c.status, "PASS")
            self.assertIn("schema v16", c.detail)
            self.assertNotIn("migrate", c.detail)

            c2 = doctor.check_db_writable(db_path, expected_schema=17)
            self.assertIn("code expects v17", c2.detail)

    def test_check_never_writes_to_the_db(self):
        import sqlite3
        with tempfile.TemporaryDirectory() as tmp:
            db_path = os.path.join(tmp, "paper_trades.db")
            conn = sqlite3.connect(db_path)
            conn.execute("PRAGMA user_version = 5")
            conn.commit()
            conn.close()
            before = os.path.getmtime(db_path)
            doctor.check_db_writable(db_path, expected_schema=16)
            after = os.path.getmtime(db_path)
            self.assertEqual(before, after)


class ConfigCheckTest(unittest.TestCase):
    def test_missing_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = doctor.check_config(os.path.join(tmp, "config.json"))
            self.assertEqual(c.status, "FAIL")
            self.assertIn("not found", c.detail)
            self.assertTrue(c.fix)

    def test_invalid_json_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w") as f:
                f.write("{not json")
            c = doctor.check_config(path)
            self.assertEqual(c.status, "FAIL")
            self.assertIn("invalid JSON", c.detail)

    def test_bad_key_named_in_detail(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w") as f:
                json.dump({"filters": {"delta_min": 0.5, "delta_max": 0.2}}, f)
            c = doctor.check_config(path)
            self.assertEqual(c.status, "WARN")
            self.assertIn("delta_min", c.detail)
            self.assertIn("delta_max", c.detail)

    def test_valid_config_passes(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            with open(path, "w") as f:
                json.dump({}, f)  # empty cfg -> validate_core_config uses safe defaults
            c = doctor.check_config(path)
            self.assertEqual(c.status, "PASS")

    def test_real_repo_config_is_valid(self):
        # The actual config.json this repo ships should itself pass.
        c = doctor.check_config("config.json")
        self.assertEqual(c.status, "PASS", c.detail)


class SchedulerCheckTest(unittest.TestCase):
    def test_missing_state_warns_not_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            c = doctor.check_scheduler(os.path.join(tmp, "state.json"))
            self.assertEqual(c.status, "WARN")
            self.assertIn("fresh install", c.detail)
            self.assertTrue(c.fix)

    def test_never_writes_state_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            doctor.check_scheduler(path)
            self.assertFalse(os.path.exists(path))

    def test_healthy_state_passes(self):
        import datetime
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            today = datetime.date(2026, 7, 31)
            state = {
                "last_autolog": {"ds": "2026-07-31", "sps": "2026-07-31",
                                 "ss": "2026-07-31", "ics": "2026-07-31"},
                "last_checkpoint": "2026-07-30",
                "last_track_record": "2026-07-30",
                "last_chain_archive": "2026-07-31",
                "last_morning_briefing": "2026-07-31",
            }
            with open(path, "w") as f:
                json.dump(state, f)
            # jobs=[] pins "launchctl reported nothing failing", so the check
            # reads the seeded state rather than this machine's real scheduler.
            c = doctor.check_scheduler(path, now=today, jobs=[])
            self.assertEqual(c.status, "PASS")

    def test_a_dead_scheduler_outranks_staleness_and_names_the_real_fix(self):
        # The state file says everything ran today — which it does whenever the
        # operator opens the app by hand. launchctl is the only witness that the
        # jobs themselves are refusing to start, and its verdict must win.
        from datetime import date as _date

        from src.maintenance_health import LaunchdJob

        today = _date(2026, 7, 31)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            with open(path, "w") as f:
                json.dump({"last_autolog": {"ds": "2026-07-31", "sps": "2026-07-31",
                                            "ss": "2026-07-31", "ics": "2026-07-31"},
                           "last_checkpoint": "2026-07-30",
                           "last_track_record": "2026-07-30",
                           "last_chain_archive": "2026-07-31",
                           "last_morning_briefing": "2026-07-31"}, f)
            dead = [LaunchdJob(label="com.ollie.options.maintenance", pid=None,
                               last_exit_status=78)]
            c = doctor.check_scheduler(path, now=today, jobs=dead)
            self.assertEqual(c.status, "FAIL")
            self.assertIn("Login Items", c.fix)
            # "open the app" is the one action that cannot fix exit 78
            self.assertNotIn("open the launcher", c.fix)


    def test_critical_scheduler_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "state.json")
            # State present but every job stamped far in the past.
            state = {"last_autolog": {"ds": "2020-01-01", "sps": "2020-01-01",
                                      "ss": "2020-01-01", "ics": "2020-01-01"}}
            with open(path, "w") as f:
                json.dump(state, f)
            c = doctor.check_scheduler(path)
            self.assertEqual(c.status, "FAIL")


class OptionalEnvTest(unittest.TestCase):
    def test_read_env_keys_from_dotenv_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, ".env")
            with open(path, "w") as f:
                f.write('OPENROUTER_API_KEY=sk-abc123\n# comment\nPOLYGON_API_KEY="poly-key"\n')
            values = doctor.read_env_keys(path, environ={})
            self.assertEqual(values["OPENROUTER_API_KEY"], "sk-abc123")
            self.assertEqual(values["POLYGON_API_KEY"], "poly-key")
            self.assertNotIn("SEC_EDGAR_CONTACT", values)

    def test_missing_dotenv_falls_back_to_environ(self):
        with tempfile.TemporaryDirectory() as tmp:
            values = doctor.read_env_keys(os.path.join(tmp, ".env"),
                                          environ={"SEC_EDGAR_CONTACT": "ops@example.com"})
            self.assertEqual(values["SEC_EDGAR_CONTACT"], "ops@example.com")

    def test_absence_is_never_a_failure(self):
        c = doctor.check_optional_env("OPENROUTER_API_KEY", {})
        self.assertEqual(c.status, "PASS")
        self.assertIn("optional", c.detail)
        self.assertIn("not set", c.detail)

    def test_present_reports_set(self):
        c = doctor.check_optional_env("POLYGON_API_KEY", {"POLYGON_API_KEY": "x"})
        self.assertEqual(c.status, "PASS")
        self.assertIn("set", c.detail)
        self.assertNotIn("not set", c.detail)


class RenderTest(unittest.TestCase):
    def setUp(self):
        self._saved_color = fmt._COLOR_ENABLED
        fmt._COLOR_ENABLED = False  # pin, never toggle via env vars

    def tearDown(self):
        fmt._COLOR_ENABLED = self._saved_color

    def test_render_lists_every_check_and_run_command(self):
        checks = [
            doctor.CheckResult("python version", "PASS", "3.12.1"),
            doctor.CheckResult("dependencies", "WARN", "numpy drift", "pip install -r requirements-lock.txt"),
            doctor.CheckResult("config.json", "FAIL", "not found", "restore config.json"),
        ]
        out = doctor.render(checks)
        self.assertIn("python version", out)
        self.assertIn("dependencies", out)
        self.assertIn("config.json", out)
        self.assertIn("restore config.json", out)
        self.assertIn(doctor.RUN_COMMAND, out)

    def test_all_pass_has_no_fix_section(self):
        checks = [doctor.CheckResult("python version", "PASS", "3.12.1")]
        out = doctor.render(checks)
        self.assertIn("Everything checks out", out)
        self.assertNotIn("Fix these", out)


class FreshCloneSimulationTest(unittest.TestCase):
    """The acceptance scenario: every path points into an empty tmp dir —
    config moved aside, no .env, no lock file, no db, no maintenance state.
    Never touches the real repo's config.json/.env/paper_trades.db. Network is
    disabled (network_probe=False) so the suite stays offline; the two probes
    themselves are exercised separately under NetworkCheckMockedTest."""

    def test_fresh_clone_produces_actionable_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            checks = doctor.run_doctor(
                config_path=os.path.join(tmp, "config.json"),
                db_path=os.path.join(tmp, "paper_trades.db"),
                reports_dir=os.path.join(tmp, "reports"),
                logs_dir=os.path.join(tmp, "logs"),
                lock_path=os.path.join(tmp, "requirements-lock.txt"),
                env_path=os.path.join(tmp, ".env"),
                state_path=os.path.join(tmp, "state.json"),
                network_probe=False,
            )
            names = {c.name for c in checks}
            self.assertIn("config.json", names)
            self.assertIn("dependencies", names)
            self.assertIn("scheduler", names)

            by_name = {c.name: c for c in checks}
            self.assertEqual(by_name["config.json"].status, "FAIL")
            self.assertEqual(by_name["dependencies"].status, "FAIL")
            self.assertEqual(by_name["scheduler"].status, "WARN")

            # Every non-passing check must carry a one-line fix — that's the
            # "actionable output per failure" bar.
            for c in checks:
                if c.status != "PASS":
                    self.assertTrue(c.fix, f"{c.name} ({c.status}) has no fix text")

            # Nothing in the tmp dir was created by the doctor itself.
            self.assertEqual(os.listdir(tmp), [])

    def test_fresh_clone_never_touches_the_real_repo_paths(self):
        # Sanity: the simulation above must not have been silently satisfied
        # by falling back to the real repo's config.json/.env in cwd.
        with tempfile.TemporaryDirectory() as tmp:
            fake_config = os.path.join(tmp, "config.json")
            self.assertFalse(os.path.exists(fake_config))
            c = doctor.check_config(fake_config)
            self.assertEqual(c.status, "FAIL")
            self.assertFalse(os.path.exists(fake_config))  # doctor didn't create it


class RunDoctorRealRepoTest(unittest.TestCase):
    """Sanity check against the real repo paths (network disabled) — should
    not raise, and should reflect this repo's own config as valid."""

    def test_real_repo_smoke(self):
        checks = doctor.run_doctor(network_probe=False)
        by_name = {c.name: c for c in checks}
        self.assertEqual(by_name["config.json"].status, "PASS")
        self.assertEqual(by_name["python version"].status, "PASS")


if __name__ == "__main__":
    unittest.main()
