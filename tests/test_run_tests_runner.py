"""The local runner's own honesty check.

The runner exists to make an under-collecting run visible. That only works if
its exit code means something: a module that cannot import because pytest is
absent is expected locally and must not turn the run red, while a module that
fails to import for any other reason is a real breakage and must.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.run_tests import CollectingSuite, is_missing_pytest


class TestImportErrorClassification(unittest.TestCase):
    def test_missing_pytest_is_expected_locally(self):
        exc = ModuleNotFoundError("No module named 'pytest'", name="pytest")
        self.assertTrue(is_missing_pytest(exc))

    def test_missing_pytest_plugin_is_also_expected(self):
        exc = ModuleNotFoundError(
            "No module named 'pytest_asyncio'", name="pytest_asyncio"
        )
        self.assertTrue(is_missing_pytest(exc))

    def test_any_other_missing_module_is_a_real_breakage(self):
        exc = ModuleNotFoundError("No module named 'src.gone'", name="src.gone")
        self.assertFalse(is_missing_pytest(exc))

    def test_a_syntax_error_is_a_real_breakage(self):
        self.assertFalse(is_missing_pytest(SyntaxError("bad")))


class TestFixtureCleanup(unittest.TestCase):
    def test_finished_group_releases_unreachable_fixture_cycles(self):
        import gc
        import weakref
        refs = []

        class Fixture:
            pass

        class Case(unittest.TestCase):
            def runTest(self):
                fixture = Fixture()
                fixture.cycle = fixture
                refs.append(weakref.ref(fixture))

        enabled = gc.isenabled()
        gc.disable()
        try:
            CollectingSuite([Case()]).run(unittest.TestResult())
            self.assertIsNone(refs[0]())
        finally:
            if enabled:
                gc.enable()


if __name__ == "__main__":
    unittest.main()
