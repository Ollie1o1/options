"""Releasing yfinance's WAL-backed caches while idle at a menu.

The property under test is narrow and mechanical: every cache manager that
exposes close_db() gets closed, nothing raises, and a yfinance that has been
upgraded out from under us degrades to a no-op rather than an exception on the
way into a menu prompt.
"""
import io
import os
import sys
import types
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cache_release import release_yfinance_caches  # noqa: E402


class _FakeManager:
    def __init__(self, raises=False):
        self.closed = 0
        self._raises = raises

    def close_db(self):
        self.closed += 1
        if self._raises:
            raise RuntimeError("database is locked")


def _install_fake_cache(monkey_modules, **managers):
    mod = types.ModuleType("yfinance.cache")
    for name, mgr in managers.items():
        setattr(mod, name, mgr)
    monkey_modules["yfinance.cache"] = mod
    yf = monkey_modules.setdefault("yfinance", types.ModuleType("yfinance"))
    yf.cache = mod
    return mod


_UNSET = object()


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k)
                       for k in ("yfinance", "yfinance.cache")}
        # `_install_fake_cache` reaches through `sys.modules["yfinance"]` and
        # assigns `.cache` on it. When yfinance is already imported that IS the
        # real module, and restoring the two sys.modules entries does not undo
        # an attribute set on an object they both still point at.
        real_yf = self._saved.get("yfinance")
        self._saved_cache_attr = (
            getattr(real_yf, "cache", _UNSET) if real_yf is not None else _UNSET)

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        real_yf = self._saved.get("yfinance")
        if real_yf is not None:
            if self._saved_cache_attr is _UNSET:
                if hasattr(real_yf, "cache"):
                    delattr(real_yf, "cache")
            else:
                real_yf.cache = self._saved_cache_attr

    def test_closes_every_manager(self):
        tz, cookie, isin = _FakeManager(), _FakeManager(), _FakeManager()
        _install_fake_cache(sys.modules, _TzDBManager=tz,
                            _CookieDBManager=cookie, _ISINDBManager=isin)
        closed = release_yfinance_caches()
        self.assertEqual(
            sorted(closed),
            ["_CookieDBManager", "_ISINDBManager", "_TzDBManager"])
        self.assertEqual((tz.closed, cookie.closed, isin.closed), (1, 1, 1))

    def test_a_manager_that_raises_does_not_stop_the_others(self):
        # This runs on the way into a menu prompt; one locked database must not
        # take the launcher down or prevent the other two from being released.
        tz, cookie, isin = _FakeManager(raises=True), _FakeManager(), _FakeManager()
        _install_fake_cache(sys.modules, _TzDBManager=tz,
                            _CookieDBManager=cookie, _ISINDBManager=isin)
        closed = release_yfinance_caches()
        self.assertNotIn("_TzDBManager", closed)
        self.assertIn("_CookieDBManager", closed)
        self.assertEqual((cookie.closed, isin.closed), (1, 1))

    def test_a_renamed_manager_is_skipped_not_crashed(self):
        # yfinance upgrade renames a manager: we close what we can and report
        # what we closed, so the miss is visible in the return value.
        cookie = _FakeManager()
        _install_fake_cache(sys.modules, _CookieDBManager=cookie)
        self.assertEqual(release_yfinance_caches(), ["_CookieDBManager"])

    def test_no_yfinance_at_all_is_a_noop(self):
        sys.modules["yfinance"] = None  # import raises
        sys.modules.pop("yfinance.cache", None)
        self.assertEqual(release_yfinance_caches(), [])


class RealYFinanceTest(unittest.TestCase):
    """Against the installed yfinance, not a fake: the manager names this
    module hardcodes must still exist, or the release silently does nothing."""

    def test_the_hardcoded_manager_names_still_exist(self):
        try:
            import yfinance.cache as cache
        except Exception:  # pragma: no cover - yfinance not installed
            self.skipTest("yfinance not installed")
        for name in ("_TzDBManager", "_CookieDBManager", "_ISINDBManager"):
            self.assertTrue(hasattr(cache, name), f"yfinance renamed {name}")
            self.assertTrue(hasattr(getattr(cache, name), "close_db"))

    def test_the_caches_are_wal_backed(self):
        # The whole reason for closing them. If yfinance stops using WAL, the
        # sidecar files stop existing and this module stops being necessary.
        try:
            import inspect

            import yfinance.cache as cache
        except Exception:  # pragma: no cover
            self.skipTest("yfinance not installed")
        src = inspect.getsource(cache._TzDBManager)
        self.assertIn("wal", src.lower())


def _run_release_tests():
    """Run ReleaseTest in-process, discarding its output."""
    suite = unittest.TestLoader().loadTestsFromTestCase(ReleaseTest)
    unittest.TextTestRunner(stream=io.StringIO(), verbosity=0).run(suite)


class FakeInstallLeavesNoTraceTest(unittest.TestCase):
    """`_install_fake_cache` must not mutate a real, already-imported yfinance.

    This is what has been reddening CI. `setdefault("yfinance", ...)` returns
    the REAL module whenever yfinance is already in `sys.modules`, and the next
    line does `yf.cache = <fake module>` on it. `tearDown` restored the two
    `sys.modules` entries but never that attribute, so the fake survived.

    `import yfinance.cache as cache` resolves through the parent package's
    attribute, so the next test to run got `_FakeManager` instances and
    `inspect.getsource` raised
    `TypeError: module, class, ... was expected, got _FakeManager`.

    It passed locally because the local runner imports yfinance later, so
    `setdefault` created a throwaway module that popping did clean up. Under
    pytest something imports yfinance first, and the order flips.
    """

    def test_a_real_yfinance_is_unchanged_after_the_release_tests_run(self):
        try:
            import yfinance
            import yfinance.cache  # noqa: F401
        except Exception:  # pragma: no cover - yfinance not installed
            self.skipTest("yfinance not installed")

        before_module = sys.modules.get("yfinance.cache")
        before_attr = getattr(yfinance, "cache", None)

        _run_release_tests()

        self.assertIs(sys.modules.get("yfinance.cache"), before_module,
                      "sys.modules entry was not restored")
        self.assertIs(getattr(yfinance, "cache", None), before_attr,
                      "the yfinance.cache ATTRIBUTE was left pointing at the fake")

    def test_the_real_managers_survive_a_fake_install(self):
        # The consequence, stated in the terms the failing test hit.
        try:
            import yfinance  # noqa: F401
            import yfinance.cache  # noqa: F401
        except Exception:  # pragma: no cover
            self.skipTest("yfinance not installed")

        _run_release_tests()

        import inspect

        import yfinance.cache as cache
        inspect.getsource(cache._TzDBManager)  # must not raise


if __name__ == "__main__":
    unittest.main()
