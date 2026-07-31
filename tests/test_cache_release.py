"""Releasing yfinance's WAL-backed caches while idle at a menu.

The property under test is narrow and mechanical: every cache manager that
exposes close_db() gets closed, nothing raises, and a yfinance that has been
upgraded out from under us degrades to a no-op rather than an exception on the
way into a menu prompt.
"""
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


class ReleaseTest(unittest.TestCase):
    def setUp(self):
        self._saved = {k: sys.modules.get(k)
                       for k in ("yfinance", "yfinance.cache")}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v

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


if __name__ == "__main__":
    unittest.main()
