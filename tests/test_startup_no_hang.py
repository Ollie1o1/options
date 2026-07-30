"""Interactive startup must not block the mode menu on portfolio maintenance.

`_render_regime_with_exit_enforcement` runs `pm.update_positions()` (auto-closing
positions past their stops) in a daemon thread and waits for it before returning
so the menu can print. That wait used to be 60s — matched to update_positions'
worst-case internal yfinance timeouts — so a rate-limited data feed turned every
launch into a minute-long hang. The wait is now bounded; exit enforcement is a
daemon and idempotent, so overrunning it just finishes in the background.
"""
from __future__ import annotations

import tempfile
import threading
import time
import unittest

import src.options_screener as S
from src import regime_dashboard


class _NoopPM:
    def update_positions(self):
        pass


class StartupDoesNotHangTest(unittest.TestCase):
    def setUp(self):
        self._orig_dash = regime_dashboard.print_regime_dashboard
        self._orig_timeout = S._EXIT_ENFORCE_JOIN_TIMEOUT
        regime_dashboard.print_regime_dashboard = lambda width: print("DASH")
        # A temp cache dir keeps each test on the cold path, so the patched
        # renderer above is actually exercised rather than replayed from a
        # cache the developer's machine happens to have warm.
        self.cache_dir = tempfile.mkdtemp()

    def tearDown(self):
        regime_dashboard.print_regime_dashboard = self._orig_dash
        S._EXIT_ENFORCE_JOIN_TIMEOUT = self._orig_timeout

    def _render(self, pm, **kw):
        kw.setdefault("cache_dir", self.cache_dir)
        return S._render_regime_with_exit_enforcement(
            pm, 80, spinner_factory=_null_spinner, **kw)

    def test_slow_update_positions_does_not_block_past_the_bound(self):
        S._EXIT_ENFORCE_JOIN_TIMEOUT = 0.5

        class _SlowPM:
            enforced = False

            def update_positions(self):
                time.sleep(5)          # simulate a rate-limited data feed
                _SlowPM.enforced = True

        t0 = time.time()
        out = self._render(_SlowPM())
        elapsed = time.time() - t0

        # Bounded by the join timeout (+ a little slack), NOT the 5s update.
        self.assertLess(elapsed, 2.5)
        self.assertIn("DASH", out)

    def test_a_fast_update_completes_inline(self):
        S._EXIT_ENFORCE_JOIN_TIMEOUT = 5.0
        done = {"v": False}

        class _FastPM:
            def update_positions(self):
                done["v"] = True

        self._render(_FastPM())
        self.assertTrue(done["v"])

    def test_the_bound_is_not_a_minute(self):
        # Guards the specific regression: 60s was the hang.
        self.assertLessEqual(S._EXIT_ENFORCE_JOIN_TIMEOUT, 20.0)

    def test_the_bound_no_longer_swallows_a_nine_second_update(self):
        # update_positions() measured 8.94s on the real book; a bound above that
        # put ~9s in front of the menu on every single launch.
        self.assertLess(S._EXIT_ENFORCE_JOIN_TIMEOUT, 5.0)

    def test_startup_does_not_wait_on_exit_enforcement_at_all(self):
        # Once the dashboard is cache-first, this join IS the startup cost:
        # a 2s bound measured as exactly 2.00s to menu. Default must not block.
        self.assertEqual(S._EXIT_ENFORCE_JOIN_TIMEOUT, 0.0)

    def test_a_slow_update_leaves_the_menu_immediate(self):
        started = threading.Event()

        class _SlowPM:
            def update_positions(self):
                started.set()
                time.sleep(5)

        t0 = time.time()
        self._render(_SlowPM())
        self.assertLess(time.time() - t0, 0.5)
        # Still actually running behind the user, not skipped.
        self.assertTrue(started.wait(timeout=2.0))

    def test_exit_enforcement_still_runs_to_completion_in_the_background(self):
        finished = threading.Event()

        class _PM:
            def update_positions(self):
                time.sleep(0.2)
                finished.set()

        self._render(_PM())
        self.assertTrue(finished.wait(timeout=3.0),
                        "exits were never enforced, only deferred")


class StartupServesTheDashboardFromCache(unittest.TestCase):
    """The dashboard costs 5-10s of live fetches. Once cached it must render
    instantly, and it must say how old it is."""

    def setUp(self):
        self._orig_dash = regime_dashboard.print_regime_dashboard
        self.cache_dir = tempfile.mkdtemp()
        self.calls = []

        def _slow_dash(width):
            self.calls.append(width)
            time.sleep(3)
            print("LIVE DASH")

        regime_dashboard.print_regime_dashboard = _slow_dash

    def tearDown(self):
        regime_dashboard.print_regime_dashboard = self._orig_dash

    def _render(self, **kw):
        kw.setdefault("cache_dir", self.cache_dir)
        return S._render_regime_with_exit_enforcement(
            _NoopPM(), 80, spinner_factory=_null_spinner, **kw)

    def test_first_launch_renders_live_and_shows_the_panel(self):
        out = self._render()
        self.assertIn("LIVE DASH", out)
        self.assertEqual(len(self.calls), 1)

    def test_second_launch_does_not_re_render(self):
        self._render()
        self._render()
        self.assertEqual(len(self.calls), 1, "second launch re-fetched the market")

    def test_second_launch_is_fast(self):
        self._render()
        t0 = time.time()
        out = self._render()
        self.assertLess(time.time() - t0, 1.0)
        self.assertIn("LIVE DASH", out)

    def test_a_cached_render_is_stamped_with_its_age(self):
        # Stale market data must never be presented as if it were live.
        from src.panel_cache import store_panel
        from datetime import datetime, timedelta
        old = datetime.now() - timedelta(minutes=7)
        store_panel("regime_dashboard", 80, "OLD DASH\n", old,
                    cache_dir=self.cache_dir)
        out = self._render()
        self.assertIn("OLD DASH", out)
        # Match the stamp we add, not a bare "as of" — the real macro/rates line
        # already contains "as of", so the loose substring proves nothing.
        self.assertIn(f"market data as of {old:%H:%M}", out)

    def test_a_just_rendered_panel_carries_no_stale_stamp(self):
        out = self._render()
        self.assertNotIn("market data as of", out)

    def test_a_freshly_cached_panel_carries_no_stale_stamp(self):
        # Within min_age of the render, the stamp is noise, not information.
        self._render()
        self.assertNotIn("market data as of", self._render())


import contextlib


@contextlib.contextmanager
def _null_spinner(*_a, **_k):
    yield


if __name__ == "__main__":
    unittest.main()
