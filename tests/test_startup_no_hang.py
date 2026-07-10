"""Interactive startup must not block the mode menu on portfolio maintenance.

`_render_regime_with_exit_enforcement` runs `pm.update_positions()` (auto-closing
positions past their stops) in a daemon thread and waits for it before returning
so the menu can print. That wait used to be 60s — matched to update_positions'
worst-case internal yfinance timeouts — so a rate-limited data feed turned every
launch into a minute-long hang. The wait is now bounded; exit enforcement is a
daemon and idempotent, so overrunning it just finishes in the background.
"""
from __future__ import annotations

import time
import unittest

import src.options_screener as S
from src import regime_dashboard


class StartupDoesNotHangTest(unittest.TestCase):
    def setUp(self):
        self._orig_dash = regime_dashboard.print_regime_dashboard
        self._orig_timeout = S._EXIT_ENFORCE_JOIN_TIMEOUT
        regime_dashboard.print_regime_dashboard = lambda width: print("DASH")

    def tearDown(self):
        regime_dashboard.print_regime_dashboard = self._orig_dash
        S._EXIT_ENFORCE_JOIN_TIMEOUT = self._orig_timeout

    def test_slow_update_positions_does_not_block_past_the_bound(self):
        S._EXIT_ENFORCE_JOIN_TIMEOUT = 0.5

        class _SlowPM:
            enforced = False

            def update_positions(self):
                time.sleep(5)          # simulate a rate-limited data feed
                _SlowPM.enforced = True

        t0 = time.time()
        out = S._render_regime_with_exit_enforcement(
            _SlowPM(), 80, spinner_factory=_null_spinner)
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

        S._render_regime_with_exit_enforcement(_FastPM(), 80, spinner_factory=_null_spinner)
        self.assertTrue(done["v"])

    def test_the_bound_is_not_a_minute(self):
        # Guards the specific regression: 60s was the hang.
        self.assertLessEqual(S._EXIT_ENFORCE_JOIN_TIMEOUT, 20.0)


import contextlib


@contextlib.contextmanager
def _null_spinner(*_a, **_k):
    yield


if __name__ == "__main__":
    unittest.main()
