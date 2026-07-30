"""The portfolio viewer priced the whole book in silence.

`check_pnl` fetches one live quote per leg across every open position — 143 open
trades on the real book, up to four legs each — through a thread pool, with no
indicator at all. For as long as the feed took, the viewer looked frozen.

The bar has to survive the degraded path too: `ui` is set to None when its
import fails, so reaching for `ui.progress_bar` unguarded would convert a
cosmetic missing-dependency case into an AttributeError in the middle of
pricing.
"""
import unittest

import src.check_pnl as C


class ProgressHelper(unittest.TestCase):
    def test_returns_a_usable_bar_when_ui_is_available(self):
        bar = C._progress(5, "Pricing")
        bar.update(1)
        bar.close()

    def test_degrades_to_a_no_op_when_ui_is_unavailable(self):
        orig_flag, orig_ui = C._HAS_UI_CP, C.ui
        try:
            C._HAS_UI_CP = False
            C.ui = None
            bar = C._progress(5, "Pricing")   # must not raise
            bar.update(1)
            bar.close()
        finally:
            C._HAS_UI_CP, C.ui = orig_flag, orig_ui

    def test_the_no_op_bar_matches_the_real_one_s_interface(self):
        # Whatever the caller got, these three calls must always be safe.
        for bar in (C._NoProgress(), C._progress(3, "Pricing")):
            bar.update()
            bar.update(2)
            bar.close()


if __name__ == "__main__":
    unittest.main()
