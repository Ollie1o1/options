"""The scoring phase of a scan ran silently.

Phase 1 (fetching) has had a progress bar for a long time. Phase 2 (scoring
every fetched chain) had nothing — and it is not fast: measured at 1.90s for a
single ticker (SPY), so a 111-ticker scan spends minutes here. The user watched
the fetch bar reach 100% and then faced a still screen with no indication that
anything was still running.

`_progress_bar` is the shared factory both phases use, so "is there an
indicator?" has one answer in one place, and a headless/automated run degrades
to a no-op rather than writing control characters into a log file.
"""
import io
import unittest

from src.options_screener import _progress_bar


class DisabledBar(unittest.TestCase):
    """Automation (`--auto`, cron, the catch-up windows) redirects stdout to a
    log; a live bar there is thousands of carriage returns in a file."""

    def test_disabled_bar_writes_nothing(self):
        sink = io.StringIO()
        bar = _progress_bar(10, "Scoring", enabled=False, stream=sink)
        bar.update(1)
        bar.close()
        self.assertEqual(sink.getvalue(), "")

    def test_disabled_bar_supports_the_full_protocol(self):
        bar = _progress_bar(10, "Scoring", enabled=False, stream=io.StringIO())
        bar.update()
        bar.update(5)
        bar.close()
        bar.close()  # idempotent — the scan closes in a finally

    def test_a_zero_total_does_not_explode(self):
        # An empty ticker list is a legitimate scan (everything filtered out).
        bar = _progress_bar(0, "Scoring", enabled=False, stream=io.StringIO())
        bar.update(1)
        bar.close()


class EnabledBar(unittest.TestCase):
    def test_enabled_bar_writes_its_label(self):
        sink = io.StringIO()
        bar = _progress_bar(4, "Scoring", enabled=True, stream=sink)
        bar.update(1)
        bar.close()
        self.assertIn("Scoring", sink.getvalue())

    def test_enabled_bar_reflects_progress(self):
        # Assert the counter, not the rendered text: tqdm throttles redraws on a
        # time interval, so four rapid updates legitimately render once.
        sink = io.StringIO()
        bar = _progress_bar(4, "Scoring", enabled=True, stream=sink)
        for _ in range(4):
            bar.update(1)
        self.assertEqual(bar.n, 4)
        bar.close()

    def test_enabled_bar_survives_a_double_close(self):
        sink = io.StringIO()
        bar = _progress_bar(2, "Scoring", enabled=True, stream=sink)
        bar.update(1)
        bar.close()
        bar.close()


if __name__ == "__main__":
    unittest.main()
