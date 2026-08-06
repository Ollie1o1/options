"""Tests that scan_errors.log entries carry a timestamp.

Both writers used to emit a bare `=== SYMBOL (mode) ===` banner. When a recurring
SPY failure turned up in the log there was no way to tell whether it had fired
that morning or weeks earlier — dating it meant diffing the traceback's line
numbers against every commit that had touched the file. A timestamp per entry
turns that archaeology into reading one line.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_scan_error_log_timestamps -v
"""
from __future__ import annotations

import os
import re
import tempfile
import unittest
from datetime import datetime
from unittest import mock

from src import data_fetching, options_screener as osc

# "=== 2026-08-06 16:32:10 | <who> ==="
HEADER = re.compile(
    r"^=== (\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \| (.+) ===$", re.MULTILINE
)


class _LogCapture:
    """Point both writers at a scratch logs/ dir and read back what they wrote."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = self._tmp.name
        os.makedirs(os.path.join(root, "logs"), exist_ok=True)
        # Both writers derive logs/ from dirname(dirname(__file__)) of their own
        # module, so redirect the module file rather than the cwd.
        fake_src = os.path.join(root, "src", "x.py")
        self._patches = [
            mock.patch.object(osc, "__file__", fake_src),
            mock.patch.object(data_fetching, "__file__", fake_src),
        ]
        for p in self._patches:
            p.start()
        self.path = os.path.join(root, "logs", "scan_errors.log")
        return self

    def __exit__(self, *exc):
        for p in self._patches:
            p.stop()
        self._tmp.cleanup()
        return False

    def read(self) -> str:
        if not os.path.exists(self.path):
            return ""
        with open(self.path) as f:
            return f.read()


class ScorerErrorLogTest(unittest.TestCase):
    """_score_fetched_data's handler — the one the SPY failures came through."""

    def _write_one(self):
        with _LogCapture() as cap:
            # A data_result missing "df" is exactly the shape that has been
            # failing in the wild.
            result = osc._score_fetched_data(
                "SPY", {"context": {}}, "Discovery scan", 7, 45,
                0.04, {}, {}, "swing",
            )
            return cap.read(), result

    def test_entry_is_written_and_timestamped(self):
        text, _ = self._write_one()
        m = HEADER.search(text)
        self.assertIsNotNone(m, f"no timestamped header found in:\n{text}")

    def test_timestamp_parses_and_is_recent(self):
        text, _ = self._write_one()
        m = HEADER.search(text)
        stamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        self.assertLess(abs((datetime.now() - stamp).total_seconds()), 120)

    def test_symbol_and_mode_survive(self):
        text, _ = self._write_one()
        m = HEADER.search(text)
        self.assertEqual(m.group(2), "SPY (Discovery scan)")

    def test_traceback_still_written(self):
        text, _ = self._write_one()
        self.assertIn("Traceback", text)
        self.assertIn("KeyError", text)

    def test_error_is_still_reported_on_the_result(self):
        _, result = self._write_one()
        self.assertFalse(result["success"])
        self.assertIsNotNone(result["error"])

    def test_entries_append_not_overwrite(self):
        with _LogCapture() as cap:
            for _ in range(3):
                osc._score_fetched_data(
                    "SPY", {"context": {}}, "Discovery scan", 7, 45,
                    0.04, {}, {}, "swing",
                )
            self.assertEqual(len(HEADER.findall(cap.read())), 3)


class RetryDecoratorErrorLogTest(unittest.TestCase):
    """The retry decorator's handler in data_fetching."""

    def test_entry_is_timestamped(self):
        @data_fetching.retry_with_backoff(retries=1, backoff_in_seconds=0)
        def _boom(symbol):
            raise ValueError("chain went sideways")

        with _LogCapture() as cap:
            with self.assertRaises(ValueError):
                _boom("SPY")
            text = cap.read()

        m = HEADER.search(text)
        self.assertIsNotNone(m, f"no timestamped header found in:\n{text}")
        stamp = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        self.assertLess(abs((datetime.now() - stamp).total_seconds()), 120)
        self.assertIn("_boom(SPY)", m.group(2))


if __name__ == "__main__":
    unittest.main()
