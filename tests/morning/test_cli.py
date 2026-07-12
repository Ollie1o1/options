import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import json
import tempfile
import unittest

from src.morning.__main__ import main


def _tiny_sidecar(tmp):
    panels = {pid: None for pid in
              ("health", "market", "vol", "macro_events", "signals",
               "portfolio", "gate")}
    panels["notes"] = ["note"]
    data = {"meta": {"schema": 1, "date": "2026-07-10", "generated_at": "x",
                     "session": "closed", "sidecar": "2026-07-10.json",
                     "title": "Morning Briefing — 2026-07-10"},
            "panels": panels,
            "failures": []}
    path = os.path.join(tmp, "2026-07-10.json")
    with open(path, "w") as f:
        json.dump(data, f)
    return path


class TestCli(unittest.TestCase):
    def test_from_sidecar_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            side = _tiny_sidecar(tmp)
            rc = main(["--from", side, "--out-dir", tmp])
            self.assertEqual(rc, 0)
            html = os.path.join(tmp, "2026-07-10.html")
            self.assertTrue(os.path.exists(html))
            with open(html) as f:
                self.assertIn("Morning Briefing", f.read())

    def test_from_missing_sidecar_errors(self):
        rc = main(["--from", "/nonexistent/x.json"])
        self.assertNotEqual(rc, 0)

    def test_from_rerender_does_not_touch_maintenance_state(self):
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            side = _tiny_sidecar(tmp)
            with mock.patch("src.morning.__main__._mark_maintenance_state") as mm:
                rc = main(["--from", side, "--out-dir", tmp])
            self.assertEqual(rc, 0)
            mm.assert_not_called()

    def test_fresh_build_marks_maintenance_state(self):
        import json as _json
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            with open(_tiny_sidecar(tmp)) as f:
                data = _json.load(f)
            with mock.patch("src.morning.__main__._mark_maintenance_state") as mm, \
                 mock.patch("src.morning.collect.build", return_value=data):
                rc = main(["--out-dir", tmp])
            self.assertEqual(rc, 0)
            mm.assert_called_once_with("2026-07-10")


if __name__ == "__main__":
    unittest.main()
