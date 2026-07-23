import contextlib
import io
import os
import sys
import tempfile
import unittest

from src import formatting as fmt
from src import settings
from src.options_screener import main as screener_main


class ScreenerAppliesSavedThemeTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_path = settings._SETTINGS_PATH
        fd, self._tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        settings._SETTINGS_PATH = self._tmp_path
        settings.save_settings({"theme": "matrix_terminal"})

    def tearDown(self):
        settings._SETTINGS_PATH = self._orig_path
        if os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        fmt.set_theme("quant_desk")

    def test_main_applies_saved_theme_before_version_exit(self):
        fmt.set_theme("quant_desk")
        orig_argv = sys.argv
        sys.argv = ["prog", "--version"]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                with self.assertRaises(SystemExit):
                    screener_main()
        finally:
            sys.argv = orig_argv
        self.assertEqual(fmt.get_theme(), "matrix_terminal")


if __name__ == "__main__":
    unittest.main()
