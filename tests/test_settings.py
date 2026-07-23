import os
import tempfile
import unittest

from src import formatting as fmt
from src import settings


class SettingsTestCase(unittest.TestCase):
    def setUp(self):
        self._orig_path = settings._SETTINGS_PATH
        fd, self._tmp_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self._tmp_path)  # start absent, like a fresh checkout
        settings._SETTINGS_PATH = self._tmp_path

    def tearDown(self):
        settings._SETTINGS_PATH = self._orig_path
        if os.path.exists(self._tmp_path):
            os.remove(self._tmp_path)
        fmt.set_theme("quant_desk")

    def test_load_defaults_when_file_missing(self):
        self.assertEqual(settings.load_settings(), {"theme": "quant_desk"})

    def test_save_then_load_round_trips(self):
        settings.save_settings({"theme": "matrix_terminal"})
        self.assertEqual(settings.load_settings(), {"theme": "matrix_terminal"})

    def test_load_falls_back_to_defaults_on_corrupt_file(self):
        with open(settings._SETTINGS_PATH, "w") as f:
            f.write("{not valid json")
        self.assertEqual(settings.load_settings(), {"theme": "quant_desk"})

    def test_load_fills_missing_keys_with_defaults(self):
        with open(settings._SETTINGS_PATH, "w") as f:
            f.write("{}")
        self.assertEqual(settings.load_settings(), {"theme": "quant_desk"})

    def test_get_theme_reads_persisted_value(self):
        settings.save_settings({"theme": "amber_crt"})
        self.assertEqual(settings.get_theme(), "amber_crt")

    def test_set_theme_persists_and_applies(self):
        ok = settings.set_theme("cyberpunk_neon")
        self.assertTrue(ok)
        self.assertEqual(settings.get_theme(), "cyberpunk_neon")
        self.assertEqual(fmt.get_theme(), "cyberpunk_neon")

    def test_set_theme_rejects_unknown_name(self):
        settings.set_theme("quant_desk")
        ok = settings.set_theme("not_a_real_theme")
        self.assertFalse(ok)
        self.assertEqual(settings.get_theme(), "quant_desk")
        self.assertEqual(fmt.get_theme(), "quant_desk")

    def test_apply_saved_theme_sets_fmt_theme(self):
        settings.save_settings({"theme": "amber_crt"})
        fmt.set_theme("quant_desk")  # reset, simulate a fresh process
        settings.apply_saved_theme()
        self.assertEqual(fmt.get_theme(), "amber_crt")

    def test_apply_saved_theme_defaults_when_no_file(self):
        fmt.set_theme("matrix_terminal")  # simulate a leftover non-default state
        settings.apply_saved_theme()
        self.assertEqual(fmt.get_theme(), "quant_desk")


if __name__ == "__main__":
    unittest.main()
