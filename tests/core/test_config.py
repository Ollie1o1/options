import json, os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.config import load_config

class TestLoadConfig(unittest.TestCase):
    def test_reads_section_with_default(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"crypto": {"auto_log_enabled": True}}, f)
            path = f.name
        cfg = load_config(path)
        self.assertEqual(cfg.section("crypto")["auto_log_enabled"], True)
        self.assertEqual(cfg.section("missing", default={"k": 1})["k"], 1)
        os.unlink(path)

    def test_missing_file_returns_empty_sections(self):
        cfg = load_config("/no/such/config.json")
        self.assertEqual(cfg.section("crypto", default={}), {})

if __name__ == "__main__":
    unittest.main()
