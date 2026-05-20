import inspect, os, sys, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.crypto import auto_logger, exit_enforcer

class TestConfigUnified(unittest.TestCase):
    def test_both_modules_use_core_config(self):
        self.assertIn("core.config", inspect.getsource(auto_logger._load_config))
        self.assertIn("core.config", inspect.getsource(exit_enforcer._load_config))

if __name__ == "__main__":
    unittest.main()
