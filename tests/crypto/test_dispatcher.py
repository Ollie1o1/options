import os, subprocess, sys, unittest
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class TestDispatcher(unittest.TestCase):
    def _run(self, *args):
        env = dict(os.environ); env["PYTHONPATH"] = ROOT
        return subprocess.run([sys.executable, "-m", "src.crypto", *args],
                              cwd=ROOT, env=env, capture_output=True, text=True)

    def test_help_lists_verbs(self):
        r = self._run("--help")
        self.assertEqual(r.returncode, 0)
        for verb in ("scan", "log", "exits", "pnl", "backtest"):
            self.assertIn(verb, r.stdout)

    def test_unknown_verb_errors(self):
        r = self._run("frobnicate")
        self.assertNotEqual(r.returncode, 0)

if __name__ == "__main__":
    unittest.main()
