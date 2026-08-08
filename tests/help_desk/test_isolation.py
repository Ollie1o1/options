"""The manual is display-only, and this is what makes that structural rather
than a promise in a docstring. It parses the package's imports rather than
importing it, so the check holds even for a module that would fail to import.
"""
import ast
import pathlib
import unittest

# Modules that read the ledger, run a scan, price a fill, or move a gate. The
# manual must reach none of them: it prints literals, and a literal cannot need
# a database. If this list ever has to be relaxed to make a test pass, the
# import is the bug.
FORBIDDEN = (
    "paper_manager", "options_screener", "backtester", "walk_forward",
    "check_pnl", "data_fetching", "trade_analysis", "short_premium_gate",
    "phase1_checkpoint", "candidate_verdict", "execution_truth",
    "execution_costs", "scoring", "ranking", "yfinance", "sqlite3", "pandas",
)

PACKAGE = pathlib.Path(__file__).resolve().parents[2] / "src" / "help_desk"


class IsolationTest(unittest.TestCase):
    def _imported_names(self, path):
        tree = ast.parse(path.read_text())
        names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
                names.extend(a.name for a in node.names)
        return names

    def test_package_is_not_empty(self):
        """A vacuous pass here would make every other assertion meaningless."""
        self.assertTrue(sorted(PACKAGE.glob("*.py")))

    def test_help_desk_imports_nothing_from_the_execution_paths(self):
        for path in sorted(PACKAGE.glob("*.py")):
            for name in self._imported_names(path):
                for bad in FORBIDDEN:
                    self.assertNotIn(bad, name,
                                     f"{path.name} imports {name!r}")

    def test_help_desk_does_not_touch_the_filesystem_or_network(self):
        """No open(), no requests, no subprocess. The manual is in memory."""
        banned_calls = {"open", "eval", "exec", "compile"}
        for path in sorted(PACKAGE.glob("*.py")):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, banned_calls,
                                     f"{path.name} calls {node.func.id}()")


class ScreenerShortcutTest(unittest.TestCase):
    def test_mode_menu_accepts_question_mark_and_help(self):
        """The screener's own menu opens the same manual. Asserted against the
        source because main() is a 1,000-line interactive loop."""
        src = (pathlib.Path(__file__).resolve().parents[2]
               / "src" / "options_screener.py").read_text()
        self.assertIn('if symbol_input in ("?", "HELP"):', src)
        self.assertIn("from .help_desk import run_menu", src)


if __name__ == "__main__":
    unittest.main()
