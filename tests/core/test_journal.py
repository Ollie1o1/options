import os, sys, tempfile, unittest
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from src.core.journal import append_entry

class TestJournal(unittest.TestCase):
    def test_appends_with_header_and_timestamp(self):
        d = tempfile.mkdtemp()
        p = os.path.join(d, "J.md")
        append_entry(p, title="2026-05-18 backfill", body="fixed 22 rows")
        append_entry(p, title="2026-05-19 followup", body="verified")
        text = open(p).read()
        self.assertIn("## 2026-05-18 backfill", text)
        self.assertIn("fixed 22 rows", text)
        self.assertIn("## 2026-05-19 followup", text)
        self.assertLess(text.index("2026-05-18"), text.index("2026-05-19"))

if __name__ == "__main__":
    unittest.main()
