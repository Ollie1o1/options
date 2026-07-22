"""Tests for the CDR (Canadian Depositary Receipt) lookup module
(src/longterm/cdr.py) — a pure, local JSON lookup, zero network cost."""
import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.longterm import cdr as CDR


class TestLoadCdrMap(unittest.TestCase):
    def test_missing_file_returns_empty_dict(self):
        result = CDR.load_cdr_map(path="/nonexistent/path/cdr_map.json")
        self.assertEqual(result, {})

    def test_malformed_json_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{not valid json")
            path = f.name
        try:
            result = CDR.load_cdr_map(path=path)
        finally:
            os.unlink(path)
        self.assertEqual(result, {})

    def test_non_dict_json_returns_empty_dict(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(["AAPL", "MSFT"], f)
            path = f.name
        try:
            result = CDR.load_cdr_map(path=path)
        finally:
            os.unlink(path)
        self.assertEqual(result, {})

    def test_source_key_excluded_from_map(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"_source": "note", "AAPL": "AAPL"}, f)
            path = f.name
        try:
            result = CDR.load_cdr_map(path=path)
        finally:
            os.unlink(path)
        self.assertNotIn("_SOURCE", result)
        self.assertEqual(result, {"AAPL": "AAPL"})

    def test_keys_and_values_uppercased(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"ko": "cola"}, f)
            path = f.name
        try:
            result = CDR.load_cdr_map(path=path)
        finally:
            os.unlink(path)
        self.assertEqual(result, {"KO": "COLA"})


class TestCdrFor(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump({"KO": "COLA", "AAPL": "AAPL"}, self.tmp)
        self.tmp.close()

    def tearDown(self):
        os.unlink(self.tmp.name)

    def test_known_ticker_with_different_cdr_symbol(self):
        self.assertEqual(CDR.cdr_for("KO", path=self.tmp.name), "COLA")

    def test_known_ticker_with_matching_cdr_symbol(self):
        self.assertEqual(CDR.cdr_for("AAPL", path=self.tmp.name), "AAPL")

    def test_case_insensitive_lookup(self):
        self.assertEqual(CDR.cdr_for("ko", path=self.tmp.name), "COLA")

    def test_unknown_ticker_returns_none(self):
        self.assertIsNone(CDR.cdr_for("ZZZZ", path=self.tmp.name))


if __name__ == "__main__":
    unittest.main()
