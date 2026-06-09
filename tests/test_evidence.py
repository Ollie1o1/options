"""Tests for src/evidence.py and the track-record renderer — pure, no network."""

import json
import os
import sqlite3
import tempfile
import unittest

from src.evidence import load_model_evidence, format_evidence_banner, GATE_TARGET_N


class TestLoadModelEvidence(unittest.TestCase):
    def _write_fixtures(self, d):
        wf = {
            "generated_at": "2026-05-29T11:27:48",
            "strategy": "Long Call",
            "n_total_trades": 94,
            "pooled_ic": 0.10214,
            "pooled_pvalue": 0.48029,
        }
        with open(os.path.join(d, "walk_forward_long_call_2026-05-29.json"), "w") as f:
            json.dump(wf, f)
        tsv = (
            "date\tweeks\tn\tic\tp\tdecision\n"
            "2026-05-29\t0\t0\t0.0000\t1.0000\tGATHERING\n"
            "2026-06-07\t1\t2\t0.0000\t1.0000\tGATHERING\n"
        )
        with open(os.path.join(d, "checkpoint_history.tsv"), "w") as f:
            f.write(tsv)

    def test_loads_from_fixtures(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_fixtures(d)
            ev = load_model_evidence(d)
            self.assertAlmostEqual(ev["pooled_ic"], 0.10214, places=4)
            self.assertAlmostEqual(ev["p_value"], 0.48029, places=4)
            self.assertEqual(ev["n_oos"], 94)
            self.assertEqual(ev["cohort_n"], 2)
            self.assertEqual(ev["gate_decision"], "GATHERING")
            self.assertEqual(ev["as_of"], "2026-06-07")

    def test_missing_files_safe_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            ev = load_model_evidence(d)
            self.assertIsNone(ev["pooled_ic"])
            self.assertIsNone(ev["p_value"])
            self.assertEqual(ev["n_oos"], 0)
            self.assertEqual(ev["cohort_n"], 0)
            self.assertEqual(ev["gate_decision"], "UNKNOWN")
            self.assertIsNone(ev["as_of"])

    def test_banner_with_evidence(self):
        ev = {
            "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94,
            "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-07",
        }
        banner = format_evidence_banner(ev)
        self.assertIn("EXPERIMENTAL", banner)
        self.assertIn("OOS IC +0.10 (p=0.48, n=94)", banner)
        self.assertIn(f"GATHERING (n=2/{GATE_TARGET_N})", banner)

    def test_banner_without_walkforward(self):
        ev = {
            "pooled_ic": None, "p_value": None, "n_oos": 0,
            "cohort_n": 0, "gate_decision": "UNKNOWN", "as_of": None,
        }
        banner = format_evidence_banner(ev)
        self.assertIn("OOS IC n/a", banner)


class TestTrackRecordRender(unittest.TestCase):
    def _seed_db(self):
        conn = sqlite3.connect(":memory:")
        conn.execute(
            """CREATE TABLE trades (
                entry_id INTEGER, date TEXT, ticker TEXT, expiration TEXT,
                strike REAL, type TEXT, entry_price REAL, strategy_name TEXT,
                status TEXT, exit_price REAL, exit_date TEXT, pnl_pct REAL,
                pnl_usd REAL, exit_reason TEXT, paper_only INTEGER
            )"""
        )
        rows = [
            # pnl_pct stored as a fraction (0.428 == +42.8%)
            (1, "2026-05-10", "AAPL", "2026-06-19", 150, "call", 3.5, "Long Call",
             "CLOSED", 5.0, "2026-05-20", 0.428, 150.0, "target", 0),
            (2, "2026-05-11", "MSFT", "2026-06-19", 400, "put", 5.0, "Long Put",
             "CLOSED", 2.5, "2026-05-21", -0.50, -250.0, "stop", 0),
            (3, "2026-05-12", "NVDA", "2026-06-19", 120, "call", 4.0, "Long Call",
             "OPEN", None, None, None, None, None, 0),
        ]
        conn.executemany(
            "INSERT INTO trades VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows
        )
        conn.commit()
        return conn

    def test_render_from_seeded_db(self):
        from scripts.publish_track_record import fetch_closed_trades, render_track_record

        conn = self._seed_db()
        closed = fetch_closed_trades(conn)
        self.assertEqual(len(closed), 2)  # only CLOSED rows

        ev = {
            "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94,
            "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-07",
        }
        md = render_track_record(closed, ev)
        # methodology caveat present
        self.assertIn("paper", md.lower())
        self.assertIn("VALIDATION_POWER", md)
        # summary stats
        self.assertIn("Closed trades", md)
        self.assertIn("50.0%", md)  # win rate: 1 of 2 winners
        # both tickers in the table
        self.assertIn("AAPL", md)
        self.assertIn("MSFT", md)
        # gate status surfaced
        self.assertIn("GATHERING", md)


if __name__ == "__main__":
    unittest.main()
