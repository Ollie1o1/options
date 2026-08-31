"""Tests for src/evidence.py and the track-record renderer — pure, no network."""

import json
import os
import sqlite3
import tempfile
import unittest
from datetime import date

from src.evidence import (
    load_model_evidence, format_evidence_banner, GATE_TARGET_N,
    WALK_FORWARD_STALE_DAYS,
)


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
            # wf_as_of is the walk-forward artifact's OWN date — unlike
            # as_of, it is never bumped forward by the (weekly, so almost
            # always more recent) checkpoint date. Staleness flagging reads
            # this field specifically.
            self.assertEqual(ev["wf_as_of"], "2026-05-29T11:27:48")

    def test_missing_files_safe_defaults(self):
        with tempfile.TemporaryDirectory() as d:
            ev = load_model_evidence(d)
            self.assertIsNone(ev["pooled_ic"])
            self.assertIsNone(ev["p_value"])
            self.assertEqual(ev["n_oos"], 0)
            self.assertEqual(ev["cohort_n"], 0)
            self.assertEqual(ev["gate_decision"], "UNKNOWN")
            self.assertIsNone(ev["as_of"])
            self.assertIsNone(ev["wf_as_of"])

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


class TestWalkForwardStaleness(unittest.TestCase):
    """The banner shows the walk-forward artifact's own age and flags it once
    it's older than WALK_FORWARD_STALE_DAYS — the artifact only regenerates
    monthly (src/maintenance.py due_walk_forward), so a stale number left
    unlabeled would read as current evidence."""

    def _ev(self, wf_as_of):
        return {
            "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94,
            "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-07",
            "wf_as_of": wf_as_of,
            "cohort_ic_pearson": 0.048, "cohort_ic_spearman": -0.020,
        }

    def test_shows_as_of_date_and_age(self):
        ev = self._ev("2026-05-29T11:27:48")
        banner = format_evidence_banner(ev, today=date(2026, 7, 31))
        self.assertIn("as of 2026-05-29", banner)
        self.assertIn("63d old", banner)  # 2026-05-29 -> 2026-07-31

    def test_flags_when_older_than_30_days(self):
        ev = self._ev("2026-05-29T11:27:48")
        banner = format_evidence_banner(ev, today=date(2026, 7, 31))
        self.assertIn(f"STALE >{WALK_FORWARD_STALE_DAYS}d", banner)

    def test_no_flag_at_exactly_30_days(self):
        ev = self._ev("2026-06-01")
        banner = format_evidence_banner(ev, today=date(2026, 7, 1))  # 30 days
        self.assertNotIn("STALE", banner)
        self.assertIn("30d old", banner)

    def test_no_flag_when_fresh(self):
        ev = self._ev("2026-07-25")
        banner = format_evidence_banner(ev, today=date(2026, 7, 31))
        self.assertNotIn("STALE", banner)
        self.assertIn("6d old", banner)

    def test_no_age_segment_when_wf_as_of_unrecorded(self):
        ev = self._ev(None)
        banner = format_evidence_banner(ev, today=date(2026, 7, 31))
        self.assertNotIn("walk-forward as of", banner)
        self.assertNotIn("STALE", banner)
        # cohort segment still renders on its own second line
        self.assertIn("cohort IC", banner)

    def test_lines_stay_within_ui_banner_100_char_budget(self):
        # A reviewer flagged the pre-restructure single-line banner at ~123
        # chars against ui.banner's 100-char rule; adding the as_of date only
        # grows it further, so the banner is two lines instead. Every line
        # must independently respect the budget.
        ev = self._ev("2026-05-29T11:27:48")
        banner = format_evidence_banner(ev, today=date(2026, 7, 31))
        lines = banner.split("\n")
        self.assertGreaterEqual(len(lines), 2)
        for ln in lines:
            self.assertLessEqual(len(ln), 100, msg=f"line too long ({len(ln)}): {ln!r}")

    def test_single_line_when_nothing_extra_to_report(self):
        ev = {
            "pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94,
            "cohort_n": 2, "gate_decision": "GATHERING", "as_of": "2026-06-07",
            "wf_as_of": None, "cohort_ic_pearson": None, "cohort_ic_spearman": None,
        }
        banner = format_evidence_banner(ev, today=date(2026, 7, 31))
        self.assertNotIn("\n", banner)


class TestBannerSurfacesRefusal(unittest.TestCase):
    """Tasks 1-2 added a refusal path: when purging leaves too few usable
    folds, run_walk_forward writes an artifact with every statistic None
    instead of an IC (see src/walk_forward.py::_refused_summary). evidence.py
    reads the NEWEST walk_forward_*.json, so a refusal that rendered as an
    empty slot would look like 'not computed yet' rather than 'computed and
    refused' — worse than the pre-purge behaviour, since it also silently
    keeps whatever older non-refused artifact off the banner."""

    def _write(self, d, name, payload):
        with open(os.path.join(d, name), "w") as fh:
            json.dump(payload, fh)

    def test_a_refused_walk_forward_is_named_not_left_blank(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "walk_forward_long_call_2026-08-29.json", {
                "generated_at": "2026-08-29T10:00:00",
                "strategy": "Long Call", "n_total_trades": 253,
                "n_folds": 0, "n_folds_attempted": 15, "n_folds_dropped": 15,
                "refused": True,
                "refused_reason": (
                    "only 0 of 15 folds kept 54+ training trades after "
                    "purging (minimum 3); widen train_size or wait for more "
                    "closed trades"
                ),
                "pooled_ic": None, "pooled_pvalue": None,
                "fold_ic_mean": None, "folds_ic_positive": None,
            })
            ev = load_model_evidence(reports_dir=d)
            self.assertTrue(ev["wf_refused"])
            self.assertIsNone(ev["pooled_ic"])
            self.assertIn("only 0 of 15 folds", ev["wf_refused_reason"])
            text = format_evidence_banner(ev, today=date(2026, 8, 29))
            self.assertIn("refused", text.lower())
            for ln in text.split("\n"):
                self.assertLessEqual(len(ln), 100,
                                      msg=f"line too long ({len(ln)}): {ln!r}")

    def test_a_refused_walk_forward_reports_zero_n_oos(self):
        # Important-2 regression: n_oos used to be set unconditionally from
        # n_total_trades, so a refusal (which scores ZERO trades out of
        # sample) reported the strategy's whole book size as if it were the
        # walk-forward's own trade count — lending a refusal false weight
        # in any consumer that renders n_oos without also checking
        # wf_refused (see src/morning/render.py, src/tearsheet/render.py).
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "walk_forward_short_put_2026-08-29.json", {
                "generated_at": "2026-08-29T10:00:00",
                "strategy": "Short Put", "n_total_trades": 108,
                "n_folds": 0, "n_folds_attempted": 0, "n_folds_dropped": 0,
                "refused": True,
                "refused_reason": (
                    "no fold could be formed: 108 trades < "
                    "train_size+test_size=110"
                ),
                "pooled_ic": None, "pooled_pvalue": None,
                "fold_ic_mean": None, "folds_ic_positive": None,
            })
            ev = load_model_evidence(reports_dir=d)
            self.assertTrue(ev["wf_refused"])
            self.assertEqual(ev["n_oos"], 0,
                             "a refusal scored nothing out of sample")

    def test_a_normal_walk_forward_is_not_marked_refused(self):
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "walk_forward_long_call_2026-08-29.json", {
                "generated_at": "2026-08-29T10:00:00",
                "strategy": "Long Call", "n_total_trades": 253,
                "n_folds": 12, "refused": False, "refused_reason": None,
                "pooled_ic": 0.11, "pooled_pvalue": 0.04,
                "fold_ic_mean": 0.09, "folds_ic_positive": 8,
            })
            ev = load_model_evidence(reports_dir=d)
            self.assertFalse(ev["wf_refused"])
            self.assertIsNone(ev["wf_refused_reason"])
            self.assertAlmostEqual(ev["pooled_ic"], 0.11)
            text = format_evidence_banner(ev, today=date(2026, 8, 29))
            self.assertNotIn("refused", text.lower())

    def test_refusal_line_stays_in_budget_alongside_age_and_cohort(self):
        # The worst case: a refusal, a stale walk-forward date, AND a cohort
        # IC all present together — the three segments cannot share one line
        # (age + cohort alone already fill it), so this pins the refusal
        # reason to its own line and checks it independently.
        ev = {
            "pooled_ic": None, "p_value": None, "n_oos": 253,
            "cohort_n": 12, "gate_decision": "GATHERING", "as_of": "2026-05-29",
            "wf_as_of": "2026-05-29T11:27:48",
            "wf_refused": True,
            "wf_refused_reason": (
                "only 0 of 15 folds kept 54+ training trades after purging "
                "(minimum 3); widen train_size or wait for more closed trades"
            ),
            "cohort_ic_pearson": 0.048, "cohort_ic_spearman": -0.020,
        }
        text = format_evidence_banner(ev, today=date(2026, 8, 29))
        lines = text.split("\n")
        self.assertEqual(len(lines), 3)
        for ln in lines:
            self.assertLessEqual(len(ln), 100,
                                  msg=f"line too long ({len(ln)}): {ln!r}")
        self.assertIn("refused", lines[-1].lower())

    def test_missing_refused_key_treated_as_not_refused(self):
        # Artifacts written before Tasks 1-2 have no "refused" key at all.
        with tempfile.TemporaryDirectory() as d:
            self._write(d, "walk_forward_long_call_2026-05-29.json", {
                "generated_at": "2026-05-29T11:27:48",
                "strategy": "Long Call", "n_total_trades": 94,
                "pooled_ic": 0.10214, "pooled_pvalue": 0.48029,
            })
            ev = load_model_evidence(reports_dir=d)
            self.assertFalse(ev["wf_refused"])
            self.assertIsNone(ev["wf_refused_reason"])


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
