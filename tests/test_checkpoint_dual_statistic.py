"""The checkpoint reports Pearson and rank IC side by side, and reports the
short-premium family beside the gate — without letting either touch the gate.

Pearson IC on floored option returns is driven by the handful of +100%
take-profits; the rank statistic is the one that survives that skew, and the
two routinely disagree in sign. Both are therefore printed everywhere one used
to be. None of it is a gate change: DECISIONS.md 2026-06-07 forbids silent gate
changes, and the redesign in docs/GATE_REDESIGN_SPEC.md is unsigned.
"""
import os
import sqlite3
import sys
import tempfile
import unittest
import unittest.mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.phase1_checkpoint import (  # noqa: E402
    _format_markdown,
    compute_checkpoint,
    short_premium_report,
    write_checkpoint,
)


# ── fixtures ────────────────────────────────────────────────────────────────

def _make_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        "CREATE TABLE trades (entry_id INTEGER PRIMARY KEY AUTOINCREMENT,"
        " date TEXT, strategy_name TEXT, status TEXT, paper_only INTEGER,"
        " quality_score REAL, pnl_pct REAL, pnl_usd REAL, capital_at_risk REAL)"
    )
    return conn


def _seed_long_calls(path, scores, returns, date="2026-06-01", capital_at_risk=500.0):
    conn = _make_db(path)
    conn.executemany(
        "INSERT INTO trades (date, strategy_name, status, paper_only,"
        " quality_score, pnl_pct, capital_at_risk)"
        f" VALUES ('{date}', 'Long Call', 'CLOSED', 0, ?, ?, ?)",
        [(float(s), float(r), capital_at_risk) for s, r in zip(scores, returns)],
    )
    conn.commit()
    conn.close()


def _rank_twin_cohorts(slope):
    """Two 60-trade cohorts with the *same* Pearson IC and opposite rank IC.

    Scores sit in two tight clumps, so two trades can be far apart in score
    *rank* while nearly identical in score *value*. Reversing the return order
    inside a clump therefore swings the rank IC hard while barely moving the
    Pearson one, and the two clumps' spacings are chosen so their Pearson
    effects cancel to machine precision. ``slope`` shifts both cohorts along
    the same line, which moves Pearson into a chosen gate band without ever
    separating the two cohorts' Pearson values.
    """
    n_lo, n_hi = 40, 20
    step_lo = 0.0001
    lo_returns = np.round(np.linspace(-0.40, 0.60, n_lo), 6)
    hi_returns = np.round(np.linspace(-0.25, 0.35, n_hi), 6)
    i_lo, i_hi = np.arange(n_lo), np.arange(n_hi)

    swing_lo = float(np.dot(i_lo, lo_returns[::-1] - lo_returns))
    swing_hi = float(np.dot(i_hi, hi_returns - hi_returns[::-1]))
    step_hi = -step_lo * swing_lo / swing_hi

    scores = np.concatenate([0.500 + step_lo * i_lo, 0.800 + step_hi * i_hi])
    aligned = np.concatenate([lo_returns, hi_returns[::-1]]) + slope * scores
    inverted = np.concatenate([lo_returns[::-1], hi_returns]) + slope * scores
    return scores, aligned, inverted


def _seed_short_premium(path, rows, conn=None, date="2026-06-01",
                        strategy="Bull Put", status="CLOSED", paper_only=1):
    """rows: (quality_score, pnl_usd, capital_at_risk)"""
    close = conn is None
    conn = conn or _make_db(path)
    conn.executemany(
        "INSERT INTO trades (date, strategy_name, status, paper_only,"
        " quality_score, pnl_usd, capital_at_risk)"
        f" VALUES ('{date}', '{strategy}', '{status}', {paper_only}, ?, ?, ?)",
        rows,
    )
    conn.commit()
    if close:
        conn.close()
    return conn


class _TmpDB(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db = os.path.join(self.tmp.name, "trades.db")
        self.out = os.path.join(self.tmp.name, "reports")

    def tearDown(self):
        self.tmp.cleanup()


# ── the decision must not notice the rank IC ────────────────────────────────

class TestDecisionIgnoresRankIC(_TmpDB):
    """Same Pearson, opposite rank structure, identical decision — under v1.

    Written when the reporting change shipped, to prove the gate still decided
    on exactly what it decided on before. Gate v2 (signed 2026-07-31)
    deliberately DOES read the rank IC, so this property now belongs to v1
    alone — and it is still worth pinning, because v1 must keep answering as it
    always did for its verdict to remain auditable beside v2's.
    """

    def _decide(self, scores, returns, today="2026-07-28"):
        path = os.path.join(self.tmp.name, f"c{abs(hash(returns.tobytes())) % 10**8}.db")
        _seed_long_calls(path, scores, returns)
        return compute_checkpoint(path, "2026-05-27", today=today, gate_version=1)

    def _assert_twins_decide_alike(self, slope, expected):
        scores, aligned, inverted = _rank_twin_cohorts(slope)
        a = self._decide(scores, aligned)
        b = self._decide(scores, inverted)

        # Precondition: same Pearson inputs to the decision block...
        self.assertEqual(a["n_trades"], b["n_trades"])
        self.assertEqual(a["weeks_elapsed"], b["weeks_elapsed"])
        self.assertAlmostEqual(a["ic_pearson"], b["ic_pearson"], places=12)
        self.assertAlmostEqual(a["p_pearson"], b["p_pearson"], places=12)
        # ...but genuinely different rank structure, opposite in sign.
        self.assertGreater(abs(a["ic_spearman"] - b["ic_spearman"]), 0.30)
        self.assertGreater(a["ic_spearman"], 0.0)
        self.assertLess(b["ic_spearman"], 0.0)

        self.assertEqual(a["decision"], expected,
                         f"expected {expected}, got {a['decision']} "
                         f"(IC={a['ic_pearson']:+.4f}, p={a['p_pearson']:.4f})")
        self.assertEqual(a["decision"], b["decision"],
                         "the rank IC moved the gate decision")

    def test_stop_band_is_rank_invariant(self):
        self._assert_twins_decide_alike(0.0, "STOP")

    def test_extend_band_is_rank_invariant(self):
        self._assert_twins_decide_alike(0.26, "EXTEND")

    def test_ready_band_is_rank_invariant(self):
        self._assert_twins_decide_alike(1.0, "READY")

    def test_short_premium_rows_cannot_move_the_decision(self):
        scores, aligned, _ = _rank_twin_cohorts(0.0)
        _seed_long_calls(self.db, scores, aligned)
        before = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                                    max_capital_at_risk=750.0)
        conn = sqlite3.connect(self.db)
        # A wildly profitable, perfectly rank-ordered credit book.
        _seed_short_premium(self.db, [(0.5 + i * 0.01, 200.0 + 10.0 * i, 500.0)
                                      for i in range(60)], conn=conn)
        conn.close()
        after = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                                   max_capital_at_risk=750.0)

        self.assertEqual(before["decision"], after["decision"])
        self.assertEqual(before["n_trades"], after["n_trades"])
        self.assertAlmostEqual(before["ic_pearson"], after["ic_pearson"], places=12)
        self.assertEqual(before["short_premium"]["n"], 0)
        self.assertEqual(after["short_premium"]["n"], 60)


# ── both statistics, everywhere one was shown ───────────────────────────────

class TestDualStatisticSurfaces(_TmpDB):
    def _result(self, **kw):
        scores, aligned, _ = _rank_twin_cohorts(0.0)
        _seed_long_calls(self.db, scores, aligned)
        return compute_checkpoint(self.db, "2026-05-27", today="2026-07-28", **kw)

    def test_markdown_shows_both_statistics_for_the_gate_cohort(self):
        md = _format_markdown(self._result())
        self.assertIn("Pearson IC (gate statistic)", md)
        self.assertIn("Spearman rank IC", md)

    def test_markdown_shows_both_statistics_for_the_affordable_subset(self):
        md = _format_markdown(self._result(max_capital_at_risk=750.0))
        subset = md.split("Affordable subset")[1]
        self.assertIn("Pearson IC", subset)
        self.assertIn("Spearman rank IC", subset)

    def test_affordable_subset_carries_a_rank_ic(self):
        r = self._result(max_capital_at_risk=750.0)
        self.assertIsNotNone(r["ic_spearman_affordable"])
        self.assertIsNotNone(r["p_spearman_affordable"])

    def test_sign_disagreement_is_called_out(self):
        # Pearson positive, rank negative on the inverted twin.
        scores, _, inverted = _rank_twin_cohorts(0.26)
        _seed_long_calls(self.db, scores, inverted)
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28")
        self.assertGreater(r["ic_pearson"], 0)
        self.assertLess(r["ic_spearman"], 0)
        self.assertIn("Statistics disagree in sign", _format_markdown(r))

    def test_a_degenerate_cohort_is_not_reported_as_agreement(self):
        """Both statistics at 0.000 is an absent measurement, not a match."""
        _seed_long_calls(self.db, [0.7, 0.8], [0.10, 0.20])
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28")
        self.assertEqual(r["ic_pearson"], 0.0)
        self.assertEqual(r["ic_spearman"], 0.0)
        md = _format_markdown(r)
        self.assertIn("Statistic agreement: n/a", md)
        self.assertNotIn("Statistics agree in sign.", md)

    def test_gate_status_file_reports_the_rank_ic(self):
        # Under v1 the rank IC was shown but explicitly not decisive.
        r = self._result(gate_version=1)
        self.assertEqual(r["decision"], "STOP")
        write_checkpoint(r, output_dir=self.out)
        with open(os.path.join(self.out, "GATE_STATUS.md")) as f:
            text = f.read()
        self.assertIn("Rank IC", text)
        self.assertIn("not the gate statistic", text)

    def test_gate_status_names_the_rank_ic_as_decisive_under_v2(self):
        # Under v2 it IS the statistic, and the file must say so — a reader
        # who cannot tell which number decided cannot audit the decision.
        # GATE_STATUS.md is only written for terminal verdicts, so seed a
        # cohort v2 genuinely stops on: score and return move opposite ways.
        import numpy as _np
        n = 60
        scores = _np.linspace(0.30, 0.95, n)
        returns = _np.linspace(0.90, -0.90, n)
        _seed_long_calls(self.db, scores, returns)
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28",
                               gate_version=2)
        self.assertEqual(r["decision"], "STOP")
        write_checkpoint(r, output_dir=self.out)
        with open(os.path.join(self.out, "GATE_STATUS.md")) as f:
            text = f.read()
        self.assertIn("THE gate statistic under v2", text)
        self.assertIn("rule v2", text)
        self.assertIn("v1 says", text)  # superseded verdict still visible


# ── history TSV: new columns at the end, old rows still readable ────────────

class TestHistoryTsvIsAppendCompatible(_TmpDB):
    def _write_checkpoint(self):
        scores, aligned, _ = _rank_twin_cohorts(0.0)
        _seed_long_calls(self.db, scores, aligned)
        r = compute_checkpoint(self.db, "2026-05-27", today="2026-07-28")
        write_checkpoint(r, output_dir=self.out)
        with open(os.path.join(self.out, "checkpoint_history.tsv")) as f:
            return f.read()

    def test_new_file_header_ends_with_the_rank_columns(self):
        header = self._write_checkpoint().splitlines()[0]
        self.assertEqual(
            header.split("\t"),
            ["date", "weeks", "n", "ic", "p", "decision", "spearman", "p_spearman"],
        )

    def test_appended_row_carries_both_statistics(self):
        row = self._write_checkpoint().splitlines()[1].split("\t")
        self.assertEqual(len(row), 8)
        float(row[6])
        float(row[7])

    def test_legacy_rows_survive_verbatim(self):
        os.makedirs(self.out, exist_ok=True)
        legacy = (
            "date\tweeks\tn\tic\tp\tdecision\n"
            "2026-06-01\t0\t7\t-0.6541\t0.1109\tGATHERING\n"
            "2026-07-29\t9\t70\t0.0485\t0.6902\tEXTEND\n"
        )
        path = os.path.join(self.out, "checkpoint_history.tsv")
        with open(path, "w") as f:
            f.write(legacy)

        lines = self._write_checkpoint().splitlines()
        self.assertIn("spearman", lines[0])
        # Historical rows are untouched, still six fields.
        self.assertEqual(lines[1], "2026-06-01\t0\t7\t-0.6541\t0.1109\tGATHERING")
        self.assertEqual(lines[2], "2026-07-29\t9\t70\t0.0485\t0.6902\tEXTEND")
        self.assertEqual(len(lines[3].split("\t")), 8)

    def test_evidence_reads_a_mixed_width_history(self):
        from src.evidence import load_model_evidence

        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "checkpoint_history.tsv"), "w") as f:
            f.write("date\tweeks\tn\tic\tp\tdecision\tspearman\tp_spearman\n"
                    "2026-07-21\t7\t43\t0.1057\t0.5001\tGATHERING\n"
                    "2026-07-29\t9\t70\t0.0485\t0.6902\tEXTEND\t-0.0200\t0.8730\n")
        ev = load_model_evidence(self.out)
        self.assertEqual(ev["cohort_n"], 70)
        self.assertEqual(ev["gate_decision"], "EXTEND")
        self.assertAlmostEqual(ev["cohort_ic_pearson"], 0.0485, places=6)
        self.assertAlmostEqual(ev["cohort_ic_spearman"], -0.02, places=6)

    def test_evidence_reads_a_legacy_only_history(self):
        from src.evidence import load_model_evidence

        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "checkpoint_history.tsv"), "w") as f:
            f.write("date\tweeks\tn\tic\tp\tdecision\n"
                    "2026-07-29\t9\t70\t0.0485\t0.6902\tEXTEND\n")
        ev = load_model_evidence(self.out)
        self.assertEqual(ev["cohort_n"], 70)
        self.assertAlmostEqual(ev["cohort_ic_pearson"], 0.0485, places=6)
        self.assertIsNone(ev["cohort_ic_spearman"])

    def test_header_rewrite_is_atomic_and_leaves_no_debris(self):
        """The history cannot be regenerated from anything, so the rewrite must
        never be able to leave it truncated."""
        from src.phase1_checkpoint import _ensure_history_header

        os.makedirs(self.out, exist_ok=True)
        path = os.path.join(self.out, "checkpoint_history.tsv")
        legacy = ("date\tweeks\tn\tic\tp\tdecision\n"
                  "2026-07-29\t9\t70\t0.0485\t0.6902\tEXTEND\n")
        with open(path, "w") as f:
            f.write(legacy)

        import pathlib
        real_replace = os.replace
        seen = {}

        def _spy(src, dst):
            seen["src"], seen["dst"] = str(src), str(dst)
            return real_replace(src, dst)

        with unittest.mock.patch("src.phase1_checkpoint.os.replace", _spy):
            _ensure_history_header(pathlib.Path(path))

        # Renamed into place from a sibling temp file, not written over.
        self.assertEqual(seen["dst"], path)
        self.assertNotEqual(seen["src"], path)
        self.assertEqual(os.path.dirname(seen["src"]), self.out)
        # No temp files survive.
        self.assertEqual([f for f in os.listdir(self.out) if f.endswith(".tmp")], [])
        with open(path) as f:
            lines = f.read().splitlines()
        self.assertIn("spearman", lines[0])
        self.assertEqual(lines[1], "2026-07-29\t9\t70\t0.0485\t0.6902\tEXTEND")

    def test_a_failed_atomic_write_removes_its_temp_file(self):
        from src.phase1_checkpoint import _atomic_write

        import pathlib
        os.makedirs(self.out, exist_ok=True)
        path = pathlib.Path(self.out) / "checkpoint_history.tsv"
        path.write_text("date\tweeks\tn\tic\tp\tdecision\n")

        def _boom(src, dst):
            raise OSError("rename failed")

        with unittest.mock.patch("src.phase1_checkpoint.os.replace", _boom):
            with self.assertRaises(OSError):
                _atomic_write(path, "clobbered\n")

        # Original intact, no debris left behind.
        self.assertEqual(path.read_text(), "date\tweeks\tn\tic\tp\tdecision\n")
        self.assertEqual(os.listdir(self.out), ["checkpoint_history.tsv"])

    def test_evidence_reads_wide_rows_under_a_legacy_header(self):
        """A clone whose header predates the column still finds the rank IC."""
        from src.evidence import load_model_evidence

        os.makedirs(self.out, exist_ok=True)
        with open(os.path.join(self.out, "checkpoint_history.tsv"), "w") as f:
            f.write("date\tweeks\tn\tic\tp\tdecision\n"
                    "2026-07-29\t9\t70\t0.0485\t0.6902\tEXTEND\t-0.0200\t0.8730\n")
        ev = load_model_evidence(self.out)
        self.assertEqual(ev["cohort_n"], 70)
        self.assertAlmostEqual(ev["cohort_ic_spearman"], -0.02, places=6)


# ── short-premium cohort ────────────────────────────────────────────────────

class TestShortPremiumCohort(_TmpDB):
    def test_counts_the_whole_credit_family(self):
        conn = _make_db(self.db)
        for strategy in ("Bull Put", "Bear Call", "Short Put"):
            _seed_short_premium(self.db, [(0.7, 50.0, 500.0)], conn=conn,
                                strategy=strategy)
        _seed_short_premium(self.db, [(0.7, 50.0, 500.0)], conn=conn,
                            strategy="Iron Condor")
        _seed_short_premium(self.db, [(0.7, 50.0, 500.0)], conn=conn,
                            strategy="Long Call")
        conn.close()
        sp = short_premium_report(self.db, "2026-05-27")
        self.assertEqual(sp["n"], 3)

    def test_excludes_open_pre_window_and_unaffordable_rows(self):
        conn = _make_db(self.db)
        _seed_short_premium(self.db, [(0.7, 50.0, 500.0)], conn=conn)
        _seed_short_premium(self.db, [(0.7, 50.0, 500.0)], conn=conn, status="OPEN")
        _seed_short_premium(self.db, [(0.7, 50.0, 500.0)], conn=conn, date="2026-05-01")
        _seed_short_premium(self.db, [(0.7, 50.0, 9000.0)], conn=conn)
        conn.close()
        self.assertEqual(short_premium_report(self.db, "2026-05-27",
                                              max_capital_at_risk=750.0)["n"], 1)
        # Without a cap the affordable filter is the only one that lifts.
        self.assertEqual(short_premium_report(self.db, "2026-05-27")["n"], 2)

    def test_includes_paper_only_rows(self):
        """The whole family is logged paper_only=1; filtering it would empty
        the block permanently."""
        _seed_short_premium(self.db, [(0.7, 50.0, 500.0)], paper_only=1)
        self.assertEqual(short_premium_report(self.db, "2026-05-27")["n"], 1)

    def test_return_on_risk_is_capital_weighted_with_a_median_beside_it(self):
        # One big loser and three small winners: the weighted and median
        # figures must disagree, which is the whole reason both are reported.
        _seed_short_premium(self.db, [
            (0.6, 60.0, 300.0), (0.7, 60.0, 300.0), (0.8, 60.0, 300.0),
            (0.9, -600.0, 3000.0),
        ])
        sp = short_premium_report(self.db, "2026-05-27")
        self.assertEqual(sp["n"], 4)
        self.assertAlmostEqual(sp["sum_pnl"], -420.0, places=6)
        self.assertAlmostEqual(sp["sum_capital_at_risk"], 3900.0, places=6)
        self.assertAlmostEqual(sp["ror_sum"], -420.0 / 3900.0, places=9)
        self.assertAlmostEqual(sp["ror_median"], 0.2, places=9)
        self.assertLess(sp["ror_sum"], 0.0)  # weighted and median disagree

    def test_rank_ic_is_scored_against_return_on_risk(self):
        # Return-on-risk falls as the score rises; dollar P&L rises. Only a
        # RoR-based IC gets the sign right.
        _seed_short_premium(self.db, [
            (0.5, 100.0, 200.0), (0.6, 150.0, 500.0),
            (0.7, 200.0, 1000.0), (0.8, 250.0, 2000.0),
        ])
        sp = short_premium_report(self.db, "2026-05-27")
        self.assertAlmostEqual(sp["ic_spearman"], -1.0, places=6)
        self.assertLess(sp["ic_pearson"], 0.0)

    def test_ic_is_none_below_three_trades(self):
        _seed_short_premium(self.db, [(0.6, 10.0, 100.0), (0.7, 20.0, 100.0)])
        sp = short_premium_report(self.db, "2026-05-27")
        self.assertEqual(sp["n"], 2)
        self.assertIsNone(sp["ic_spearman"])
        self.assertIsNotNone(sp["ror_sum"])

    def test_empty_cohort_reports_zero_not_a_crash(self):
        _make_db(self.db).close()
        sp = short_premium_report(self.db, "2026-05-27")
        self.assertEqual(sp["n"], 0)
        self.assertIsNone(sp["ror_sum"])

    def test_a_db_without_the_v16_columns_degrades_quietly(self):
        conn = sqlite3.connect(self.db)
        conn.execute("CREATE TABLE trades (date TEXT, strategy_name TEXT,"
                     " status TEXT, quality_score REAL, pnl_pct REAL)")
        conn.commit()
        conn.close()
        self.assertEqual(short_premium_report(self.db, "2026-05-27")["n"], 0)

    def test_markdown_block_is_labelled_reporting_only(self):
        scores, aligned, _ = _rank_twin_cohorts(0.0)
        _seed_long_calls(self.db, scores, aligned)
        conn = sqlite3.connect(self.db)
        _seed_short_premium(self.db, [(0.5 + i * 0.01, 20.0 * i - 100.0, 400.0)
                                      for i in range(10)], conn=conn)
        conn.close()
        md = _format_markdown(compute_checkpoint(self.db, "2026-05-27",
                                                 today="2026-07-28",
                                                 max_capital_at_risk=750.0))
        self.assertIn("Short-premium cohort — REPORTING ONLY, not a gate", md)
        self.assertIn("Rank IC (quality_score vs return-on-risk)", md)
        self.assertIn("Return on risk (sum P&L / sum capital at risk)", md)
        self.assertIn("Median per-trade return on risk", md)
        self.assertIn("REPORTING ONLY — not a gate.", md)
        # It must sit after the gate decision, never inside it.
        self.assertLess(md.index("## Gate decision"),
                        md.index("## Short-premium cohort"))


# ── evidence banner ─────────────────────────────────────────────────────────

class TestEvidenceBannerDualStatistic(unittest.TestCase):
    def _ev(self, **kw):
        ev = {"pooled_ic": 0.10, "p_value": 0.48, "n_oos": 94, "cohort_n": 70,
              "gate_decision": "EXTEND", "as_of": "2026-07-29"}
        ev.update(kw)
        return ev

    def test_banner_shows_both_cohort_statistics(self):
        from src.evidence import format_evidence_banner

        banner = format_evidence_banner(
            self._ev(cohort_ic_pearson=0.0485, cohort_ic_spearman=-0.02))
        self.assertIn("cohort IC +0.049 pearson / -0.020 rank", banner)

    def test_banner_says_rank_na_when_the_history_predates_the_column(self):
        from src.evidence import format_evidence_banner

        banner = format_evidence_banner(
            self._ev(cohort_ic_pearson=0.0485, cohort_ic_spearman=None))
        self.assertIn("+0.049 pearson / rank n/a", banner)

    def test_banner_omits_the_segment_when_no_checkpoint_exists(self):
        from src.evidence import format_evidence_banner

        banner = format_evidence_banner(self._ev())
        self.assertNotIn("cohort IC", banner)
        self.assertIn("EXTEND (n=70/", banner)


if __name__ == "__main__":
    unittest.main()
