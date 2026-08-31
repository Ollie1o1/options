"""Tests for src/walk_forward.py — walk-forward OOS IC harness.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_walk_forward -v
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import numpy as np

from src.backtest_optimizer import WEIGHT_KEYS
from src.walk_forward import (
    MIN_FOLDS, MIN_TRAIN_AFTER_PURGE, TRIALS_PER_FOLD, Trade, _as_date,
    _format_markdown, build_folds, load_trades, purge_overlapping,
    run_walk_forward,
)
from src.paper_manager import PaperManager

# DB column names in insertion order (matches _COMPONENT_COLS in walk_forward)
_WEIGHT_KEY_TO_COL = {
    "pop":              "pop_score",
    "em_realism":       "em_realism_score",
    "iv_mispricing":    "iv_mispricing_score",
    "rr":               "rr_score",
    "momentum":         "momentum_score",
    "iv_rank":          "iv_rank_score",
    "liquidity":        "liquidity_score",
    "catalyst":         "catalyst_score",
    "theta":            "theta_score",
    "ev":               "ev_score",
    "trader_pref":      "trader_pref_score",
    "iv_edge":          "iv_edge_score",
    "skew_align":       "skew_align_score",
    "gamma_theta":      "gamma_theta_score",
    "pcr":              "pcr_score",
    "gex":              "gex_score",
    "oi_change":        "oi_change_score",
    "sentiment":        "sentiment_score_norm",
    "option_rvol":      "option_rvol_score",
    "vrp":              "vrp_score",
    "gamma_pin":        "gamma_pin_score",
    "max_pain":         "max_pain_score",
    "iv_velocity":      "iv_velocity_score",
    "gamma_magnitude":  "gamma_magnitude_score",
    "vega_risk":        "vega_risk_score",
    "term_structure":   "term_structure_score",
    "spread":           "spread_score",
}

_COMPONENT_DB_COLS = [_WEIGHT_KEY_TO_COL[k] for k in WEIGHT_KEYS]

# Index of the signal-carrying column in the components array.
# We inject signal via pop_score (WEIGHT_KEYS index 0).
_SIGNAL_COL_IDX = WEIGHT_KEYS.index("pop")


def _seed_db(
    db_path: str,
    n_trades: int,
    ic_target: float = 0.0,
    seed: int = 42,
    hold_days: int = 0,
) -> None:
    """Create a fully migrated paper_trades.db and insert n_trades closed Long Call rows.

    Signal construction:
      - pop_score is drawn from Uniform[0,1]; all other component columns = 0.5.
      - pnl_pct = ic_target * pop_score + sqrt(1 - ic_target**2) * noise,
        where noise ~ N(0, 0.3). This gives a theoretical Pearson IC of
        approximately ic_target between pop_score and pnl_pct.
      - paper_only = 0 for all rows (eligible for validation cohort).
      - Dates span forward from 2023-01-02 in daily steps.
      - exit_date = entry_date + hold_days (default 0, a zero-day hold).
    """
    # Initialise schema via PaperManager (runs all migrations).
    pm = PaperManager(db_path=db_path, config_path="config.json")

    rng = np.random.default_rng(seed)
    pop_vals = rng.uniform(0.0, 1.0, n_trades)
    noise = rng.normal(0.0, 0.30, n_trades)
    coef = float(ic_target)
    noise_scale = float(np.sqrt(max(1.0 - ic_target**2, 0.0)))
    pnl_vals = coef * pop_vals + noise_scale * noise

    score_cols = ", ".join(_COMPONENT_DB_COLS)
    placeholders = ", ".join(["?"] * len(_COMPONENT_DB_COLS))

    with sqlite3.connect(db_path) as conn:
        for i in range(n_trades):
            trade_date = f"2023-01-{(i % 28) + 1:02d}" if i < 28 else f"2023-{(i // 28) + 1:02d}-{(i % 28) + 1:02d}"
            # Simpler date: just increment from a base
            from datetime import date, timedelta
            base = date(2023, 1, 2)
            entry = base + timedelta(days=i)
            entry_date = entry.isoformat()
            exit_date = (entry + timedelta(days=hold_days)).isoformat()

            # Build component values: signal in pop_score, rest neutral
            comp_vals = [0.5] * len(_COMPONENT_DB_COLS)
            comp_vals[_SIGNAL_COL_IDX] = float(pop_vals[i])

            sql = (
                f"INSERT INTO trades "
                f"(date, ticker, expiration, strike, type, entry_price, quality_score, "
                f"strategy_name, status, exit_price, exit_date, pnl_pct, pnl_usd, "
                f"paper_only, {score_cols}) "
                f"VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, {placeholders})"
            )
            params = (
                entry_date,
                f"TICK{i:04d}",
                "2024-01-19",
                100.0 + i,
                "call",
                2.00,
                float(pop_vals[i]),
                "Long Call",
                "CLOSED",
                0.50,
                exit_date,
                float(pnl_vals[i]),
                float(pnl_vals[i]) * 200.0,
                0,
                *comp_vals,
            )
            conn.execute(sql, params)
        conn.commit()


class TestLeakPrevention(unittest.TestCase):
    """Fold boundaries must be strictly non-overlapping."""

    def test_five_folds_no_leak(self):
        """94 trades, train=44, test=10, step=10 => 5 folds; no train/test overlap."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=94)
            trades = load_trades(db, strategy="Long Call")
            folds = list(build_folds(trades, train_size=44, test_size=10, step=10))

            self.assertEqual(len(folds), 5, f"Expected 5 folds, got {len(folds)}")
            for idx, (train_ids, test_ids) in enumerate(folds):
                overlap = set(train_ids) & set(test_ids)
                self.assertSetEqual(
                    overlap,
                    set(),
                    f"Fold {idx} has {len(overlap)} leaking rowids: {overlap}",
                )


class TestRecoversKnownIC(unittest.TestCase):
    """With synthetic signal, pooled OOS IC should be positive."""

    def test_positive_pooled_ic(self):
        """200 trades with ic_target=0.15 => pooled_ic > 0 and n_folds >= 4."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=200, ic_target=0.15, seed=7)
            result = run_walk_forward(
                db_path=db,
                strategy="Long Call",
                train_size=80,
                test_size=20,
                step=20,
            )
            self.assertGreaterEqual(
                result["n_folds"],
                4,
                f"Expected >= 4 folds, got {result['n_folds']}",
            )
            self.assertGreater(
                result["pooled_ic"],
                0.0,
                f"Expected positive pooled_ic, got {result['pooled_ic']:.4f}",
            )


class TestPaperOnlyExclusion(unittest.TestCase):
    """Rows with paper_only=1 must not appear in load_trades output."""

    def test_paper_only_excluded(self):
        """20 trades inserted; first 10 flagged paper_only=1 => only 10 returned."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=20)

            # Flag the first 10 rows as paper_only
            with sqlite3.connect(db) as conn:
                rows = conn.execute(
                    "SELECT rowid FROM trades ORDER BY rowid LIMIT 10"
                ).fetchall()
                ids = [r[0] for r in rows]
                conn.execute(
                    f"UPDATE trades SET paper_only=1 WHERE rowid IN ({','.join('?' * len(ids))})",
                    ids,
                )
                conn.commit()

            trades = load_trades(db, strategy="Long Call")
            self.assertEqual(
                len(trades),
                10,
                f"Expected 10 non-paper-only trades, got {len(trades)}",
            )
            # Confirm none of the returned rowids are in the paper_only set
            returned_ids = {t.rowid for t in trades}
            self.assertFalse(
                returned_ids & set(ids),
                "Some paper_only=1 rows leaked into load_trades output",
            )


class TestWritesReportFiles(unittest.TestCase):
    """run_walk_forward must write .json and .md into output_dir."""

    def test_report_files_created(self):
        """300 trades, train=120 => a genuine successful run writes both files.

        train_size=44 (the pre-purge default) drops every fold below the
        MIN_TRAIN_AFTER_PURGE floor and refuses; that would make this test
        exercise the refusal path while still claiming to test a normal
        write, since file-existence and key-presence assertions pass on
        either branch. train=120 with 300 trades clears the floor on every
        fold, so `refused` is asserted False to pin the premise.
        """
        import glob
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            out_dir = os.path.join(tmp, "reports")
            _seed_db(db, n_trades=300, seed=5)
            result = run_walk_forward(
                db_path=db,
                strategy="Long Call",
                train_size=120,
                test_size=10,
                step=10,
                output_dir=out_dir,
            )
            self.assertFalse(result["refused"],
                              "this test means to exercise a successful run")
            json_files = glob.glob(os.path.join(out_dir, "walk_forward_*.json"))
            md_files = glob.glob(os.path.join(out_dir, "walk_forward_*.md"))
            self.assertTrue(
                len(json_files) >= 1,
                f"No walk_forward_*.json found in {out_dir}",
            )
            self.assertTrue(
                len(md_files) >= 1,
                f"No walk_forward_*.md found in {out_dir}",
            )
            self.assertIn("json_path", result)
            self.assertIn("md_path", result)

    def test_per_fold_markdown_table_carries_n_train_purged(self):
        # Minor-5 regression: the per-fold table used to omit n_train_purged
        # — this branch's headline number — even though the JSON carried it
        # per fold. hold_days=10 guarantees a nonzero purge to check the
        # value actually lands in the right column, not just the header.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=300, seed=5, hold_days=10)
            result = run_walk_forward(db_path=db, strategy="Long Call",
                                      train_size=120, test_size=10, step=10)
            self.assertFalse(result["refused"])
            md = _format_markdown(result)
            self.assertIn("n_train_purged", md)
            first_fold = result["folds"][0]
            self.assertGreater(first_fold["n_train_purged"], 0)
            self.assertIn(
                f"| {first_fold['fold']} | {first_fold['n_train']} | "
                f"{first_fold['n_train_purged']} | {first_fold['n_test']} | ",
                md)


def _t(rowid: int, entry: str, exit_: str) -> Trade:
    return Trade(rowid=rowid, entry_date=entry, exit_date=exit_,
                 pnl_pct=0.0, components=np.zeros(len(WEIGHT_KEYS)))


# _seed_db's fixture inserts exactly one trade per calendar day starting
# 2023-01-02 (see its docstring), so trade index i (0-based, matching
# build_folds' slice position) sits on this date and at this rowid.
def _seed_base_date():
    from datetime import date
    return date(2023, 1, 2)


def _day(i: int) -> str:
    from datetime import timedelta
    return (_seed_base_date() + timedelta(days=i)).isoformat()


def _rowid_of(i: int) -> int:
    return i + 1


def _force_purge(db_path: str, lo_i: int, hi_i: int, new_exit_i: int) -> None:
    """Push exit_date for trade indices [lo_i, hi_i] out to day `new_exit_i`.

    A single global `hold_days` in `_seed_db` purges every fold identically
    (or none of them): with one trade per calendar day and contiguous
    train/test slices, the purge count for fold k only depends on
    `min(hold_days, train_size)`, which is the same for every k. There is no
    way to make a global `hold_days` drop SOME folds but not others through
    that fixture, so tests that need a mixed outcome patch specific rows'
    exit_date directly instead, targeted so only the ONE fold whose test
    window starts at day `new_exit_i` sees the overlap (see call sites).
    """
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE trades SET exit_date = ? WHERE rowid BETWEEN ? AND ?",
            (_day(new_exit_i), _rowid_of(lo_i), _rowid_of(hi_i)))
        conn.commit()


class TestPurgeOverlapping(unittest.TestCase):
    """A training trade still open during the test window leaks its outcome."""

    def _test_block(self):
        # test window spans 2026-03-10 .. 2026-03-20
        return [_t(100, "2026-03-10", "2026-03-15"),
                _t(101, "2026-03-12", "2026-03-20")]

    def test_a_train_trade_closing_before_the_window_is_kept(self):
        train = [_t(1, "2026-03-01", "2026-03-05")]
        self.assertEqual(
            [t.rowid for t in purge_overlapping(train, self._test_block())], [1])

    def test_a_train_trade_still_open_into_the_window_is_purged(self):
        # entered before the window, exits inside it — its outcome is
        # determined by the same price path the test block is scored on.
        train = [_t(2, "2026-03-05", "2026-03-12")]
        self.assertEqual(purge_overlapping(train, self._test_block()), [])

    def test_a_train_trade_spanning_the_whole_window_is_purged(self):
        train = [_t(3, "2026-03-01", "2026-03-25")]
        self.assertEqual(purge_overlapping(train, self._test_block()), [])

    def test_a_trade_closing_exactly_on_the_window_open_is_purged(self):
        # Same day is still the same price path. Boundary must be inclusive.
        train = [_t(4, "2026-03-01", "2026-03-10")]
        self.assertEqual(purge_overlapping(train, self._test_block()), [])

    def test_a_trade_entering_exactly_on_the_window_close_is_purged(self):
        # The window's latest exit is 2026-03-20. Opening a position that day
        # still shares that day's price path with the test block.
        train = [_t(5, "2026-03-20", "2026-03-25")]
        self.assertEqual(purge_overlapping(train, self._test_block()), [])

    def test_a_trade_entering_the_day_after_the_window_close_is_kept(self):
        # The complement of the above: one calendar day past the window's
        # latest exit no longer shares any price path with the test block.
        train = [_t(6, "2026-03-21", "2026-03-25")]
        self.assertEqual(
            [t.rowid for t in purge_overlapping(train, self._test_block())], [6])

    def test_purging_is_idempotent(self):
        train = [_t(1, "2026-03-01", "2026-03-05"), _t(2, "2026-03-05", "2026-03-12")]
        once = purge_overlapping(train, self._test_block())
        twice = purge_overlapping(once, self._test_block())
        self.assertEqual([t.rowid for t in once], [t.rowid for t in twice])

    def test_an_empty_test_block_purges_nothing(self):
        train = [_t(1, "2026-03-01", "2026-03-05")]
        self.assertEqual(len(purge_overlapping(train, [])), 1)

    def test_load_trades_carries_exit_date(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=60)
            for t in load_trades(db, strategy="Long Call"):
                self.assertTrue(t.exit_date, "exit_date must be populated")

    def test_the_fixture_can_produce_overlapping_holds(self):
        # Guards the fixture change below: with hold_days=0 nothing can ever
        # be purged, and every purge assertion in this file would be vacuous.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=40, hold_days=10)
            ts = load_trades(db, strategy="Long Call")
            spans = {(_as_date(t.exit_date) - _as_date(t.entry_date)).days
                     for t in ts}
            self.assertEqual(spans, {10})

    def test_hold_days_zero_purges_nothing_through_the_real_pipeline(self):
        # The default fixture (hold_days=0, one zero-day-hold trade per
        # calendar day) run through the real load_trades -> purge_overlapping
        # path must be inert, pinning the claim that hold_days=0 leaves
        # existing behaviour byte-identical. The guard test above only shows
        # the fixture CAN overlap at hold_days=10; nothing else exercises
        # purge_overlapping against real loaded rows at the default.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=60, hold_days=0)
            trades = load_trades(db, strategy="Long Call")
            train, test = trades[:40], trades[40:]
            purged = purge_overlapping(train, test)
            self.assertEqual([t.rowid for t in purged], [t.rowid for t in train])


class TestPurgeFloorAndRefusal(unittest.TestCase):
    def test_a_fold_below_the_training_floor_is_dropped_and_counted(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=200, seed=3)
            # Push fold 5's last 7 training trades (i=103..109) into its own
            # test window (i=110..119) so ONLY fold 5 drops below
            # MIN_TRAIN_AFTER_PURGE (60 - 7 = 53 < 54); every other fold's
            # test window starts on a different day so this patch leaves
            # them untouched (see _force_purge's docstring for why a single
            # global hold_days cannot produce this mixed outcome). Without
            # a genuine drop, "kept + dropped == attempted" is trivially
            # true at dropped=0 and this test cannot fail no matter what
            # n_folds is set to — which is exactly how it missed Important
            # 1's n_folds=0 hardcoding.
            _force_purge(db, lo_i=103, hi_i=109, new_exit_i=110)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=60, test_size=10, step=10)
            self.assertEqual(r["n_folds_attempted"], 14)
            self.assertEqual(r["n_folds_dropped"], 1,
                             "exactly fold 5 should drop below the training floor")
            self.assertEqual(r["n_folds"], 13)
            self.assertEqual(
                r["n_folds"] + r["n_folds_dropped"], r["n_folds_attempted"],
                "every attempted fold must be either kept or counted as dropped")

    def test_surviving_folds_all_clear_the_floor(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=300, seed=5, hold_days=10)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=120, test_size=10, step=10)
            for f in r["folds"]:
                self.assertGreaterEqual(f["n_train"], MIN_TRAIN_AFTER_PURGE)

    def test_n_train_reports_the_purged_count_not_the_requested_one(self):
        # hold_days=10 with one trade per day means roughly the last 10 train
        # trades are still open when the test window opens. Without a non-zero
        # hold this assertion could never fail.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=300, seed=5, hold_days=10)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=120, test_size=10, step=10)
            self.assertTrue(r["folds"], "expected surviving folds to assert on")
            self.assertLess(r["folds"][0]["n_train"], 120,
                            "purging must actually remove trades")
            self.assertGreater(r["folds"][0]["n_train_purged"], 0)

    def test_a_zero_hold_fixture_purges_nothing(self):
        # The complement: proves the purge is driven by the interval, not by
        # position in the fold.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=300, seed=5, hold_days=0)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=120, test_size=10, step=10)
            for f in r["folds"]:
                self.assertEqual(f["n_train_purged"], 0)

    def test_too_few_surviving_folds_refuses(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=70, seed=9)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=44, test_size=10, step=10)
            self.assertTrue(r["refused"])
            self.assertIsNone(r["pooled_ic"],
                              "a refused run must not report 0.0 as an IC")
            self.assertIsNone(r["fold_ic_mean"])
            self.assertIn("fold", r["refused_reason"].lower())

    def test_no_folds_formed_at_all_does_not_blame_purging(self):
        # 50 trades < train_size(44) + test_size(10) = 54, so build_folds
        # never even forms a fold — purging has nothing to act on and
        # "widen train_size" would make the shortfall worse, not better.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=50, seed=11)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=44, test_size=10, step=10)
            self.assertTrue(r["refused"])
            self.assertEqual(r["n_folds_attempted"], 0)
            reason = r["refused_reason"].lower()
            self.assertNotIn("after purging", reason)
            self.assertNotIn("widen", reason)

    def test_folds_formed_but_all_dropped_by_the_floor_blames_purging(self):
        # 70 trades forms folds at train=44/test=10/step=10, but every fold's
        # 44 training trades sits below MIN_TRAIN_AFTER_PURGE (54) even
        # before any purge — this is the branch where the purge-floor
        # message and its "widen train_size" advice are the correct ones.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=70, seed=9)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=44, test_size=10, step=10)
            self.assertTrue(r["refused"])
            self.assertGreater(r["n_folds_attempted"], 0)
            self.assertIn("after purging", r["refused_reason"].lower())

    def test_refusal_still_reports_the_folds_it_measured(self):
        # Important-1 regression: a refusal used to hardcode n_folds=0 and
        # n_trials=0 even when some folds WERE measured (fit and scored)
        # before the run refused for having fewer surviving folds than
        # MIN_FOLDS. That made the JSON disagree with its own
        # refused_reason ("only 1 of 4 folds kept..." next to a "n_folds":
        # 0" field) and silently dropped the 200 trials genuinely spent
        # fitting the one measured fold from n_trials.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=104, seed=13)
            # Push folds 0, 1, 2's last 7 training trades into their own
            # test windows so each drops below the 54-trade floor; fold 3
            # (i=30..89 train, i=90..99 test) is left untouched and
            # survives, giving exactly the "1 of 4" split the refused
            # message names.
            _force_purge(db, lo_i=53, hi_i=59, new_exit_i=60)   # drops fold 0
            _force_purge(db, lo_i=63, hi_i=69, new_exit_i=70)   # drops fold 1
            _force_purge(db, lo_i=73, hi_i=79, new_exit_i=80)   # drops fold 2
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=60, test_size=10, step=10)

            self.assertTrue(r["refused"])
            self.assertEqual(r["n_folds_attempted"], 4)
            self.assertEqual(r["n_folds_dropped"], 3)
            self.assertEqual(
                r["n_folds"], 1,
                "the one fold that cleared the purge floor must be "
                "counted, not hardcoded to 0")
            self.assertEqual(
                r["n_folds"] + r["n_folds_dropped"], r["n_folds_attempted"],
                "measured + dropped must reconcile to attempted even on a refusal")
            self.assertEqual(
                r["n_trials"], TRIALS_PER_FOLD * 1,
                "trials genuinely spent fitting the measured fold must be reported")
            self.assertIn("1 of 4 folds", r["refused_reason"])
            # Refusal still means no statistic is reported, regardless of
            # how many folds were measured along the way.
            self.assertIsNone(r["pooled_ic"])
            self.assertIsNone(r["fold_ic_mean"])
            self.assertIsNone(r["folds_ic_positive"])

    def test_refused_markdown_does_not_claim_zero_kept(self):
        # Same fixture as above, checked through _format_markdown rather
        # than the raw dict: the header line is what an operator actually
        # reads, and "0 kept of 4 attempted" next to a reason saying "1 of 4
        # folds kept" is the two-numbers-one-label defect this branch exists
        # to close.
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=104, seed=13)
            _force_purge(db, lo_i=53, hi_i=59, new_exit_i=60)
            _force_purge(db, lo_i=63, hi_i=69, new_exit_i=70)
            _force_purge(db, lo_i=73, hi_i=79, new_exit_i=80)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=60, test_size=10, step=10)
            md = _format_markdown(r)
            self.assertIn(
                "1 measured of 4 attempted (3 dropped below the training floor)",
                md)
            self.assertNotIn("0 kept", md)

    def test_a_refusal_writes_its_artifacts_to_disk(self):
        # src/maintenance.py is the only production caller of
        # run_walk_forward and it always passes output_dir="reports" —
        # refusal-with-output_dir is the path every scheduled run takes when
        # the book is too thin, not a hypothetical. Read the file back off
        # disk rather than trusting the in-memory dict: the dict being right
        # is not evidence the write happened correctly.
        import glob
        import json
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            out_dir = os.path.join(tmp, "reports")
            _seed_db(db, n_trades=70, seed=9)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=44, test_size=10, step=10,
                                 output_dir=out_dir)
            self.assertTrue(r["refused"])

            json_files = glob.glob(os.path.join(out_dir, "walk_forward_*.json"))
            md_files = glob.glob(os.path.join(out_dir, "walk_forward_*.md"))
            self.assertEqual(len(json_files), 1,
                              f"expected one report json in {out_dir}")
            self.assertEqual(len(md_files), 1,
                              f"expected one report md in {out_dir}")

            with open(json_files[0]) as fh:
                on_disk = json.load(fh)
            self.assertIs(on_disk["refused"], True)
            self.assertIsNone(on_disk["pooled_ic"])

    def test_a_successful_run_is_not_marked_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=400, ic_target=0.15, seed=7)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=120, test_size=20, step=20)
            self.assertFalse(r["refused"])
            self.assertIsNotNone(r["pooled_ic"])


class SummaryReportsNoSharpeBarTest(unittest.TestCase):
    """Walk-forward measures IC, not Sharpe.

    A `search_bar_sharpe` sat in both summary shapes with nothing to compare
    against: this module produces no Sharpe anywhere. A bar with no counterpart
    cannot inform a decision, only suggest a rigour that was never applied.

    Asserted on real output from both paths rather than on a constructed dict —
    the two shapes are built in different places and drifted apart before.
    """

    def test_the_refusal_shape_carries_no_sharpe_bar(self):
        from src.walk_forward import _refused_summary
        refused = _refused_summary(
            "unused.db", "Long Call", n_total=5,
            train_size=100, test_size=10, step=10,
            n_attempted=0, n_dropped=0, n_measured=0,
            reason="too few trades")
        self.assertNotIn("search_bar_sharpe", refused)
        self.assertIn("n_trials", refused)
        self.assertTrue(refused["refused"])

    def test_the_success_shape_carries_no_sharpe_bar(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=400, ic_target=0.15, seed=7)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=120, test_size=20, step=20)
            self.assertFalse(r["refused"])
            self.assertNotIn("search_bar_sharpe", r)
            self.assertIn("n_trials", r)

    def test_no_summary_key_mentions_sharpe_at_all(self):
        """This module has no Sharpe to report, so none should be named."""
        with tempfile.TemporaryDirectory() as tmp:
            db = os.path.join(tmp, "trades.db")
            _seed_db(db, n_trades=400, ic_target=0.15, seed=7)
            r = run_walk_forward(db_path=db, strategy="Long Call",
                                 train_size=120, test_size=20, step=20)
            self.assertEqual([k for k in r if "sharpe" in k.lower()], [])


if __name__ == "__main__":
    unittest.main()
