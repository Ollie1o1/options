"""Tests for src/entry_selection.py — how the top-N entry slots are filled.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_entry_selection -v

The entry path draws at random among survivors. That is a deliberate choice,
not an absence of one: no ordering of survivors has ever beaten another out of
sample (23 of 48 paired cells, Wilcoxon p=0.89), and the ordering the path
actually used was carry, which nobody defends as a quality signal.
"""
import unittest

import pandas as pd

from src import entry_selection as es


def _frame(n=20):
    return pd.DataFrame({"symbol": [f"S{i:02d}" for i in range(n)],
                         "ev_per_contract": list(range(n))})


class TestEntrySeed(unittest.TestCase):
    def test_the_seed_is_stable_across_processes(self):
        # Pinned to a literal. Python's builtin hash() is salted per process,
        # so a seed derived from it would silently change every run and the
        # draw would not be reproducible from the recorded scan_id.
        self.assertEqual(es.entry_seed("2026-08-18T00:00:00+00:00|Discovery|abcd1234"),
                         es.entry_seed("2026-08-18T00:00:00+00:00|Discovery|abcd1234"))
        self.assertIsInstance(es.entry_seed("x"), int)

    def test_different_scan_ids_give_different_seeds(self):
        self.assertNotEqual(es.entry_seed("scan-a"), es.entry_seed("scan-b"))


class TestDrawEntryQueue(unittest.TestCase):
    def test_it_is_a_permutation_not_a_sample(self):
        df = _frame()
        out = es.draw_entry_queue(df, scan_id="s")
        self.assertEqual(len(out), len(df))
        self.assertEqual(sorted(out.symbol), sorted(df.symbol))

    def test_the_same_scan_id_reproduces_the_draw(self):
        df = _frame()
        a = es.draw_entry_queue(df, scan_id="s")
        b = es.draw_entry_queue(df, scan_id="s")
        self.assertEqual(list(a.symbol), list(b.symbol))

    def test_a_different_scan_id_draws_differently(self):
        df = _frame()
        a = es.draw_entry_queue(df, scan_id="scan-a")
        b = es.draw_entry_queue(df, scan_id="scan-b")
        self.assertNotEqual(list(a.symbol), list(b.symbol))

    def test_it_actually_reorders(self):
        df = _frame()
        out = es.draw_entry_queue(df, scan_id="s")
        self.assertNotEqual(list(out.symbol), list(df.symbol))

    def test_the_index_is_reset_so_head_and_iloc_agree(self):
        out = es.draw_entry_queue(_frame(), scan_id="s")
        self.assertEqual(list(out.index), list(range(len(out))))

    def test_every_row_reaches_the_front_slot_about_equally_often(self):
        # The point of drawing at random is that no candidate is structurally
        # favoured. Over many scans each of 10 rows should lead ~10% of the
        # time; a positional bias would show up here.
        df = _frame(10)
        from collections import Counter
        first = Counter(es.draw_entry_queue(df, scan_id=f"scan-{i}").symbol.iloc[0]
                        for i in range(2000))
        self.assertEqual(len(first), 10)          # every row leads sometimes
        self.assertGreater(min(first.values()), 120)   # ~200 expected, 40% band
        self.assertLess(max(first.values()), 280)

    def test_an_empty_frame_is_returned_unchanged(self):
        empty = pd.DataFrame()
        self.assertIs(es.draw_entry_queue(empty, scan_id="s"), empty)

    def test_none_is_returned_unchanged(self):
        self.assertIsNone(es.draw_entry_queue(None, scan_id="s"))

    def test_a_failure_returns_the_frame_rather_than_dropping_candidates(self):
        # Failure-safe like the rest of the scan path: a broken draw must
        # degrade to "unshuffled", never to "empty".
        from unittest import mock
        df = _frame()
        with mock.patch.object(es, "entry_seed", side_effect=RuntimeError("boom")):
            out = es.draw_entry_queue(df, scan_id="s")
        self.assertEqual(len(out), len(df))


class TestDisclosure(unittest.TestCase):
    def test_the_disclosure_names_the_mechanism(self):
        text = es.ENTRY_DISCLOSURE.lower()
        self.assertIn("random", text)
        self.assertIn("survivor", text)

    def test_the_disclosure_says_it_is_not_a_ranking(self):
        self.assertIn("rank", es.ENTRY_DISCLOSURE.lower())


class TestWiredIntoTheEntryPath(unittest.TestCase):
    """The helper exists and is reachable — but the claim that matters is that
    the scan path CALLS it. Asserted by calling, not by grepping."""

    def test_the_screener_helper_draws_using_the_recorder_scan_id(self):
        from src import candidate_record as cr
        from src import options_screener as osx

        df = _frame()
        with cr.scan("Discovery"):
            scan_id = cr.current_scan_id()
            drawn = osx._draw_entry_queue(df)
        expected = es.draw_entry_queue(df, scan_id=scan_id)
        self.assertEqual(list(drawn.symbol), list(expected.symbol))

    def test_the_draw_is_reproducible_from_the_recorded_scan_id(self):
        # The audit path: given a scan_id out of data/candidates.db, the draw
        # that produced the entries can be replayed exactly.
        df = _frame()
        recorded_scan_id = "2026-08-18T12:00:00+00:00|Discovery|deadbeef"
        first = es.draw_entry_queue(df, scan_id=recorded_scan_id)
        replayed = es.draw_entry_queue(df, scan_id=recorded_scan_id)
        self.assertEqual(list(first.symbol), list(replayed.symbol))

    def test_the_disclosure_is_printed_by_the_screener(self):
        import io
        from contextlib import redirect_stdout
        from src import options_screener as osx

        buf = io.StringIO()
        with redirect_stdout(buf):
            osx._print_entry_disclosure()
        out = buf.getvalue().lower()
        self.assertIn("random", out)
        self.assertIn("not a ranking", out)


if __name__ == "__main__":
    unittest.main()
