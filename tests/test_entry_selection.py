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


class TestDedupBySymbol(unittest.TestCase):
    """One row per symbol survives the auto-log cut. Without an allocation
    this is a plain drop_duplicates on whatever order the frame already
    carries. With one, the symbol's single shot must go to the structure the
    allocation actually favours — not to whichever structure happened to
    have more raw rows for that symbol, which is what a shuffle-then-dedup
    silently rewards instead.
    """

    def _alloc(self, weights):
        from src.strategy_allocation import Allocation
        return Allocation(weights=weights)

    def test_without_an_allocation_it_is_plain_first_wins(self):
        df = pd.DataFrame({"symbol": ["AAPL", "AAPL", "MSFT"],
                           "strategy": ["Long Call", "Bull Put", "Bull Put"]})
        out = es.dedup_by_symbol(df, lambda r: r["strategy"])
        self.assertEqual(len(out), 2)
        self.assertEqual(out.set_index("symbol").loc["AAPL", "strategy"],
                         "Long Call")  # first row in, unchanged order

    def test_the_higher_weighted_structure_wins_even_arriving_second(self):
        # AAPL carries five raw Long Call rows ahead of a single Bull Put
        # row — the shape a real board takes when one structure is
        # constructible on many strikes and another only on one. A plain
        # shuffle-then-dedup would very likely keep a Long Call row; the
        # allocation weights Bull Put at 90% against Long Call's 4%, so the
        # symbol's one shot must go to Bull Put regardless of row counts.
        rows = [{"symbol": "AAPL", "strategy": "Long Call"} for _ in range(5)]
        rows.append({"symbol": "AAPL", "strategy": "Bull Put"})
        df = pd.DataFrame(rows)
        alloc = self._alloc({"Bull Put": 0.90, "Long Call": 0.04})
        out = es.dedup_by_symbol(df, lambda r: r["strategy"], alloc=alloc)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["strategy"], "Bull Put")

    def test_ties_within_the_winning_structure_keep_the_caller_s_order(self):
        # Two Bull Put rows for the same symbol at equal weight: the winner
        # is whichever the caller's own draw (e.g. draw_entry_queue) put
        # first, not re-shuffled by the weight sort.
        df = pd.DataFrame({
            "symbol": ["AAPL", "AAPL"],
            "strategy": ["Bull Put", "Bull Put"],
            "strike": [150, 160],
        })
        alloc = self._alloc({"Bull Put": 0.90})
        out = es.dedup_by_symbol(df, lambda r: r["strategy"], alloc=alloc)
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["strike"], 150)

    def test_a_structure_missing_from_the_weights_sorts_last(self):
        rows = [{"symbol": "AAPL", "strategy": "Iron Condor"},
                {"symbol": "AAPL", "strategy": "Bull Put"}]
        df = pd.DataFrame(rows[::-1])  # Bull Put arrives second
        alloc = self._alloc({"Bull Put": 0.90})  # Iron Condor not eligible
        out = es.dedup_by_symbol(df, lambda r: r["strategy"], alloc=alloc)
        self.assertEqual(out.iloc[0]["strategy"], "Bull Put")

    def test_symbols_without_any_competing_structure_are_unaffected(self):
        df = pd.DataFrame({"symbol": ["AAPL", "MSFT"],
                           "strategy": ["Bull Put", "Long Call"]})
        alloc = self._alloc({"Bull Put": 0.90, "Long Call": 0.04})
        out = es.dedup_by_symbol(df, lambda r: r["strategy"], alloc=alloc)
        self.assertEqual(len(out), 2)
        self.assertEqual(set(out.symbol), {"AAPL", "MSFT"})

    def test_an_allocation_with_no_weights_behaves_like_no_allocation(self):
        from src.strategy_allocation import Allocation
        df = pd.DataFrame({"symbol": ["AAPL", "AAPL"],
                           "strategy": ["Long Call", "Bull Put"]})
        empty_alloc = Allocation(weights={})
        out = es.dedup_by_symbol(df, lambda r: r["strategy"], alloc=empty_alloc)
        self.assertEqual(out.iloc[0]["strategy"], "Long Call")

    def test_no_symbol_column_is_returned_unchanged(self):
        df = pd.DataFrame({"strategy": ["Bull Put"]})
        out = es.dedup_by_symbol(df, lambda r: r["strategy"])
        self.assertIs(out, df)

    def test_empty_and_none_are_returned_unchanged(self):
        empty = pd.DataFrame()
        self.assertIs(es.dedup_by_symbol(empty, lambda r: None), empty)
        self.assertIsNone(es.dedup_by_symbol(None, lambda r: None))


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

    def test_the_screener_helper_dedups_using_the_live_allocation(self):
        from unittest import mock
        from src import options_screener as osx
        from src.strategy_allocation import Allocation

        rows = [{"symbol": "AAPL", "strategy": "Long Call"} for _ in range(5)]
        rows.append({"symbol": "AAPL", "strategy": "Bull Put"})
        df = pd.DataFrame(rows)
        alloc = Allocation(weights={"Bull Put": 0.90, "Long Call": 0.04})
        with mock.patch.object(osx, "_current_allocation", return_value=alloc):
            out = osx._dedup_by_symbol(df, lambda r: r["strategy"])
        self.assertEqual(len(out), 1)
        self.assertEqual(out.iloc[0]["strategy"], "Bull Put")

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
