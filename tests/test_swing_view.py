"""Tests for src/swing_view.py — the cost cap and breakeven move for a 1-3 day hold.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python scripts/run_tests.py swing_view

The view refuses on ONE thing, round-trip cost, because that is the one thing
measured to separate long options at this horizon. Most of what is asserted
here is that a missing number is never read as a good one.
"""
import contextlib
import io
import math
import unittest

import pandas as pd

from src import swing_view as sv


def _row(**over):
    """A long call quoted 4.90 x 5.10: 4% round trip, well inside the cap."""
    row = {"symbol": "AAPL", "type": "call", "strike": 200.0,
           "expiration": "2026-11-20", "dte": 38.0,
           "bid": 4.90, "ask": 5.10, "delta": 0.50, "gamma": 0.02,
           "theta": -0.10, "underlying": 200.0, "hv_30d": 0.30}
    row.update(over)
    return row


class RoundTripCost(unittest.TestCase):
    def test_is_the_spread_over_the_mid(self):
        self.assertAlmostEqual(sv.round_trip_cost_pct(4.90, 5.10), 0.04)

    def test_missing_or_one_sided_quote_is_not_answerable(self):
        self.assertIsNone(sv.round_trip_cost_pct(None, 5.10))
        self.assertIsNone(sv.round_trip_cost_pct(float("nan"), 5.10))
        self.assertIsNone(sv.round_trip_cost_pct(0.0, 5.10))
        self.assertIsNone(sv.round_trip_cost_pct(5.10, 4.90))

    def test_locked_market_costs_zero_not_none(self):
        self.assertEqual(sv.round_trip_cost_pct(5.00, 5.00), 0.0)


class BreakevenMove(unittest.TestCase):
    def test_recovers_the_spread_and_the_decay(self):
        # 3 sessions = 4.2 calendar days. Gain needed: 0.20 spread + 0.42 decay.
        need = 0.20 + 0.10 * 4.2
        move = sv.breakeven_move(4.90, 5.10, 0.50, 0.02, -0.10, 3)
        self.assertAlmostEqual(0.50 * move + 0.5 * 0.02 * move ** 2, need)

    def test_put_delta_sign_does_not_matter(self):
        self.assertAlmostEqual(sv.breakeven_move(4.90, 5.10, -0.50, 0.02, -0.10, 3),
                               sv.breakeven_move(4.90, 5.10, 0.50, 0.02, -0.10, 3))

    def test_without_gamma_falls_back_to_delta_alone(self):
        move = sv.breakeven_move(4.90, 5.10, 0.50, None, -0.10, 3)
        self.assertAlmostEqual(move, (0.20 + 0.42) / 0.50)

    def test_zero_theta_is_a_real_zero(self):
        self.assertAlmostEqual(sv.breakeven_move(4.90, 5.10, 0.50, None, 0.0, 3),
                               0.20 / 0.50)

    def test_missing_delta_or_theta_is_not_answerable(self):
        self.assertIsNone(sv.breakeven_move(4.90, 5.10, None, 0.02, -0.10, 3))
        self.assertIsNone(sv.breakeven_move(4.90, 5.10, 0.0, 0.02, -0.10, 3))
        self.assertIsNone(sv.breakeven_move(4.90, 5.10, 0.50, 0.02, None, 3))


class TypicalMove(unittest.TestCase):
    def test_is_one_sigma_over_the_horizon(self):
        self.assertAlmostEqual(sv.typical_move(200.0, 0.30, 3),
                               200.0 * 0.30 * math.sqrt(3 / 252))

    def test_missing_vol_is_not_answerable(self):
        self.assertIsNone(sv.typical_move(200.0, None, 3))
        self.assertIsNone(sv.typical_move(200.0, 0.0, 3))
        self.assertIsNone(sv.typical_move(None, 0.30, 3))


class SwingBoard(unittest.TestCase):
    def test_refuses_above_the_cap_and_keeps_at_it(self):
        df = pd.DataFrame([
            _row(symbol="AAA", bid=4.90, ask=5.10),          # 4.0%
            _row(symbol="BBB", bid=9.70, ask=10.30),         # 6.0%, at the cap
            _row(symbol="CCC", bid=4.70, ask=5.30),          # 12.0%
        ])
        res = sv.swing_board(df, max_cost_pct=0.06)
        self.assertEqual(list(res.kept["symbol"]), ["AAA", "BBB"])
        self.assertEqual((res.scanned, res.over_cap, res.unpriced), (3, 1, 0))

    def test_unquotable_row_is_counted_apart_and_never_kept(self):
        df = pd.DataFrame([_row(), _row(symbol="ZZZ", bid=float("nan"))])
        res = sv.swing_board(df)
        self.assertEqual(list(res.kept["symbol"]), ["AAPL"])
        self.assertEqual((res.over_cap, res.unpriced), (0, 1))

    def test_orders_by_cost_and_shows_one_contract_per_name_and_side(self):
        df = pd.DataFrame([
            _row(symbol="AAA", strike=200.0, bid=4.80, ask=5.00),   # 4.08%
            _row(symbol="AAA", strike=205.0, bid=4.95, ask=5.05),   # 2.0%
            _row(symbol="AAA", type="put", strike=195.0, bid=4.85, ask=5.05),
            _row(symbol="BBB", bid=4.97, ask=5.03),                 # 1.2%
        ])
        res = sv.swing_board(df)
        got = list(zip(res.kept["symbol"], res.kept["type"], res.kept["strike"]))
        self.assertEqual(got, [("BBB", "call", 200.0), ("AAA", "call", 205.0),
                               ("AAA", "put", 195.0)])

    def test_annotates_the_move_needed_against_the_typical_one(self):
        res = sv.swing_board(pd.DataFrame([_row()]))
        r = res.kept.iloc[0]
        need = sv.breakeven_move(4.90, 5.10, 0.50, 0.02, -0.10, 3)
        typical = sv.typical_move(200.0, 0.30, 3)
        self.assertAlmostEqual(r["swing_cost_pct"], 0.04)
        self.assertAlmostEqual(r["swing_need_pct"], need / 200.0)
        self.assertAlmostEqual(r["swing_typical_pct"], typical / 200.0)
        self.assertAlmostEqual(r["swing_need_vs_typical"], need / typical)

    def test_missing_vol_keeps_the_row_with_a_blank_not_a_zero(self):
        res = sv.swing_board(pd.DataFrame([_row(hv_30d=None)]))
        self.assertEqual(len(res.kept), 1)
        self.assertTrue(pd.isna(res.kept.iloc[0]["swing_need_vs_typical"]))

    def test_ignores_rows_that_are_not_a_single_call_or_put(self):
        df = pd.DataFrame([_row(), _row(symbol="SPY", type="Bull Put")])
        res = sv.swing_board(df)
        self.assertEqual(res.scanned, 1)

    def test_empty_and_none_are_empty_results(self):
        for df in (None, pd.DataFrame()):
            res = sv.swing_board(df)
            self.assertEqual((len(res.kept), res.scanned), (0, 0))

    def test_does_not_mutate_the_board_it_was_given(self):
        df = pd.DataFrame([_row()])
        before = list(df.columns)
        sv.swing_board(df)
        self.assertEqual(list(df.columns), before)


class BoardKept(unittest.TestCase):
    def _two(self):
        return pd.DataFrame([_row(symbol="AAA"), _row(symbol="BBB")])

    def test_marks_which_rows_the_board_kept(self):
        df = self._two()
        res = sv.swing_board(df, board_kept=df[df["symbol"] == "AAA"])
        got = dict(zip(res.kept["symbol"], res.kept["swing_board_kept"]))
        self.assertEqual(got, {"AAA": True, "BBB": False})

    def test_a_board_that_kept_nothing_marks_every_row_refused(self):
        res = sv.swing_board(self._two(), board_kept=pd.DataFrame())
        self.assertEqual(list(res.kept["swing_board_kept"]), [False, False])

    def test_same_name_different_strike_is_a_different_contract(self):
        df = pd.DataFrame([_row(strike=200.0), _row(strike=205.0, bid=4.95, ask=5.05)])
        res = sv.swing_board(df, board_kept=df[df["strike"] == 200.0])
        self.assertEqual(list(res.kept["strike"]), [205.0])
        self.assertFalse(res.kept.iloc[0]["swing_board_kept"])

    def test_without_a_board_frame_the_column_is_absent(self):
        self.assertNotIn("swing_board_kept", sv.swing_board(self._two()).kept.columns)

    def test_unmatchable_board_frame_leaves_the_column_absent(self):
        res = sv.swing_board(self._two(), board_kept=pd.DataFrame({"x": [1]}))
        self.assertNotIn("swing_board_kept", res.kept.columns)


class Rendering(unittest.TestCase):
    def _render(self, df, **kw):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            sv.print_swing_view(sv.swing_board(df, **kw.pop("board", {})), **kw)
        return buf.getvalue()

    def test_prints_the_row_the_cap_and_what_was_refused(self):
        out = self._render(pd.DataFrame([
            _row(), _row(symbol="CCC", bid=4.70, ask=5.30)]))
        self.assertIn("AAPL", out)
        self.assertIn("4.0%", out)
        self.assertNotIn("CCC", out)
        self.assertIn("1 of 2", out)
        self.assertIn("6%", out)

    def test_says_it_is_not_a_view_on_direction(self):
        out = self._render(pd.DataFrame([_row()]))
        self.assertIn("direction", out)

    def test_missing_ratio_renders_as_na(self):
        out = self._render(pd.DataFrame([_row(hv_30d=None)]))
        self.assertIn("n/a", out)

    def test_truncation_is_stated(self):
        df = pd.DataFrame([_row(symbol=f"S{i:02d}") for i in range(5)])
        out = self._render(df, max_rows=2)
        self.assertIn("3 more", out)

    def test_refused_rows_are_labelled_and_explained(self):
        df = pd.DataFrame([_row(symbol="AAA"), _row(symbol="BBB")])
        out = self._render(df, board={"board_kept": df[df["symbol"] == "AAA"]})
        self.assertIn("Board", out)
        self.assertRegex(out, r"AAA.*kept")
        self.assertRegex(out, r"BBB.*refused")
        self.assertIn("did not", out)

    def test_no_board_column_without_a_board_frame(self):
        self.assertNotIn("Board", self._render(pd.DataFrame([_row()])))

    def test_nothing_scanned_prints_nothing(self):
        self.assertEqual(self._render(pd.DataFrame()), "")

    def test_everything_refused_still_says_so(self):
        out = self._render(pd.DataFrame([_row(bid=4.70, ask=5.30)]))
        self.assertIn("1 of 1", out)


if __name__ == "__main__":
    unittest.main()
