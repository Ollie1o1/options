"""Expected return on risk, decomposed rather than learned whole.

Predicting return directly FAILED its guard on this book: walk-forward slope
0.368, 95% CI [-0.252, 0.987]. Option returns are heavy-tailed and 609
out-of-sample points cannot pin a conditional mean from five features.

This decomposes instead:

    E[return on risk] = p * W_s + (1 - p) * L_s

`p` is the calibrated probability, which is already validated out-of-sample.
`W_s` and `L_s` are the mean win and mean loss magnitudes of structure `s`,
which are properties of its payoff geometry and are measured, not learned. That
imposes what is known instead of asking five features to rediscover it.

The magnitudes are why a strategy allowlist by NAME is the wrong instrument.
Measured 2026-08-24 on return on capital at risk:

    Bear Call   wins 59.3% at +30.20%, loses at -55.54%  =>  -4.73%
    Long Call   wins 38.6% at +82.79%, loses at -52.42%  =>  -0.18%
    Bull Put    wins 65.0% at +52.13%, loses at -48.72%  =>  +16.80%

A high win rate is not an edge and a low one is not a disqualification. Only
the product is, and the product is a per-CONTRACT quantity.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest

import numpy as np
import pandas as pd

from src import expected_return as er
from src import pop_calibration as pc


def _frame(n=1200, seed=4):
    """Two structures with deliberately opposite geometry.

    `wide` wins rarely but hugely; `tight` wins often but small. Their mean
    returns are close, so anything that ranks by win rate alone gets this
    backwards.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        wide = i % 2 == 0
        d = float(rng.uniform(0.05, 0.60))
        p = (1.0 - d) if not wide else d * 0.6
        won = rng.random() < p
        if wide:
            ret = 1.60 if won else -0.50
        else:
            ret = 0.10 if won else -0.30
        rows.append({
            "abs_delta": d, "dte": 30.0, "entry_iv": 0.30,
            "iv_rank_score": 0.5, "credit_to_width": 0.0 if wide else 0.25,
            "strategy": "Long Call" if wide else "Bull Put",
            "is_short": 0 if wide else 1,
            "entry_date": (pd.to_datetime("2026-01-01")
                           + pd.Timedelta(days=i // 8)).strftime("%Y-%m-%d"),
            "won": int(won), "ret_on_risk": ret,
        })
    return pd.DataFrame(rows)


class TestMagnitudes(unittest.TestCase):

    def test_win_and_loss_magnitudes_are_recovered_per_structure(self):
        mags = er.magnitudes(_frame())
        self.assertAlmostEqual(mags["Long Call"].win, 1.60, places=6)
        self.assertAlmostEqual(mags["Long Call"].loss, -0.50, places=6)
        self.assertAlmostEqual(mags["Bull Put"].win, 0.10, places=6)
        self.assertAlmostEqual(mags["Bull Put"].loss, -0.30, places=6)

    def test_a_structure_with_no_losses_yet_is_not_given_a_zero_loss(self):
        """A structure that has only ever won has an UNKNOWN loss, not a
        costless one. Zero would make its expected return unbeatable."""
        df = _frame()
        df = df[(df["strategy"] != "Bull Put") | (df["ret_on_risk"] > 0)]
        mags = er.magnitudes(df)
        self.assertLess(mags["Bull Put"].loss, 0.0)
        self.assertFalse(mags["Bull Put"].own_losses)

    def test_an_unseen_structure_falls_back_to_the_pooled_geometry(self):
        mags = er.magnitudes(_frame())
        got = er.geometry_for(mags, "Jade Lizard")
        self.assertGreater(got.win, 0.0)
        self.assertLess(got.loss, 0.0)


class TestExpectedReturn(unittest.TestCase):

    def test_it_is_the_probability_weighted_payoff(self):
        mags = er.magnitudes(_frame())
        model = pc.fit(_frame())
        row = {"abs_delta": 0.3, "dte": 30, "entry_iv": 0.3,
               "iv_rank_score": 0.5, "credit_to_width": 0.25,
               "strategy": "Bull Put"}
        p = pc.probability_for(row, model)
        assert p is not None
        got = er.expected_return_for(row, model, mags)
        assert got is not None
        g = er.geometry_for(mags, "Bull Put")
        self.assertAlmostEqual(got, p * g.win + (1 - p) * g.loss, places=9)

    def test_no_model_gives_no_number(self):
        self.assertIsNone(er.expected_return_for({"abs_delta": 0.3}, None, {}))


class TestWalkForward(unittest.TestCase):

    def test_it_never_trains_on_its_own_day(self):
        oos = er.walk_forward(_frame(), seed_n=400, step=50)
        self.assertGreater(len(oos), 0)
        self.assertTrue((oos["trained_through"] < oos["entry_date"]).all())

    def test_a_real_edge_survives_out_of_sample(self):
        oos = er.walk_forward(_frame(), seed_n=400, step=50)
        rel = pc.return_reliability(oos, n_buckets=5)
        ok, reason = pc.ship_check_return(rel)
        self.assertTrue(ok, reason)

    def test_the_two_structures_interleave_rather_than_stack(self):
        """The whole point. If ranking were really by structure name, every
        row of one would sit above every row of the other. Instead the two
        distributions must overlap, so a good contract of the weaker structure
        outranks a poor contract of the stronger one."""
        oos = er.walk_forward(_frame(), seed_n=400, step=50)
        a = oos[oos["strategy"] == "Long Call"]["predicted"]
        b = oos[oos["strategy"] == "Bull Put"]["predicted"]
        self.assertGreater(len(a), 0)
        self.assertGreater(len(b), 0)
        self.assertGreater(b.max(), a.min(),
                           "every Bull Put ranks below every Long Call — "
                           "this is an allowlist wearing a number")
        self.assertGreater(a.max(), b.min())


class TestChoose(unittest.TestCase):

    def test_the_best_expected_return_is_selected_whatever_its_name(self):
        board = pd.DataFrame([
            {"strategy": "Bull Put", "expected_return": 0.02},
            {"strategy": "Long Call", "expected_return": 0.11},
            {"strategy": "Iron Condor", "expected_return": -0.04},
        ])
        self.assertEqual(er.choose(board)["strategy"], "Long Call")

    def test_an_all_negative_board_selects_nothing(self):
        """Refuse, do not rank. The best of a bad set is still bad."""
        board = pd.DataFrame([
            {"strategy": "Bull Put", "expected_return": -0.02},
            {"strategy": "Long Call", "expected_return": -0.11},
        ])
        self.assertIsNone(er.choose(board))

    def test_an_empty_board_selects_nothing(self):
        self.assertIsNone(er.choose(pd.DataFrame()))


class TestOnTheRealLedger(unittest.TestCase):

    def test_the_loader_and_the_magnitudes_agree_with_the_ledger(self):
        """`p*W + (1-p)*L` must reproduce each structure's realised mean —
        it is an identity, and a mismatch means a denominator slipped."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "l.db")
            conn = sqlite3.connect(path)
            conn.execute("""CREATE TABLE trades (
                entry_id INTEGER PRIMARY KEY, date TEXT, expiration TEXT,
                strategy_name TEXT, status TEXT, pnl_usd REAL,
                entry_delta REAL, entry_iv REAL, iv_rank_score REAL,
                net_credit REAL, spread_width REAL, capital_at_risk REAL)""")
            for i, pnl in enumerate([120.0, -80.0, 200.0, -50.0, 90.0]):
                conn.execute("INSERT INTO trades VALUES "
                             "(?,?,?,?,?,?,?,?,?,?,?,?)",
                             (i + 1, f"2026-05-{i+1:02d}", "2026-06-30",
                              "Bull Put", "CLOSED", pnl, -0.3, 0.3, 0.5,
                              1.2, 5.0, 400.0))
            conn.commit()
            conn.close()
            df = pc.load_training_set(path)

        mags = er.magnitudes(df)
        g = mags["Bull Put"]
        r = df["ret_on_risk"]
        p = float((r > 0).mean())
        self.assertAlmostEqual(p * g.win + (1 - p) * g.loss,
                               float(r.mean()), places=9)


if __name__ == "__main__":
    unittest.main()
