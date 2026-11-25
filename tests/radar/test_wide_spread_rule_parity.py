"""The live etf_wide_spread scanner must pick the legs Test O1's estimator picked.

Builds a synthetic `odx_chain` (SPY, monthly expirations, $5 strikes, a vol
smile, some one-sided quotes), runs `scripts/backtest_odx_wide_60_120.run_side`
on it, and replays every weekly entry through
`src.radar.wide_spread_rule.choose_expiration` + `select_legs` on the same rows.
Every trade must have the same expiration and strikes, and every entry the
estimator skipped must be skipped (or explained by its end-of-sample rule).
"""
from __future__ import annotations

import datetime as dt
import math
import os
import sqlite3
import tempfile
import unittest
from typing import Dict, List, Tuple

import numpy as np

from scripts.backtest_odx_wide_60_120 import _select_weekly_entries, run_side
from src.radar.wide_spread_rule import LegQuote, choose_expiration, select_legs
from src.utils import bs_delta, bs_price


def _third_fridays(start: dt.date, months: int) -> List[dt.date]:
    out = []
    y, m = start.year, start.month
    for _ in range(months):
        d = dt.date(y, m, 15)
        while d.weekday() != 4:
            d += dt.timedelta(days=1)
        out.append(d)
        m += 1
        if m > 12:
            y, m = y + 1, 1
    return out


def _build_chain(path: str, *, n_days: int = 320, seed: int = 3) -> Tuple[List[str], Dict[str, float]]:
    rng = np.random.default_rng(seed)
    start = dt.date(2015, 1, 2)
    days: List[dt.date] = []
    d = start
    while len(days) < n_days:
        if d.weekday() < 5:
            days.append(d)
        d += dt.timedelta(days=1)
    spot = 200.0 * np.cumprod(1.0 + rng.normal(0.0003, 0.011, n_days))
    base_iv = np.clip(0.16 + np.cumsum(rng.normal(0, 0.004, n_days)), 0.10, 0.45)
    exps = _third_fridays(start, 20)
    con = sqlite3.connect(path)
    con.execute("""CREATE TABLE odx_chain (
        symbol TEXT, date TEXT, expiration TEXT, strike REAL, type TEXT,
        bid REAL, ask REAL, mid REAL, iv REAL,
        delta REAL, gamma REAL, theta REAL, vega REAL, rho REAL,
        volume REAL, bid_size REAL, ask_size REAL,
        underlying REAL, dte REAL,
        PRIMARY KEY (symbol, date, expiration, strike, type))""")
    con.execute("CREATE INDEX idx_odx_sym_date ON odx_chain(symbol, date)")
    rows = []
    for i, day in enumerate(days):
        s = float(spot[i])
        for e in exps:
            dte = (e - day).days
            if dte <= 0 or dte > 150:
                continue
            T = dte / 365.0
            lo, hi = int(s * 0.70 / 5) * 5, int(s * 1.25 / 5) * 5
            for k in range(lo, hi + 5, 5):
                mny = math.log(k / s)
                for typ in ("put", "call"):
                    iv = float(base_iv[i] * (1.0 - 0.9 * mny + 1.5 * mny * mny))
                    px = float(bs_price(typ, s, float(k), T, 0.02, iv))
                    if px < 0.03:
                        continue
                    half = max(0.01, 0.01 * px + 0.02 * float(rng.random()))
                    bid = max(0.0, px - half)
                    if rng.random() < 0.04:  # a one-sided quote now and then
                        bid = 0.0
                    ask = px + half
                    delta = float(bs_delta(typ, s, float(k), T, 0.02, iv))
                    rows.append(("SPY", day.isoformat(), e.isoformat(), float(k), typ,
                                 round(bid, 2), round(ask, 2), round(0.5 * (bid + ask), 3), iv,
                                 delta, None, None, None, None, 10.0, 1.0, 1.0, s, float(dte)))
    con.executemany("INSERT INTO odx_chain VALUES (" + ",".join("?" * 19) + ")", rows)
    con.commit()
    con.close()
    return [d.isoformat() for d in days], {d.isoformat(): float(spot[i]) for i, d in enumerate(days)}


class WideSpreadRuleParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp = tempfile.TemporaryDirectory()
        cls.db = os.path.join(cls._tmp.name, "odx.db")
        cls.dates, cls.spot = _build_chain(cls.db)

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp.cleanup()

    def _replay(self, side: str) -> None:
        con = sqlite3.connect(self.db)
        try:
            weekly = _select_weekly_entries(self.dates)
            idx = {d: i for i, d in enumerate(self.dates)}
            trades = run_side(con, self.dates, idx, self.spot, weekly, side)["O1-H_crossed"]
            opt_type = "put" if side == "bull_put" else "call"
            got: Dict[str, Tuple[str, float, float]] = {}
            for edate in weekly:
                rows = con.execute(
                    "SELECT expiration, strike, bid, ask, mid, delta, underlying, dte FROM odx_chain "
                    "WHERE symbol = 'SPY' AND date = ? AND type = ? AND dte BETWEEN 60 AND 120",
                    (edate, opt_type),
                ).fetchall()
                if not rows:
                    continue
                exp = choose_expiration({r[0]: float(r[7]) for r in rows})
                assert exp is not None
                er = [r for r in rows if r[0] == exp]
                quotes = [LegQuote(strike=float(r[1]), bid=r[2], ask=r[3], delta=r[5]) for r in er]
                legs, _ = select_legs(quotes, spot=float(er[0][6]), side=side, expiration=exp, dte=float(er[0][7]))
                if legs is not None:
                    got[edate] = (exp, legs.short.strike, legs.long.strike)
        finally:
            con.close()

        self.assertGreater(len(trades), 15)
        for t in trades:
            self.assertIn(t.entry_date, got)
            self.assertEqual(got[t.entry_date], (t.expiration, t.k_short, t.k_long), t.entry_date)
        traded = {t.entry_date for t in trades}
        for edate, (exp, _, _) in got.items():
            if edate not in traded:
                # The estimator's only extra skips: expiry beyond the sample, or no later bar.
                self.assertTrue(exp > self.dates[-1] or edate == self.dates[-1], edate)

    def test_bull_put_legs_match_estimator(self) -> None:
        self._replay("bull_put")

    def test_bear_call_legs_match_estimator(self) -> None:
        self._replay("bear_call")


class RuleShapeTests(unittest.TestCase):
    def _q(self, k: float, d: float, bid: float = 1.0, ask: float = 1.1) -> LegQuote:
        return LegQuote(strike=k, bid=bid, ask=ask, delta=d)

    def test_short_leg_is_delta_targeted_not_fixed_moneyness(self) -> None:
        # Low-vol fund: 6% OTM is ~-0.12 delta; the rule must walk up to ~-0.25.
        qs = [self._q(94.0, -0.12, 0.50, 0.55), self._q(96.0, -0.19, 0.80, 0.85),
              self._q(97.0, -0.24, 1.10, 1.15), self._q(91.0, -0.06, 0.20, 0.24)]
        legs, why = select_legs(qs, spot=100.0, side="bull_put", expiration="E", dte=90)
        self.assertEqual(why, "ok")
        assert legs is not None
        self.assertEqual(legs.short.strike, 97.0)
        self.assertEqual(legs.long.strike, 91.0)  # nearest to 97 - 5

    def test_reasons(self) -> None:
        far = [self._q(80.0, -0.05)]
        self.assertEqual(select_legs(far, spot=100.0, side="bull_put", expiration="E", dte=90)[1], "no_short_delta")
        only_short = [self._q(95.0, -0.25)]
        self.assertEqual(select_legs(only_short, spot=100.0, side="bull_put", expiration="E", dte=90)[1], "no_long_leg")
        wide = [self._q(95.0, -0.25, 1.00, 1.60), self._q(90.0, -0.10, 0.30, 0.70)]
        self.assertEqual(select_legs(wide, spot=100.0, side="bull_put", expiration="E", dte=90)[1], "friction_gt_25pct")
        inverted = [self._q(95.0, -0.25, 0.30, 0.35), self._q(90.0, -0.10, 0.40, 0.45)]
        self.assertEqual(select_legs(inverted, spot=100.0, side="bull_put", expiration="E", dte=90)[1], "non_pos_credit")

    def test_expiration_choice(self) -> None:
        self.assertEqual(choose_expiration({"a": 59, "b": 85, "c": 95}), "b")
        self.assertEqual(choose_expiration({"a": 85, "b": 95}), "a")  # tie -> earlier string
        self.assertIsNone(choose_expiration({"a": 30, "b": 150}))


if __name__ == "__main__":
    unittest.main()
