"""The live VALIDATED sleeve must be the backtested rule, period by period.

`scripts/backtest_vol_managed_outlook.py::_simulate_arms` is the estimator the
VALIDATED verdict rests on. These tests run it on a synthetic market and
replay every rebalance with `src.outlook.vm_rotation.target_weights`; the
period returns (costs, cash, vol scale and all) must match to 1e-12. If anyone
changes either side without the other, this fails.
"""
from __future__ import annotations

import unittest
from typing import Dict, List

import numpy as np

from scripts.backtest_vol_managed_outlook import STEP, TURNOVER_COST, _simulate_arms
from src.outlook.backtest import BENCH, DEFAULT_UNIVERSE
from src.outlook.vm_rotation import (
    align_closes,
    basket_rv,
    latest_complete_index,
    live_target,
    target_weights,
)


def _market(n: int = 900, seed: int = 7, late: Dict[str, int] | None = None) -> tuple:
    """SPY + the 16 ETFs as random walks with distinct drifts/vols; some list late."""
    rng = np.random.default_rng(seed)
    dates = [f"D{i:05d}" for i in range(n)]
    spy_r = rng.normal(0.0004, 0.011, n)
    cols: Dict[str, np.ndarray] = {BENCH: 100.0 * np.cumprod(1.0 + spy_r)}
    late = late or {}
    for k, tk in enumerate(DEFAULT_UNIVERSE):
        beta = 0.4 + 0.08 * k
        idio = rng.normal(0.0002 * ((k % 5) - 2), 0.006 + 0.0015 * k, n)
        px = 50.0 * np.cumprod(1.0 + beta * spy_r + idio)
        start = late.get(tk, 0)
        if start:
            px[:start] = np.nan
        cols[tk] = px
    return dates, cols


def _replay_vm(dates: List[str], cols: Dict[str, np.ndarray], *, scaled: bool) -> List[float]:
    """Period returns of the live rule, using the estimator's own cost/compounding."""
    n = len(dates)
    rebal: List[int] = []
    for t in range(252, n - STEP - 1, STEP):
        avail = [tk for tk in DEFAULT_UNIVERSE if tk in cols and np.all(np.isfinite(cols[tk][t - 252: t + 1]))]
        if len(avail) >= 6:
            rebal.append(t)
    prev: Dict[str, float] = {}
    out: List[float] = []
    for i, t in enumerate(rebal):
        exec_in = t + 1
        exec_out = (rebal[i + 1] + 1) if i + 1 < len(rebal) else n - 1
        tgt = target_weights(dates, cols, t)
        assert tgt is not None
        w = tgt.weights if scaled else tgt.unscaled
        to = sum(abs(w.get(k, 0.0) - prev.get(k, 0.0)) for k in set(w) | set(prev))
        eq = 1.0 - to * TURNOVER_COST
        for d in range(exec_in + 1, exec_out + 1):
            eq *= 1.0 + sum(wt * (cols[tk][d] / cols[tk][d - 1] - 1.0) for tk, wt in w.items())
        out.append(eq - 1.0)
        prev = w
    return out


class BacktestParityTests(unittest.TestCase):
    def _assert_parity(self, dates: List[str], cols: Dict[str, np.ndarray]) -> None:
        sim = _simulate_arms(dates, cols, DEFAULT_UNIVERSE)
        got_b = _replay_vm(dates, cols, scaled=True)
        got_a = _replay_vm(dates, cols, scaled=False)
        exp_b = sim["period_rets"]["VM_OUTLOOK_TOP3"].tolist()
        exp_a = sim["period_rets"]["OUTLOOK_TOP3"].tolist()
        self.assertGreater(len(exp_b), 20)
        self.assertEqual(len(got_b), len(exp_b))
        np.testing.assert_allclose(got_b, exp_b, rtol=0, atol=1e-12)
        np.testing.assert_allclose(got_a, exp_a, rtol=0, atol=1e-12)

    def test_full_history_matches_estimator(self) -> None:
        dates, cols = _market()
        self._assert_parity(dates, cols)

    def test_late_listings_match_estimator(self) -> None:
        # XLC/XLRE list years after the others, as in the real 2006+ window.
        dates, cols = _market(seed=11, late={"XLC": 400, "XLRE": 300})
        self._assert_parity(dates, cols)

    def test_several_seeds(self) -> None:
        for seed in (1, 2, 3):
            dates, cols = _market(n=700, seed=seed)
            self._assert_parity(dates, cols)


class RuleShapeTests(unittest.TestCase):
    def test_scale_uses_basket_vol_not_spy_vol(self) -> None:
        dates, cols = _market()
        t = len(dates) - 1
        tgt = target_weights(dates, cols, t)
        assert tgt is not None
        if not tgt.unscaled:
            self.skipTest("no positive names on this path")
        rv = basket_rv(cols, tgt.unscaled, t)
        assert rv is not None
        self.assertAlmostEqual(tgt.scale, float(np.clip(0.14 / rv, 0.25, 1.0)), places=12)
        spy_rv = basket_rv(cols, {BENCH: 1.0}, t)
        self.assertNotAlmostEqual(rv, spy_rv or 0.0, places=6)

    def test_only_positive_composites_are_held(self) -> None:
        dates, cols = _market(seed=5)
        for t in range(300, len(dates), 37):
            tgt = target_weights(dates, cols, t)
            assert tgt is not None
            held = set(tgt.weights)
            by = {str(r["ticker"]): float(r["score"]) for r in tgt.ranked}
            self.assertTrue(all(by[tk] > 0 for tk in held))
            self.assertLessEqual(len(held), 3)
            for w in tgt.unscaled.values():
                self.assertAlmostEqual(w, 1.0 / 3.0)
            self.assertAlmostEqual(tgt.cash, 1.0 - sum(tgt.weights.values()))

    def test_live_target_ranks_on_latest_fully_covered_session(self) -> None:
        dates, cols = _market(n=400)
        ohlcv = {}
        for tk, arr in cols.items():
            ds = list(dates)
            cs = arr.copy()
            if tk == "XLK":  # straggler: last bar not yet archived
                ds, cs = ds[:-1], cs[:-1]
            ohlcv[tk] = (ds, cs, np.zeros(len(cs)))
        _, aligned = align_closes(ohlcv, DEFAULT_UNIVERSE)
        self.assertEqual(latest_complete_index(aligned, DEFAULT_UNIVERSE), len(dates) - 2)
        tgt = live_target(ohlcv)
        assert tgt is not None
        self.assertEqual(tgt.as_of, dates[-2])
        self.assertIn("XLK", tgt.eligible)


if __name__ == "__main__":
    unittest.main()
