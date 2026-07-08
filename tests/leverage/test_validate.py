import unittest
import pandas as pd
from src.leverage.candidates import Trade
from src.leverage.validate import evaluate, render_report, CandidateReport


class _FakeCandidate:
    """Deterministic candidate: returns preset OOS trades scaled by cost so the
    harness's cost-stress path is exercised without real data."""
    name = "fake"

    def __init__(self, rets):
        self._rets = rets

    def walk_forward(self, frames, funding, costs, train_frac=0.6):
        cost = costs.get("BTC", 0.0)
        oos = [Trade("BTC", "long", i, i + 1, 100.0, 100.0, r - cost, r, 1, "time")
               for i, r in enumerate(self._rets)]
        return [], oos


def _frames():
    idx = pd.date_range("2023-01-01", periods=5, freq="D", tz="UTC")
    return {"BTC": pd.DataFrame({"close": range(5)}, index=idx)}


class TestValidate(unittest.TestCase):
    def test_promote_when_profitable_and_enough_trades(self):
        rets = [0.05, -0.02, 0.06, -0.01] * 6   # 24 trades, positive, PF>1.2
        rep = evaluate(_FakeCandidate(rets), _frames(), {"BTC": None},
                       {"BTC": 0.001}, min_n=20)
        self.assertEqual(rep.verdict, "PROMOTE")
        self.assertGreater(rep.profit_factor, 1.2)

    def test_underpowered_when_too_few_trades(self):
        rep = evaluate(_FakeCandidate([0.05, 0.06]), _frames(), {"BTC": None},
                       {"BTC": 0.0}, min_n=20)
        self.assertEqual(rep.verdict, "UNDERPOWERED")

    def test_dead_when_cost_stress_kills_it(self):
        # thin edge: dies once cost is scaled 1.5x
        rets = [0.011, -0.01] * 15   # 30 trades, barely positive at low cost
        rep = evaluate(_FakeCandidate(rets), _frames(), {"BTC": None},
                       {"BTC": 0.01}, stress=1.5, min_n=20)
        self.assertEqual(rep.verdict, "DEAD")

    def test_render_contains_names_and_verdicts(self):
        rep = evaluate(_FakeCandidate([0.05, -0.02, 0.06, -0.01] * 6), _frames(),
                       {"BTC": None}, {"BTC": 0.001}, min_n=20)
        txt = render_report([rep])
        self.assertIn("fake", txt)
        self.assertIn(rep.verdict, txt)


if __name__ == "__main__":
    unittest.main()
