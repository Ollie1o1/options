import unittest
import numpy as np
import pandas as pd
from src.probability_lab.rnd import rnd_from_smile
from src.probability_lab.structures import (
    Structure, evaluate, enumerate_structures, rank,
)


class TestStructures(unittest.TestCase):
    def setUp(self):
        self.S, self.T, self.r, self.sig = 100.0, 0.25, 0.04, 0.30
        strikes = np.linspace(60, 160, 41)
        self.d = rnd_from_smile(strikes, np.full_like(strikes, self.sig),
                                self.T, self.S, self.r)

    def test_long_call_payoff(self):
        s = Structure("Long 100 call", [("call", 100.0, 1)], entry_cost=5.0,
                      strikes_label="100")
        self.assertAlmostEqual(s.payoff_at(110.0), 10.0)
        self.assertAlmostEqual(s.payoff_at(95.0), 0.0)

    def test_vertical_payoff_capped(self):
        s = Structure("100/105 call vert", [("call", 100.0, 1), ("call", 105.0, -1)],
                      entry_cost=2.0, strikes_label="100/105")
        self.assertAlmostEqual(s.payoff_at(120.0), 5.0)   # capped at width
        self.assertAlmostEqual(s.payoff_at(100.0), 0.0)

    def test_market_ev_invariant_near_zero(self):
        # Under the RND, a call priced at its BS value has EV ~ carry ~ 0.
        from src.utils import bs_call
        prem = float(bs_call(self.S, 100.0, self.T, self.r, self.sig))
        s = Structure("Long 100 call", [("call", 100.0, 1)], entry_cost=prem,
                      strikes_label="100")
        res = evaluate(s, self.d)
        self.assertAlmostEqual(res["ev"], 0.0, delta=100 * 0.10)  # within $10

    def test_pop_long_call(self):
        s = Structure("Long 100 call", [("call", 100.0, 1)], entry_cost=5.0,
                      strikes_label="100")
        res = evaluate(s, self.d)
        self.assertAlmostEqual(res["pop"], self.d.prob_above(105.0), delta=0.02)

    def test_enumerate_and_rank(self):
        from src.utils import bs_call, bs_put
        rows = []
        for K in range(80, 121, 5):
            c = float(bs_call(self.S, K, self.T, self.r, self.sig))
            p = float(bs_put(self.S, K, self.T, self.r, self.sig))
            rows.append({"type": "call", "strike": float(K),
                         "bid": c - 0.1, "ask": c + 0.1})
            rows.append({"type": "put", "strike": float(K),
                         "bid": p - 0.1, "ask": p + 0.1})
        chain = pd.DataFrame(rows)
        structs = enumerate_structures(chain, self.S)
        self.assertGreaterEqual(len(structs), 4)
        ranked = rank(structs, self.d, self.d)
        self.assertEqual(len(ranked), len(structs))
        self.assertIn("ev_view", ranked[0])
        self.assertIn("ev_market", ranked[0])
        evs = [row["ev_view"] for row in ranked]
        self.assertEqual(evs, sorted(evs, reverse=True))


if __name__ == "__main__":
    unittest.main()
