"""The short-premium gate — §2.4 of the signed gate spec.

The properties worth defending are the ones that stop this gate flattering the
strategy it measures: the bootstrap must respect entry-day clustering, trades
whose friction exceeds their credit must not count as evidence, and READY must
not fire when the typical trade and the book disagree.
"""
import os
import sqlite3
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.short_premium_gate import (  # noqa: E402
    LEG_COUNTS,
    TRADEABILITY_MAX_FRICTION_RATIO,
    cluster_bootstrap_medians,
    cohort_caveats,
    decide_arm_a,
    friction_to_credit,
    net_return_on_risk,
    posterior_median_above,
)


class _Model:
    """Minimal cost model: a fixed friction per share, regardless of structure."""

    def __init__(self, per_share):
        self.per_share = per_share

    def friction(self, strategy, n_legs, credit, round_trip=True):
        return self.per_share * n_legs * (2 if round_trip else 1)


def _row(pnl=50.0, car=100.0, credit=0.50, strategy="Bull Put", date="2026-06-01"):
    return {"strategy_name": strategy, "quality_score": 70.0, "pnl_usd": pnl,
            "capital_at_risk": car, "date": date, "net_credit": credit,
            "entry_price": credit, "paper_only": 1}


class NetReturnTest(unittest.TestCase):
    def test_recosting_moves_pnl_by_the_friction_difference(self):
        old, new = _Model(0.05), _Model(0.15)
        # 2 legs, round trip: (0.15-0.05) * 2 * 2 = $0.40/share = $40/contract
        v = net_return_on_risk(_row(pnl=50.0, car=100.0), old, new)
        self.assertAlmostEqual(v, (50.0 - 40.0) / 100.0)

    def test_without_models_the_stored_figure_is_used(self):
        self.assertAlmostEqual(net_return_on_risk(_row(pnl=25.0, car=100.0)), 0.25)

    def test_a_zero_denominator_is_not_a_return(self):
        self.assertIsNone(net_return_on_risk(_row(car=0.0)))


class TradeabilityTest(unittest.TestCase):
    def test_friction_over_credit_is_detected(self):
        # credit $0.24/share; friction 2 legs x 2 sides x $0.162 = $0.648/share
        ratio = friction_to_credit(_row(credit=0.24), _Model(0.162))
        self.assertGreater(ratio, 1.0)

    def test_a_healthy_spread_is_well_under_the_bar(self):
        ratio = friction_to_credit(_row(credit=2.00), _Model(0.05))
        self.assertLess(ratio, TRADEABILITY_MAX_FRICTION_RATIO)

    def test_a_single_leg_short_costs_half_a_verticals_friction(self):
        self.assertEqual(LEG_COUNTS["Short Put"], 1)
        two = friction_to_credit(_row(credit=1.0, strategy="Bull Put"), _Model(0.10))
        one = friction_to_credit(_row(credit=1.0, strategy="Short Put"), _Model(0.10))
        self.assertAlmostEqual(one, two / 2)

    def test_no_credit_recorded_yields_none_not_zero(self):
        # A missing credit is a row to skip, never a free trade.
        self.assertIsNone(friction_to_credit(
            _row(credit=0.0) | {"entry_price": 0.0}, _Model(0.05)))


class ClusterBootstrapTest(unittest.TestCase):
    def test_it_resamples_days_not_trades(self):
        # Two days, perfectly split: every bootstrap median must come from a
        # mixture of whole days, so with these values it can only ever be one
        # of a small set. Resampling trades individually would produce a much
        # wider spread of medians.
        values = [1.0] * 20 + [-1.0] * 20
        days = ["d1"] * 20 + ["d2"] * 20
        boots = cluster_bootstrap_medians(values, days, n_boot=400, seed=1)
        self.assertTrue(set(np.unique(boots)).issubset({-1.0, 0.0, 1.0}))

    def test_clustering_widens_the_interval_versus_ignoring_it(self):
        # The honest point of the whole design: pretending 40 correlated
        # observations are independent manufactures confidence.
        rng = np.random.default_rng(0)
        values, days = [], []
        for d in range(4):                       # only 4 real clusters
            shock = rng.normal(0, 1.0)
            for _ in range(10):
                values.append(shock + rng.normal(0, 0.05))
                days.append(f"d{d}")
        clustered = cluster_bootstrap_medians(values, days, n_boot=600, seed=2)
        each_own = cluster_bootstrap_medians(values, list(range(len(values))),
                                             n_boot=600, seed=2)
        self.assertGreater(clustered.std(), each_own.std())

    def test_one_cluster_cannot_support_a_bootstrap(self):
        self.assertEqual(cluster_bootstrap_medians([1, 2, 3], ["d"] * 3).size, 0)
        self.assertIsNone(posterior_median_above([1, 2, 3], ["d"] * 3))

    def test_posterior_is_a_probability(self):
        values = [0.2, 0.3, -0.1, 0.4, 0.25, 0.35]
        days = ["a", "a", "b", "b", "c", "c"]
        p = posterior_median_above(values, days)
        self.assertGreaterEqual(p, 0.0)
        self.assertLessEqual(p, 1.0)

    def test_it_is_deterministic_for_a_fixed_seed(self):
        v, d = [0.1, 0.2, -0.3, 0.4], ["a", "a", "b", "b"]
        self.assertEqual(posterior_median_above(v, d, seed=7),
                         posterior_median_above(v, d, seed=7))


class ArmADecisionTest(unittest.TestCase):
    def test_high_posterior_with_coherent_returns_is_ready(self):
        d, why = decide_arm_a(120, 0.97, median_ror=0.28, capital_weighted_ror=0.19)
        self.assertEqual(d, "READY")

    def test_ready_is_withheld_when_the_median_and_the_book_disagree(self):
        # The pennies-in-front-of-a-steamroller shape: typical trade wins,
        # the book loses. Authorising capital on that would be the whole
        # failure mode of short premium.
        d, why = decide_arm_a(120, 0.98, median_ror=0.13, capital_weighted_ror=-0.01)
        self.assertEqual(d, "EXTEND")
        self.assertIn("disagree in sign", why)

    def test_low_posterior_stops(self):
        d, _ = decide_arm_a(120, 0.10, median_ror=-0.05, capital_weighted_ror=-0.05)
        self.assertEqual(d, "STOP")

    def test_thin_effective_n_gathers_regardless_of_posterior(self):
        d, why = decide_arm_a(12, 0.99, median_ror=0.5, capital_weighted_ror=0.5)
        self.assertEqual(d, "GATHERING")
        self.assertIn("effective n", why)

    def test_extend_is_bounded_here_too(self):
        mid = dict(median_ror=0.1, capital_weighted_ror=0.1)
        self.assertEqual(decide_arm_a(120, 0.5, 0, **mid)[0], "EXTEND")
        self.assertEqual(decide_arm_a(120, 0.5, 2, **mid)[0], "STOP")


class CaveatTest(unittest.TestCase):
    def test_an_all_paper_cohort_says_so_first(self):
        rows = [_row() for _ in range(40)]
        text = " ".join(cohort_caveats(rows))
        self.assertIn("paper_only=1", text)

    def test_the_unobserved_tail_is_always_declared(self):
        rows = [_row(date=f"2026-06-{d:02d}") for d in range(1, 20)]
        text = " ".join(cohort_caveats(rows))
        self.assertIn("tail is unobserved", text.lower())
        self.assertIn("no evidence against", text)

    def test_clustering_is_declared_when_entries_pile_up(self):
        rows = [_row(date="2026-06-01") for _ in range(30)]
        text = " ".join(cohort_caveats(rows))
        self.assertIn("entry days", text)

    def test_an_empty_cohort_is_not_silently_fine(self):
        self.assertEqual(cohort_caveats([]), ["Cohort is empty."])


class ExitFidelityCaveatTest(unittest.TestCase):
    """The exit-fidelity caveat states two facts — when the scheduler died and
    what share of stopped trades ran past their stop. Both were string
    literals, so the caveat asserted "dead since 2026-06-15" and "94%" no
    matter what the ledger said, and would go on asserting them after the
    scheduler was fixed. A caveat that cannot stop being true is not evidence
    about anything; it has to be measured from the same ledger it qualifies.
    """

    def _ledger(self, path, overshot, on_rule):
        """A ledger with `overshot` trades past their stop and `on_rule` on it."""
        conn = sqlite3.connect(path)
        conn.execute("CREATE TABLE trades (entry_id INTEGER PRIMARY KEY, "
                     "date TEXT, ticker TEXT, strategy_name TEXT, status TEXT, "
                     "exit_date TEXT, exit_reason TEXT, pnl_pct REAL, "
                     "pnl_usd REAL, capital_at_risk REAL)")
        n = 0
        for _ in range(overshot):
            n += 1
            conn.execute("INSERT INTO trades (entry_id, date, ticker, "
                         "strategy_name, status, exit_date, exit_reason, pnl_pct) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         (n, "2026-07-01", "AAPL", "Bull Put", "CLOSED",
                          "2026-07-10", "Stop Loss (-50%)", -0.90))
        for _ in range(on_rule):
            n += 1
            conn.execute("INSERT INTO trades (entry_id, date, ticker, "
                         "strategy_name, status, exit_date, exit_reason, pnl_pct) "
                         "VALUES (?,?,?,?,?,?,?,?)",
                         (n, "2026-07-01", "AAPL", "Bull Put", "CLOSED",
                          "2026-07-10", "Stop Loss (-50%)", -0.50))
        conn.commit(); conn.close()

    def test_the_share_is_measured_from_the_ledger(self):
        rows = [_row(date="2026-07-01") for _ in range(30)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            self._ledger(p, overshot=3, on_rule=1)  # 75%, not 94%
            text = " ".join(cohort_caveats(rows, db_path=p))
        self.assertIn("75%", text)
        self.assertNotIn("94%", text)

    def test_a_different_ledger_gives_a_different_share(self):
        rows = [_row(date="2026-07-01") for _ in range(30)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            self._ledger(p, overshot=1, on_rule=1)  # 50%
            text = " ".join(cohort_caveats(rows, db_path=p))
        self.assertIn("50%", text)

    def test_a_ledger_with_no_stop_exits_claims_no_share(self):
        # Nothing to measure is not the same as "94%". Inventing a number here
        # is the exact failure the literal represented.
        rows = [_row(date="2026-07-01") for _ in range(30)]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "t.db")
            self._ledger(p, overshot=0, on_rule=0)
            text = " ".join(cohort_caveats(rows, db_path=p))
        self.assertNotIn("94%", text)
        self.assertNotIn("% of stopped", text)

    def test_it_still_warns_without_a_readable_ledger(self):
        # The caveat qualifies the verdict; losing the measurement must not
        # silently drop the warning it carries.
        rows = [_row(date="2026-07-01") for _ in range(30)]
        text = " ".join(cohort_caveats(rows, db_path="/nonexistent/ledger.db"))
        self.assertIn("manual cadence", text)


if __name__ == "__main__":
    unittest.main()
