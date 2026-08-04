"""Tests for src/lab — expressing an idea and running it honestly.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_lab_engine -v
"""
import unittest

from src.lab import core, engine


def _flat_prices(n=400, start=100.0, drift=0.0):
    return {"TEST": [(f"2020-01-{i+1:02d}", start * (1 + drift) ** i) for i in range(n)]}


class EntryTest(unittest.TestCase):
    def test_an_entry_names_a_single_leg_and_a_holding_period(self):
        e = core.Entry(kind="long_call", delta=0.40, dte=90, hold_days=45)
        self.assertEqual(e.kind, "long_call")
        self.assertEqual(e.n_legs, 1)

    def test_an_unknown_kind_is_rejected(self):
        with self.assertRaises(ValueError):
            core.Entry(kind="butterfly", delta=0.40, dte=90, hold_days=45)

    def test_holding_longer_than_the_option_lives_is_rejected(self):
        with self.assertRaises(ValueError):
            core.Entry(kind="long_call", delta=0.40, dte=30, hold_days=60)


class ContextTest(unittest.TestCase):
    """What an idea gets to look at. Features come from price history only —
    there is no IV history to speak of (83 days), so nothing may depend on it."""

    def setUp(self):
        self.bars = [(f"d{i}", 100.0 + i) for i in range(120)]

    def test_realized_vol_of_a_straight_line_is_near_zero(self):
        ctx = core.Context.at(self.bars, index=100, symbol="TEST")
        self.assertLess(ctx.realized_vol, 0.05)

    def test_momentum_is_positive_on_a_rising_series(self):
        ctx = core.Context.at(self.bars, index=100, symbol="TEST")
        self.assertGreater(ctx.momentum_21d, 0)

    def test_context_before_enough_history_is_refused(self):
        self.assertIsNone(core.Context.at(self.bars, index=5, symbol="TEST"))


class RunTest(unittest.TestCase):
    """The loop. A long call on a flat, zero-vol series must lose exactly its
    premium plus friction — there is nothing else for it to do."""

    def test_a_long_call_on_a_dead_flat_series_loses_its_premium(self):
        bars = [(f"d{i}", 100.0) for i in range(400)]
        idea = lambda ctx: core.Entry("long_call", delta=0.40, dte=60, hold_days=30)
        res = engine.run(idea, {"TEST": bars}, iv_model=lambda ctx: 0.30, every_n=90)
        self.assertGreater(res.n, 0)
        self.assertLess(res.mean_return, 0.0)

    def test_every_trade_is_tagged_with_how_it_was_priced(self):
        bars = [(f"d{i}", 100.0) for i in range(400)]
        idea = lambda ctx: core.Entry("long_call", delta=0.40, dte=60, hold_days=30)
        res = engine.run(idea, {"TEST": bars}, iv_model=lambda ctx: 0.30, every_n=90)
        self.assertEqual({t.source for t in res.trades}, {"modeled"})

    def test_an_idea_returning_none_takes_no_trade(self):
        bars = [(f"d{i}", 100.0) for i in range(400)]
        res = engine.run(lambda ctx: None, {"TEST": bars},
                         iv_model=lambda ctx: 0.30, every_n=90)
        self.assertEqual(res.n, 0)

    def test_costs_are_charged_on_entry_and_exit(self):
        """The whole point of the execution work: a backtest that fills at the
        mid is the defect. Free trading must beat charged trading, always."""
        bars = [(f"d{i}", 100.0 + i * 0.5) for i in range(400)]
        idea = lambda ctx: core.Entry("long_call", delta=0.40, dte=60, hold_days=30)
        charged = engine.run(idea, {"TEST": bars}, iv_model=lambda c: 0.30, every_n=90)
        free = engine.run(idea, {"TEST": bars}, iv_model=lambda c: 0.30, every_n=90,
                          frictionless=True)
        self.assertLess(charged.mean_return, free.mean_return)


class IVSweepTest(unittest.TestCase):
    """The honesty gate. Tier-3 results depend on a modelled IV whose plausible
    range, measured on real DoltHub IV against realized vol, is 0.70x-1.52x.
    A verdict that flips inside that band is not a verdict."""

    def test_a_sweep_reports_a_result_at_each_multiplier(self):
        bars = [(f"d{i}", 100.0) for i in range(400)]
        idea = lambda ctx: core.Entry("long_call", delta=0.40, dte=60, hold_days=30)
        sweep = engine.sweep_iv(idea, {"TEST": bars}, base_iv=lambda c: 0.30,
                                every_n=90)
        self.assertEqual(set(sweep.results), set(engine.IV_MULTIPLIERS))

    def test_a_result_that_flips_sign_across_the_band_is_not_robust(self):
        sweep = engine.IVSweep(results={0.70: _r(+0.10), 1.00: _r(+0.01), 1.52: _r(-0.08)})
        self.assertFalse(sweep.robust)
        self.assertIn("flips", sweep.verdict)

    def test_a_result_negative_everywhere_is_robust(self):
        sweep = engine.IVSweep(results={0.70: _r(-0.10), 1.00: _r(-0.05), 1.52: _r(-0.02)})
        self.assertTrue(sweep.robust)
        self.assertIn("negative", sweep.verdict)


def _r(mean):
    return core.Result(trades=[], n=10, mean_return=mean, median_return=mean,
                       win_rate=0.5, source_counts={"modeled": 10})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()


class StrikeForDeltaTest(unittest.TestCase):
    """Delta is invertible in closed form. Searching 120 candidate strikes per
    trade cost ~10M Black-Scholes calls per configuration and made a full run
    take longer than the analysis was worth."""

    def test_the_strike_reproduces_the_requested_call_delta(self):
        from src.utils import bs_delta
        from src.lab import pricing as p
        ctx = core.Context(symbol="T", date="d", spot=100.0, realized_vol=0.30,
                           realized_vol_252d=0.30, momentum_21d=0.0,
                           momentum_63d=0.0, drawdown=0.0)
        for target in (0.20, 0.40, 0.60, 0.80):
            e = core.Entry("long_call", target, 365, 180)
            k = engine._strike_for_delta(ctx, e, 0.30)
            got = abs(float(bs_delta("call", 100.0, k, 1.0, p.DEFAULT_RATE, 0.30)))
            self.assertAlmostEqual(got, target, places=4)

    def test_the_strike_reproduces_the_requested_put_delta(self):
        from src.utils import bs_delta
        from src.lab import pricing as p
        ctx = core.Context(symbol="T", date="d", spot=100.0, realized_vol=0.30,
                           realized_vol_252d=0.30, momentum_21d=0.0,
                           momentum_63d=0.0, drawdown=0.0)
        for target in (0.20, 0.40, 0.60):
            e = core.Entry("long_put", target, 365, 180)
            k = engine._strike_for_delta(ctx, e, 0.30)
            got = abs(float(bs_delta("put", 100.0, k, 1.0, p.DEFAULT_RATE, 0.30)))
            self.assertAlmostEqual(got, target, places=4)

    def test_a_lower_delta_call_is_further_out_of_the_money(self):
        ctx = core.Context(symbol="T", date="d", spot=100.0, realized_vol=0.30,
                           realized_vol_252d=0.30, momentum_21d=0.0,
                           momentum_63d=0.0, drawdown=0.0)
        low = engine._strike_for_delta(ctx, core.Entry("long_call", 0.20, 365, 180), 0.30)
        high = engine._strike_for_delta(ctx, core.Entry("long_call", 0.60, 365, 180), 0.30)
        self.assertGreater(low, high)


class ShortSideTest(unittest.TestCase):
    """A sold option makes money when it decays. The first version of this
    engine reported short puts winning 21% of the time and losing 100% at every
    risk level, which is not a result — it is a sign error."""

    def _flat(self, n=300, price=100.0):
        return {"TEST": [(f"d{i}", price) for i in range(n)]}

    def test_a_short_put_on_a_flat_series_makes_money(self):
        """Spot never moves, so the option decays and the seller keeps it."""
        idea = lambda ctx: core.Entry("short_put", delta=0.30, dte=60, hold_days=30)
        res = engine.run(idea, self._flat(), iv_model=lambda c: 0.30, every_n=60)
        self.assertGreater(res.n, 0)
        self.assertGreater(res.mean_return, 0.0)

    def test_a_short_put_loses_when_the_underlying_collapses(self):
        bars = [(f"d{i}", 100.0 if i < 100 else 40.0) for i in range(300)]
        idea = lambda ctx: core.Entry("short_put", delta=0.30, dte=60, hold_days=30)
        res = engine.run(idea, {"TEST": bars}, iv_model=lambda c: 0.30, every_n=30)
        self.assertLess(min(t.ret for t in res.trades), 0.0)

    def test_a_short_call_loses_when_the_underlying_rips(self):
        bars = [(f"d{i}", 100.0 if i < 100 else 200.0) for i in range(300)]
        idea = lambda ctx: core.Entry("short_call", delta=0.30, dte=60, hold_days=30)
        res = engine.run(idea, {"TEST": bars}, iv_model=lambda c: 0.30, every_n=30)
        self.assertLess(min(t.ret for t in res.trades), 0.0)

    def test_a_short_return_is_measured_against_capital_at_risk(self):
        """Not against the closing debit. A cash-secured put ties up the strike;
        expressing its return against what it cost to buy back makes a cheap
        close look like an enormous gain and is why the numbers exploded."""
        idea = lambda ctx: core.Entry("short_put", delta=0.30, dte=60, hold_days=30)
        res = engine.run(idea, self._flat(), iv_model=lambda c: 0.30, every_n=60)
        # Decaying a ~0.30-delta put for 30 days cannot return more than a few
        # percent of the collateral behind it.
        self.assertLess(max(t.ret for t in res.trades), 0.25)
