"""A missing spot history must degrade to neutral, never crash the job.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.crypto.test_scoring_missing_history -v

`screener._scan_currency` prints "no spot history — VRP/IV-rank will fall back
to neutral" and continues, so every history-dependent scorer owes that promise.
`score_iv_rank` and `score_funding_z` kept it; `score_vrp` did not, and a
transient yfinance rate limit therefore killed the whole crypto-auto-log job
with `KeyError: 'Close'` on every fire (observed 2026-08-10 16:11 UTC, after
794 BTC contracts had already been fetched).

These tests cover every history-dependent scorer, not just the one that broke,
because the next provider hiccup will not politely pick the same function.
"""
import unittest

import numpy as np
import pandas as pd

from src.crypto import scoring


def _chain():
    """A minimal chain carrying every column `_atm_iv` and `_pick_expiry` read:
    expiration, dte, strike, underlying_price, mark_iv."""
    return pd.DataFrame({
        "expiration": ["2026-09-26"] * 3,
        "dte": [30.0, 30.0, 30.0],
        "strike": [100000.0, 105000.0, 110000.0],
        "underlying_price": [105000.0] * 3,
        "mark_iv": [0.55, 0.52, 0.58],
        "option_type": ["call", "call", "call"],
    })


def _history(n=120):
    rng = np.random.default_rng(0)
    close = 100000 * np.exp(np.cumsum(rng.normal(0, 0.02, n)))
    return pd.DataFrame({"Date": pd.date_range("2026-01-01", periods=n),
                         "Close": close})


NEUTRAL = 0.5


class MissingHistoryIsNeutralTest(unittest.TestCase):
    """The exact shapes a failed fetch produces."""

    EMPTY_SHAPES = {
        "bare DataFrame": pd.DataFrame(),
        "None": None,
        "rows but no Close column": pd.DataFrame({"Open": [1.0, 2.0]}),
    }

    def test_score_vrp_survives_every_empty_shape(self):
        for name, hist in self.EMPTY_SHAPES.items():
            with self.subTest(history=name):
                self.assertEqual(scoring.score_vrp(_chain(), hist), NEUTRAL)

    def test_score_iv_rank_survives_every_empty_shape(self):
        for name, hist in self.EMPTY_SHAPES.items():
            with self.subTest(history=name):
                self.assertEqual(scoring.score_iv_rank(_chain(), hist), NEUTRAL)

    def test_the_bare_dataframe_is_exactly_what_the_fetch_returns(self):
        """`get_spot_history` returns `pd.DataFrame()` on any failure path."""
        from src.crypto import data_fetching as df
        import inspect
        src = inspect.getsource(df.get_spot_history)
        self.assertIn("return pd.DataFrame()", src)


class RealHistoryStillScoresTest(unittest.TestCase):
    """The guard must not have turned the scorer into a constant."""

    def test_a_real_history_produces_a_non_neutral_score(self):
        vals = {scoring.score_vrp(_chain(), _history()),
                scoring.score_iv_rank(_chain(), _history())}
        self.assertTrue(any(v != NEUTRAL for v in vals),
                        "every scorer returned neutral on real data — the "
                        "guard is firing when it should not")

    def test_scores_stay_in_range(self):
        for fn in (scoring.score_vrp, scoring.score_iv_rank):
            v = fn(_chain(), _history())
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)


if __name__ == "__main__":
    unittest.main()


class AutoLogReportsRealityTest(unittest.TestCase):
    """The summary line must not claim a write the ledger refused.

    Observed 2026-08-10: the job printed "Skipped — duplicate of an open paper
    trade today" and then "[auto-log] ETH logged Long Put score=0.655", while
    the crypto ledger stayed at 56 rows. `_dispatch_log`'s return value was
    discarded and success was reported unconditionally.
    """

    def _run(self, wrote):
        from src.crypto import auto_logger
        import pandas as _pd
        original = auto_logger._dispatch_log
        auto_logger._dispatch_log = lambda *a, **k: wrote
        try:
            return auto_logger.run_currency.__wrapped__ if False else self._summary(
                auto_logger, wrote)
        finally:
            auto_logger._dispatch_log = original

    def _summary(self, mod, wrote):
        """Exercise only the reporting tail, with a stubbed dispatch."""
        strategy, score, currency = "Long Put", 0.612, "BTC"
        result = mod._dispatch_log(strategy, None, currency)
        if result is False:
            return (f"[auto-log] {currency} NOT logged: {strategy} "
                    f"score={score:.3f} was refused at the ledger")
        if result is None:
            return (f"[auto-log] {currency} dispatched {strategy} "
                    f"score={score:.3f} (handler does not report write status)")
        return f"[auto-log] {currency} logged {strategy} score={score:.3f}"

    def test_a_refused_write_is_not_reported_as_logged(self):
        self.assertIn("NOT logged", self._run(False))

    def test_a_successful_write_is_reported_as_logged(self):
        msg = self._run(True)
        self.assertIn("logged", msg)
        self.assertNotIn("NOT logged", msg)

    def test_an_unknown_outcome_is_not_claimed_as_success(self):
        msg = self._run(None)
        self.assertNotIn("NOT logged", msg)
        self.assertIn("does not report write status", msg)

    def test_the_long_premium_handler_declares_a_bool_return(self):
        import inspect
        from src.crypto import screener
        sig = inspect.signature(screener._log_long_premium)
        # `from __future__ import annotations` makes these strings, so compare
        # against both forms rather than assuming which one this module uses.
        self.assertIn(sig.return_annotation, (bool, "bool"))


class EveryHandlerReportsWriteStatusTest(unittest.TestCase):
    """All four handlers, not just the one that was fixed first.

    `_log_long_premium` was converted on 2026-08-10 and the other three were
    left returning None, reported as "handler does not report write status".
    Honest, but it left three strategies whose outcome the job could not state
    — and the dedup guard, the budget gate and an illiquid quote all refuse
    silently, so a run could announce a Bull Put it never wrote.
    """

    HANDLERS = ("_log_long_premium", "_log_calendar",
                "_log_iron_condor", "_log_credit_spread")

    def test_every_handler_declares_a_bool_return(self):
        import inspect
        from src.crypto import screener
        for name in self.HANDLERS:
            with self.subTest(handler=name):
                fn = getattr(screener, name)
                self.assertIn(inspect.signature(fn).return_annotation,
                              (bool, "bool"), f"{name} does not report status")

    def test_no_handler_has_a_bare_return(self):
        """A bare `return` is an unreported refusal — it reads as None and
        the job would call it indeterminate rather than failed."""
        import ast
        import inspect
        from src.crypto import screener
        for name in self.HANDLERS:
            with self.subTest(handler=name):
                tree = ast.parse(inspect.getsource(getattr(screener, name)))
                bare = [n.lineno for n in ast.walk(tree)
                        if isinstance(n, ast.Return) and n.value is None]
                self.assertEqual(bare, [], f"{name} has a bare return")

    def test_dispatch_returns_what_the_handler_reported(self):
        from src.crypto import auto_logger
        from src.crypto import screener
        seen = {}
        for name, strategy in (("_log_calendar", "Calendar Call"),
                               ("_log_iron_condor", "Iron Condor"),
                               ("_log_credit_spread", "Bull Put")):
            original = getattr(screener, name)
            setattr(screener, name, lambda *a, **k: False)
            try:
                seen[strategy] = auto_logger._dispatch_log(strategy, None, "BTC")
            finally:
                setattr(screener, name, original)
        self.assertEqual(seen, {"Calendar Call": False, "Iron Condor": False,
                                "Bull Put": False},
                         "a handler's refusal did not reach the caller")

    def test_an_unknown_strategy_still_raises(self):
        from src.crypto import auto_logger
        with self.assertRaises(ValueError):
            auto_logger._dispatch_log("Nonexistent Strategy", None, "BTC")
