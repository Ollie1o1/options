"""The scorer must be reproducible when you pin the instant it prices at.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_scoring_reproducible -v

Two causes of run-to-run drift were found on 2026-08-10, and only one was a bug:

  1. The Monte Carlo seed was built with `hash()` over a tuple containing a
     string, which Python randomises per process. Same chain, different
     process, different PoP — up to 2.0e-02 of `quality_score` movement, 2% of
     its range. Fixed in 4bceef5 with blake2b.

  2. `T_years` is `(expiry - datetime.now(utc))`, at sub-second resolution.
     Two processes 1.371 seconds apart produced `T_years` 1.371 seconds apart,
     which propagates through Black-Scholes into every Greek, `prob_profit`,
     `pop_score` and finally `quality_score` at ~7e-8.

The second is not a bug: a contract really does have less time left a second
later, and a live scan must price it that way. It is, however, a
reproducibility problem — you cannot re-run a scan and get the scan back.

So the instant is injectable. `as_of=None` keeps live behaviour exactly as it
was; passing one makes the whole scoring path a pure function of its inputs.
"""
import importlib.util
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "_scorer_fx_repro", "tests/test_scorer_signal_recovery.py")
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)

from src import options_screener as osc


def _score(as_of=None):
    """The fixture chain, scored, optionally at a pinned instant."""
    df = _fx._make_chain(n=40)
    kwargs = dict(
        df=df, min_dte=1, max_dte=400, risk_free_rate=0.04,
        config=_fx._config(), vix_regime_weights={},
    )
    if as_of is not None:
        kwargs["as_of"] = as_of
    return osc.enrich_and_score(**kwargs)


class PinnedClockTest(unittest.TestCase):

    AS_OF = datetime(2026, 8, 10, 15, 30, tzinfo=timezone.utc)

    def test_the_same_instant_gives_the_same_scores(self):
        a = _score(self.AS_OF)
        b = _score(self.AS_OF)
        np.testing.assert_array_equal(
            a["quality_score"].to_numpy(), b["quality_score"].to_numpy(),
            "same chain, same instant, different scores")

    def test_the_same_instant_gives_identical_time_to_expiry(self):
        a, b = _score(self.AS_OF), _score(self.AS_OF)
        np.testing.assert_array_equal(a["T_years"].to_numpy(),
                                      b["T_years"].to_numpy())

    def test_every_greek_is_reproducible(self):
        a, b = _score(self.AS_OF), _score(self.AS_OF)
        for col in ("delta", "gamma", "vega", "theta", "rho"):
            if col in a.columns:
                np.testing.assert_array_equal(
                    a[col].to_numpy(), b[col].to_numpy(), f"{col} drifted")

    def test_a_later_instant_really_does_shorten_time_to_expiry(self):
        """Reproducibility must not be bought by ignoring the clock."""
        early = _score(self.AS_OF)
        late = _score(self.AS_OF + timedelta(days=1))
        self.assertTrue((late["T_years"].to_numpy()
                         < early["T_years"].to_numpy()).all())

    def test_the_guarantee_does_not_expire_at_midnight(self):
        """The Monte Carlo seed is dated, so it must date off `as_of` too.

        Left on wall-clock now, pinning the instant would make a scan
        reproducible today and not tomorrow — the seed rolls over at midnight
        and PoP moves with it. A guarantee that silently expires is worse than
        no guarantee, because nobody re-checks it.
        """
        import unittest.mock as mock
        from datetime import datetime as _dt

        real = _dt.now

        class _Tomorrow(_dt):
            @classmethod
            def now(cls, tz=None):
                base = real(tz) if tz else real()
                return base + timedelta(days=1)

        today = _score(self.AS_OF)
        with mock.patch.object(osc, "datetime", _Tomorrow):
            tomorrow = _score(self.AS_OF)
        np.testing.assert_array_equal(
            today["quality_score"].to_numpy(),
            tomorrow["quality_score"].to_numpy(),
            "a pinned scan stopped reproducing when the wall clock rolled over")

    def test_live_behaviour_is_unchanged_when_no_instant_is_given(self):
        """`as_of=None` must still price at wall-clock now."""
        out = _score(None)
        expected = (out["exp_dt"] - datetime.now(timezone.utc)).dt.total_seconds() \
            / (365.0 * 24 * 3600)
        # Generous: this asserts it tracks the wall clock, not that it is exact.
        self.assertTrue(
            (np.abs(out["T_years"].to_numpy() - expected.to_numpy()) < 1e-4).all())


if __name__ == "__main__":
    unittest.main()
