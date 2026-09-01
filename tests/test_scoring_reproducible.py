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
from datetime import date, datetime, timedelta, timezone

import numpy as np

_spec = importlib.util.spec_from_file_location(
    "_scorer_fx_repro", "tests/test_scorer_signal_recovery.py")
_fx = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_fx)

from src import options_screener as osc


def _chain_anchored(anchor):
    """The fixture chain with expiries measured from `anchor`, not wall-clock.

    `_make_chain` dates its expiries off `datetime.today()`, while these tests
    pin the scoring instant to AS_OF. The gap between the two grows every day,
    so which contracts survive the DTE filters drifts with the calendar even
    though the test pinned the clock.

    On 2026-09-01 that gap put a contract exactly on the `max_dte=400`
    boundary: scoring at AS_OF kept 10 rows, scoring at AS_OF + 1 day kept 9,
    and an element-wise comparison died on shapes (9,) vs (10,). The suite
    passed on 2026-08-31 and again on 2026-09-02 — a test that fails on one
    calendar date is measuring the date, not reproducibility.

    The anchor is a FIXED CALENDAR DATE shared by every scoring instant in a
    comparison, deliberately NOT `as_of`. Re-anchoring per `as_of` would move
    each contract's expiry along with the clock, leaving `T_years` identical
    between the two runs and silently destroying what
    `test_a_later_instant_really_does_shorten_time_to_expiry` exists to check.

    Offsets are taken from the chain's OWN earliest expiry rather than from
    `datetime.today()`. Reading a clock here would reintroduce the same bug one
    level up: this module's `datetime` is not the fixture's `datetime`, so any
    test that pins one and not the other gets a silently wrong offset.
    """
    rows = _fx._make_chain(n=40).copy()
    parsed = [datetime.strptime(e, "%Y-%m-%d").date()
              for e in rows["expiration"]]
    base = min(parsed)
    rows["expiration"] = [
        (anchor + timedelta(days=(p - base).days)).strftime("%Y-%m-%d")
        for p in parsed
    ]
    return rows


# One anchor for the whole module, so the chain is identical on every run date
# and the only thing that varies is the instant it is priced at.
#
# It sits 10 days AFTER AS_OF so the chain's earliest expiry lands inside the
# scorer's min_dte=1..max_dte=400 window measured from AS_OF, and stays there on
# every run date. An anchor at or before AS_OF would put every contract in the
# past, score zero rows, and make every array assertion in this file vacuously
# true — `(a < b).all()` is True on empty arrays. Green, and measuring nothing.
# `test_the_fixture_still_has_contracts_to_score` is the guard on that.
_ANCHOR = date(2026, 8, 20)


def _score(as_of=None):
    """The fixture chain, scored, optionally at a pinned instant."""
    df = _fx._make_chain(n=40) if as_of is None else _chain_anchored(_ANCHOR)
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
        self.assertEqual(len(early), len(late),
                         "the two runs scored different numbers of contracts, "
                         "so the comparison below is not like-for-like")
        self.assertTrue((late["T_years"].to_numpy()
                         < early["T_years"].to_numpy()).all())

    def test_the_fixture_still_has_contracts_to_score(self):
        """An empty frame passes every array comparison in this module.

        `assert (a < b).all()` is vacuously True on empty arrays, so a fixture
        that has aged out of its own DTE window would turn this whole file
        green while measuring nothing. The anchored chain is dated relative to
        AS_OF and does not age, but if that ever changes this fails loudly
        rather than passing silently.
        """
        self.assertGreater(len(_score(self.AS_OF)), 0,
                           "the fixture scored no contracts — every other "
                           "assertion in this module is now vacuous")

    def test_the_fixture_does_not_drift_with_the_calendar(self):
        """The bug that broke CI on 2026-09-01, and only on 2026-09-01.

        `_make_chain` dates expiries off wall-clock `datetime.today()` while
        the test pins the scoring instant. As the gap grew, a contract crossed
        the `max_dte=400` boundary in one run but not the other: 10 rows vs 9,
        and an element-wise comparison on mismatched shapes.

        A suite that passes on 2026-08-31, fails on 2026-09-01 and passes again
        on 2026-09-02 is not measuring reproducibility — it is measuring the
        date. This asserts the row count is a function of the pinned instant
        alone, at wall-clock dates that previously disagreed.
        """
        import datetime as _dtmod
        import unittest.mock as mock

        counts = []
        for wall in (datetime(2026, 8, 31), datetime(2026, 9, 1),
                     datetime(2026, 9, 2), datetime(2027, 3, 15)):
            class _FixedToday(_dtmod.datetime):
                _w = wall

                @classmethod
                def today(cls):
                    return cls._w

            with mock.patch.object(_fx, "datetime", _FixedToday):
                counts.append((len(_score(self.AS_OF)),
                               len(_score(self.AS_OF + timedelta(days=1)))))

        self.assertEqual(len(set(counts)), 1,
                         f"row counts move with the wall clock: {counts}")
        self.assertEqual(counts[0][0], counts[0][1],
                         f"the two pinned instants disagree: {counts[0]}")

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
