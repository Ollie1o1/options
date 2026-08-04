"""Coercing a ledger column to a usable number.

`max_loss_usd` and `long_strike` arrive from SQLite as float, int, str, None,
or empty string depending on how old the row is and which writer produced it.
Three separate sites in `view_portfolio` carried the same
`float(x) if x not in (None, "", 0) else <fallback>` inside a try/except, which
is where mypy's `Argument 1 to "float" has incompatible type "Any | None"` came
from — the guard is real, but `not in (None, ...)` is not something a checker
can narrow.

These pin the behaviour of the shared helper the three sites now call. The
substantive rule is that **0 is a fallback, not a value**: a defined-risk
structure with a recorded max loss of zero is a missing figure, and treating it
as real would report a spread whose risk is nothing.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.check_pnl import _num_or_none  # noqa: E402


class UsableValuesTest(unittest.TestCase):
    def test_a_float_passes_through(self):
        self.assertEqual(_num_or_none(12.5), 12.5)

    def test_an_int_becomes_a_float(self):
        self.assertEqual(_num_or_none(12), 12.0)

    def test_a_numeric_string_is_parsed(self):
        # Legacy rows wrote numbers as text.
        self.assertEqual(_num_or_none("12.5"), 12.5)

    def test_a_negative_value_is_kept_as_is(self):
        # Sign handling belongs to the caller (which takes abs); the coercion
        # must not silently normalise it.
        self.assertEqual(_num_or_none(-3.0), -3.0)


class AbsentValuesTest(unittest.TestCase):
    """Everything here means "no figure recorded" and must yield None so the
    caller can choose its own fallback."""

    def test_none_yields_none(self):
        self.assertIsNone(_num_or_none(None))

    def test_an_empty_string_yields_none(self):
        self.assertIsNone(_num_or_none(""))

    def test_zero_yields_none(self):
        self.assertIsNone(_num_or_none(0))

    def test_zero_float_yields_none(self):
        self.assertIsNone(_num_or_none(0.0))

    def test_a_missing_key_yields_none(self):
        self.assertIsNone(_num_or_none({}.get("max_loss_usd")))


class ZeroAsAStringIsInconsistentTest(unittest.TestCase):
    """A quirk carried over deliberately rather than fixed here.

    Numeric 0 reads as absent, but the STRING "0" reads as the value 0.0 —
    because `"0" == 0` is False in Python, so the original guard let it through
    to `float()`. Inconsistent, and preserved: this pass is behaviour-neutral,
    and a display path is the wrong place to discover that changing it matters.

    In practice the ledger writes these columns numerically, so the string form
    is a legacy shape rather than a live one. Worth revisiting if a caller ever
    depends on it.
    """

    def test_the_string_zero_is_still_treated_as_a_value(self):
        self.assertEqual(_num_or_none("0"), 0.0)


class UnparseableValuesTest(unittest.TestCase):
    """A bad value must not take the portfolio view down: this runs per row
    while printing open positions."""

    def test_a_non_numeric_string_yields_none(self):
        self.assertIsNone(_num_or_none("n/a"))

    def test_a_list_yields_none(self):
        self.assertIsNone(_num_or_none([1, 2]))

    def test_a_dict_yields_none(self):
        self.assertIsNone(_num_or_none({"a": 1}))

    def test_nan_is_rejected(self):
        # float("nan") parses but poisons every sum it reaches, and a NaN
        # max_loss would silently NaN the whole concentration total.
        self.assertIsNone(_num_or_none(float("nan")))


class MatchesThePreviousExpressionTest(unittest.TestCase):
    """The helper replaces `x not in (None, "", 0)` guarding `float(x)`. Any
    input where the two disagree is a behaviour change, not a refactor — so
    the divergence is enumerated here rather than left to be discovered."""

    def _old(self, value):
        try:
            return float(value) if value not in (None, "", 0) else None
        except (TypeError, ValueError):
            return None

    def test_agrees_on_every_shape_the_ledger_produces(self):
        for value in (None, "", 0, 0.0, "0", 1, 1.5, -2.5, "3.25",
                      "n/a", [1], {"a": 1}, True, False):
            with self.subTest(value=value):
                self.assertEqual(_num_or_none(value), self._old(value),
                                 f"diverged on {value!r}")

    def test_nan_is_the_one_deliberate_divergence(self):
        # The old expression let NaN through: `nan not in (None, "", 0)` is
        # True, so `float(nan)` was returned. The caller adds that straight
        # into `total_cost_usd`, and one NaN makes the portfolio's whole
        # cost basis NaN — every downstream percentage with it.
        #
        # This is the single behaviour change in this pass, and it is a
        # narrowing: NaN now reads as "no figure", so the caller falls back to
        # cost basis exactly as it does for a missing column.
        old = self._old(float("nan"))
        self.assertNotEqual(old, old, "the old expression returned NaN")
        self.assertIsNone(_num_or_none(float("nan")))


if __name__ == "__main__":
    unittest.main()
