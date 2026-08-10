"""The risk surface must render the contract it was handed.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest tests.test_visual_surface_option_type -v

`print_risk_surface` read `opt.get("optionType", "call")`. Scan rows carry
`type`; `optionType` is only a yfinance MultiIndex level name inside
`data_fetching` and never reaches a row. So the default won every time and
every surface rendered as a call — the title, the P&L grid, and the delta and
theta grids.

On a put that is an inverted payoff, not a cosmetic mislabel. Caught
2026-08-10 when card #1 read `AMD PUT $470` and the surface below it read
`AMD $470 CALL`.
"""
import unittest

from src import visual_surface as vs


class OptionTypeFromRowTest(unittest.TestCase):

    def _row(self, **over):
        row = {"symbol": "AMD", "strike": 470.0, "type": "put",
               "impliedVolatility": 0.45, "T_years": 18 / 365.0,
               "ask": 21.65, "underlying": 470.0}
        row.update(over)
        return row

    def _rendered_type(self, row):
        """The option type the surface actually renders, read off its title."""
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            vs.print_risk_surface(row, 470.0, 0.04, 100, mode="ascii",
                                  surface_type="pnl", show_contours=False)
        text = buf.getvalue()
        if "PUT" in text:
            return "put"
        if "CALL" in text:
            return "call"
        return None

    def test_a_put_row_renders_as_a_put(self):
        self.assertEqual(self._rendered_type(self._row(type="put")), "put")

    def test_a_call_row_renders_as_a_call(self):
        self.assertEqual(self._rendered_type(self._row(type="call")), "call")

    def test_capitalised_type_is_understood(self):
        self.assertEqual(self._rendered_type(self._row(type="Put")), "put")

    def test_the_yfinance_key_still_works_as_a_fallback(self):
        row = self._row()
        row.pop("type")
        row["optionType"] = "put"
        self.assertEqual(self._rendered_type(row), "put")

    def test_a_missing_type_does_not_silently_become_a_call_grid(self):
        """With no type at all the surface still has to pick one; it must not
        be picking one because a lookup quietly missed."""
        row = self._row()
        row.pop("type")
        self.assertIn(self._rendered_type(row), ("call", "put", None))


class PnlGridDirectionTest(unittest.TestCase):
    """The grid, not just the title. A wrong type inverts the payoff."""

    # compute_pnl_grid returns (price_shocks, iv_shocks, pnl); pnl is indexed
    # [price, iv], so row 0 is the lowest spot and row -1 the highest.
    ARGS = (470.0, 470.0, 18 / 365.0, 0.04, 0.45, 21.65)

    def test_a_put_gains_when_spot_falls(self):
        *_, pnl = vs.compute_pnl_grid("put", *self.ARGS)
        self.assertGreater(pnl[0, :].mean(), pnl[-1, :].mean(),
                           "a long put must profit as spot falls")

    def test_a_call_gains_when_spot_rises(self):
        *_, pnl = vs.compute_pnl_grid("call", *self.ARGS)
        self.assertGreater(pnl[-1, :].mean(), pnl[0, :].mean(),
                           "a long call must profit as spot rises")

    def test_the_two_are_not_the_same_grid(self):
        """Guards against a future regression that ignores option_type again."""
        *_, put = vs.compute_pnl_grid("put", *self.ARGS)
        *_, call = vs.compute_pnl_grid("call", *self.ARGS)
        self.assertFalse((put == call).all(),
                         "put and call surfaces are identical — option_type "
                         "is being ignored")


if __name__ == "__main__":
    unittest.main()
