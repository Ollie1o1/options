"""The error bar on net EV is measured, and it scales with the vol level.

`ev_noise` scaled the band by `_IV_SIGMA_POINTS = {high 1.0, medium 1.5,
low 2.5}` — "how wrong a stored implied vol plausibly is". That is a
DATA-QUALITY question about the quoted IV field, and it is not the uncertainty
that governs net EV. Net EV is `BS(sigma_forecast) - market_price`; the
uncertain input is the REALIZED-VOL FORECAST, not the quote.

`scripts/vol_forecast_study.py` measured that forecast over 1,180
non-overlapping 21-day windows:

    overall error std : 10.10 IV points     (the hand-set bar said 1.0-2.5)
    median |error|    :  4.97 IV points

and, decisively, the error is PROPORTIONAL to the vol level rather than fixed:

    low vol    realized 0.132   error std  4.4 pts   ratio 0.33
    mid-low    realized 0.204   error std  5.7 pts   ratio 0.28
    mid-high   realized 0.279   error std  7.7 pts   ratio 0.28
    high vol   realized 0.461   error std 13.6 pts   ratio 0.30

A stable ~30% relative error across every regime. So the band is
`vega_dollar * 0.30 * vol * 100`, not a constant.

CONSEQUENCE, accepted deliberately: on a live board this turns 36 THIN /
23 CLEAR / 20 STRONG into 79 THIN. That is the honest reading — no single
option trade's edge is distinguishable from the uncertainty in the vol
forecast it was derived from — and it agrees with everything else the book
records (composite IC ~0, gate STOP, no entry feature predicts outcome). The
cost is that sigma becomes the binding margin on every row and masks the
friction and breakeven margins; ruled acceptable by the plan owner on
2026-08-17 in preference to a bar that certified noise as STRONG.
"""
from __future__ import annotations

import unittest

from src.tearsheet.collect import ev_noise
from src.trade_analysis import VOL_FORECAST_RELATIVE_ERROR


class TestTheBandIsProportionalToVol(unittest.TestCase):

    def _row(self, vol, vega_dollar=30.0, **kw):
        r = {"vega_dollar": vega_dollar, "hv_252d": vol, "hv_30d": vol}
        r.update(kw)
        return r

    def test_it_is_vega_times_the_measured_relative_error(self):
        got = ev_noise(self._row(0.20, vega_dollar=30.0))
        self.assertAlmostEqual(got, 30.0 * VOL_FORECAST_RELATIVE_ERROR * 20.0)

    def test_double_the_vol_doubles_the_band(self):
        low = ev_noise(self._row(0.20))
        high = ev_noise(self._row(0.40))
        self.assertAlmostEqual(high, 2 * low)

    def test_double_the_vega_doubles_the_band(self):
        self.assertAlmostEqual(ev_noise(self._row(0.25, vega_dollar=60.0)),
                               2 * ev_noise(self._row(0.25, vega_dollar=30.0)))

    def test_it_is_far_wider_than_the_old_hand_set_bar(self):
        """The old bar was 1.0-2.5 IV points regardless of vol."""
        row = self._row(0.25, vega_dollar=30.0)
        old_worst_case = 30.0 * 2.5
        self.assertGreater(ev_noise(row), 2 * old_worst_case)

    def test_the_blend_is_used_not_just_the_long_window(self):
        """One vol rule, shared with the EV itself — see ev_vol_basis."""
        both = ev_noise({"vega_dollar": 30.0, "hv_252d": 0.40, "hv_30d": 0.20})
        self.assertAlmostEqual(both, 30.0 * VOL_FORECAST_RELATIVE_ERROR * 30.0)


class TestTheQuoteQualityTableNoLongerDrivesIt(unittest.TestCase):
    """`iv_confidence` describes the QUOTE, not the forecast."""

    def test_confidence_does_not_change_the_band(self):
        base = {"vega_dollar": 30.0, "hv_252d": 0.25, "hv_30d": 0.25}
        vals = {c: ev_noise({**base, "iv_confidence": c})
                for c in ("high", "medium", "low", None)}
        self.assertEqual(len(set(round(v, 9) for v in vals.values())), 1,
                         f"quote confidence still scales the band: {vals}")


class TestFallbacks(unittest.TestCase):

    def test_no_vega_falls_back_to_the_cost_fraction(self):
        got = ev_noise({"ev_cost_per_contract": 40.0, "hv_252d": 0.25})
        self.assertAlmostEqual(got, 10.0)

    def test_no_vol_falls_back_rather_than_returning_zero(self):
        """Zero would claim perfect certainty, which is the one thing this
        number must never say."""
        got = ev_noise({"vega_dollar": 30.0, "ev_cost_per_contract": 40.0})
        self.assertGreater(got, 0.0)

    def test_nothing_at_all_is_zero(self):
        self.assertEqual(ev_noise({}), 0.0)

    def test_it_never_raises(self):
        for bad in ({"vega_dollar": "x"}, {"hv_252d": None}, {"vega_dollar": float("nan")}):
            ev_noise(bad)


class TestTheGradeConsequence(unittest.TestCase):
    """What the change is FOR: an edge that cannot clear its own uncertainty
    must not be certified STRONG."""

    def test_an_edge_inside_the_measured_band_is_not_strong(self):
        from src.worth import assess
        # $50 of net EV on a contract whose vega moves $30 per IV point, at
        # 25% vol: the band is 30 * 0.30 * 25 = $225. The edge is a fifth of it.
        row = {"ev_per_contract": 50.0, "vega_dollar": 30.0,
               "hv_252d": 0.25, "hv_30d": 0.25,
               "expiration": "2026-09-18", "date": "2026-08-17"}
        w = assess(row)
        self.assertNotIn(w.grade, ("STRONG", "CLEAR"))


class TestTheTwoVolRulesAgree(unittest.TestCase):
    """`ev_vol_basis` (vectorised, for the frame) and `vol_basis_of` (scalar,
    for the error bar) are separate implementations for speed. Two copies of a
    rule is exactly how this codebase produced a board ranked by one number and
    a table ranked by another, so the contract is pinned here."""

    def test_they_agree_across_the_cases_that_matter(self):
        import pandas as pd
        from src.trade_analysis import ev_vol_basis, vol_basis_of
        cases = [(0.30, 0.20), (0.40, None), (None, 0.22), (None, None),
                 (0.0, 0.20), (0.25, 0.0), (0.15, 0.15)]
        df = pd.DataFrame([{"hv_252d": a, "hv_30d": b} for a, b in cases])
        frame = ev_vol_basis(df)
        for i, (a, b) in enumerate(cases):
            scalar = vol_basis_of(a, b)
            f = frame.iloc[i]
            if scalar is None:
                self.assertTrue(pd.isna(f), f"case {(a, b)}: frame={f} scalar=None")
            else:
                self.assertAlmostEqual(float(f), scalar, msg=f"case {(a, b)}")


if __name__ == "__main__":
    unittest.main()
