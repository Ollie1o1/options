"""Guards for the ACCEPTED SVI fit instability (decided 2026-09-30).

`iv_surface._fit_single_expiry` runs Nelder-Mead on five badly-scaled parameters
over a slice that does not identify them, so a wall-clock nudge in `T_years` can
land it in a different basin and the whole fitted smile changes. That is LIVE
scoring behaviour and it was deliberately NOT fixed. What was measured before
accepting it (15 real boards from `data/chain_archive.db`, 2026-09-30, scored
through the real `enrich_and_score` at `as_of` and `as_of` + 0.1s):

  per-slice `iv_surface_confidence` moves  2.6e-5
  per-ROW   `iv_surface_residual`  moves   1.15e-1
  WITHIN-slice residual spread             1.79e-1
  deep-rank churn                          38 of 42 draws
  TOP-5 / TOP-10 membership change          0 of 42 draws  <- why it is accepted

The board reorders constantly in its tail and not at all at its head, and
nothing orders the board anyway: every ranking key's CI contains zero and the
board exists to REFUSE rather than to rank. Churn in an order nobody trusts is
not worth destabilising every score in the system to remove.

TWO FIXES WERE BUILT AND REJECTED ON MEASUREMENT -- see the long comment at
`options_screener.py`'s `as_of` injection point for the figures. In short:
multi-start (5 fixed starts, deterministic lowest-SSE pick) made the worst-case
drift 4x WORSE as a bare argmin and only 1.25x better at its best switch
margin, while degrading previously-stable boards 40-73x at 5.2x the runtime --
because the optima are near-equivalent (several reach an EXACTLY tied SSE) and
an argmin over near-ties is itself discontinuous. Quantizing the fit's T closed
it ~5 orders directly but only ~32x through the scorer, unexplained. Neither is
in the tree.

These two tests are deliberately DETERMINISTIC. A test that scored a chain twice
and asserted the top-N had not moved would be asserting a stochastic property
and would flake exactly the way the 2026-09-29 "red suite" did -- a flake
imitates an ordering claim perfectly. So instead they pin:

  1. the corrected MODEL of what reaches the composite, because the previous,
     WRONG model ("iv_mispricing_score IS the slice's confidence, one number per
     expiry, so the shift is uniform and cannot reorder anything") is what made
     this look harmless for a day, and
  2. the one CONDITION under which a top-N change was actually observed -- the
     `iv_mispricing` weight at 0.05 rather than its live 0.0104.

Run:
    PYTHONPATH=$PWD ~/.venvs/options/bin/python -m unittest \
        tests.test_svi_drift_board_stability -v
"""
import json
import unittest

import pandas as pd

import src.options_screener as options_screener

# At this weight a real board's top-10 changed between two runs 0.1s apart; at
# the live 0.0104 it did not in 42 draws. The guard sits between the two.
WEIGHT_AT_WHICH_TOP_N_MOVED = 0.05
MAX_SAFE_WEIGHT = 0.02


def _contract(**overrides):
    """One scorable row. Mirrors tests/test_absolute_scores._contract, which is
    built from the columns `calculate_scores` touches directly."""
    row = dict(
        strike=100.0, theta=-0.05, premium=2.0, bid=1.95, ask=2.05, volume=500,
        openInterest=2000, spread_pct=0.05, abs_delta=0.45,
        impliedVolatility=0.35, prob_profit=0.45, rr_ratio=2.0,
        underlying=100.0, T_years=0.12, vega=0.2, gamma=0.03, delta=0.45,
        event_flag="", Trend_Aligned=False, decay_warning=False, sr_warning="",
        oi_wall_warning="", macro_warning="", div_warning="",
        squeeze_play=False, is_squeezing=False, Unusual_Whale=False,
        quote_freshness="fresh", symbol="TEST", return_on_risk=0.5,
        em_realism_score=0.6, seasonal_win_rate=0.5, short_interest=0.05,
        hv_30d=0.30, iv_vs_hv=0.05, gamma_ramp=False, ev_per_contract=5.0,
        theta_decay_pressure=0.025, expected_move=5.0, max_loss=200.0,
    )
    row.update(overrides)
    return row


def _score(rows, mode="Scan"):
    df = pd.DataFrame(rows)
    df["type"] = "call"
    # ONE expiration: the point is variation WITHIN a single slice.
    df["expiration"] = "2026-09-18"
    with open("config.json") as fh:
        config = json.load(fh)
    return options_screener.calculate_scores(
        df, config, {"regime": "normal"}, "swing", mode, 7, 60)


class TheCompositeSeesAPerRowTerm(unittest.TestCase):
    """The corrected model. This is the test the earlier, wrong one lacked."""

    def test_iv_mispricing_score_varies_within_one_expiry(self):
        # Same expiry, IDENTICAL per-slice fit confidence, different per-row
        # residuals, all inside the unsaturated band (-0.2 < resid < 0 is where
        # the buyer branch clip(-resid*5, 0, 1) actually responds).
        rows = [_contract(strike=100.0 + i, iv_surface_residual=r,
                          iv_surface_confidence=0.9)
                for i, r in enumerate((-0.02, -0.06, -0.10))]
        out = _score(rows)
        got = out["iv_mispricing_score"].round(9).tolist()
        self.assertEqual(
            len(set(got)), 3,
            "iv_mispricing_score collapsed to one value per expiry: "
            f"{got}. It is clip(-resid*5,0,1)*surf_conf, a PER-ROW term. If "
            "this ever becomes one number per slice again, the 'uniform shift "
            "cannot reorder anything' claim would be restored -- and it was "
            "measured false (within-slice residual spread 1.79e-1).")
        # and it must rise with cheapness (more negative residual = cheaper)
        self.assertLess(got[0], got[1])
        self.assertLess(got[1], got[2])

    def test_the_per_slice_confidence_only_scales_that_per_row_term(self):
        """Halving the fit confidence must halve the score, not flatten it."""
        resids = (-0.02, -0.06, -0.10)
        hi = _score([_contract(strike=100.0 + i, iv_surface_residual=r,
                               iv_surface_confidence=0.9)
                     for i, r in enumerate(resids)])["iv_mispricing_score"]
        lo = _score([_contract(strike=100.0 + i, iv_surface_residual=r,
                               iv_surface_confidence=0.45)
                     for i, r in enumerate(resids)])["iv_mispricing_score"]
        for h, l in zip(hi.tolist(), lo.tolist()):
            self.assertAlmostEqual(l, h / 2.0, places=9)


class TheWeightStaysBelowTheMeasuredReorderThreshold(unittest.TestCase):
    """The accepted-risk guard: fail when the one risky condition returns."""

    def test_live_iv_mispricing_weight_is_small_enough(self):
        with open("config.json") as fh:
            w = json.load(fh)["composite_weights"]["iv_mispricing"]
        self.assertLessEqual(
            w, MAX_SAFE_WEIGHT,
            f"composite_weights.iv_mispricing is {w}, above the "
            f"{MAX_SAFE_WEIGHT} guard. The SVI fit's basin flake was ACCEPTED "
            "on the measurement that top-5/top-10 board membership did not "
            f"move in 42 draws at the live 0.0104. At "
            f"{WEIGHT_AT_WHICH_TOP_N_MOVED} a real board's top-10 DID change "
            "between two runs 0.1s apart. Raising this weight re-opens that, "
            "so re-run the top-N stability measurement before raising the "
            "guard -- do not just edit this number.")


if __name__ == "__main__":
    unittest.main()
