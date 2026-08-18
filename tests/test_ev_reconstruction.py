"""Reconstructing the EV band for trades that closed before schema 21.

`entry_ev_net` / `entry_ev_noise` only exist on 22 rows — 20 of them still
open — so "did CLEAR beat THIN" has n=2 and cannot be asked. The inputs to
those numbers ARE on the historical book: entry_vega, strike, entry date,
expiration, entry_price. With price history sliced as of the entry date, the
band is recomputable for the trades that already closed.

What this must not do:

  - price a multi-leg structure as if it were one leg (a spread's EV is not
    its short leg's — that defect is already on record)
  - see a single bar past the entry date
  - return 0.0 where it means "could not compute" — NULL is not-recorded
  - write into `entry_ev_*`, which would make a reconstruction
    indistinguishable from a scan-time record
"""
import unittest

import numpy as np
import pandas as pd

from src.ev_reconstruction import MIN_BARS, reconstruct_one


def _hist(n=400, start=100.0, seed=0, end="2026-06-30"):
    """A deterministic daily bar series ending on `end`."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.015, n)
    close = start * np.exp(np.cumsum(rets))
    idx = pd.bdate_range(end=pd.Timestamp(end), periods=n)
    return pd.DataFrame({
        "Open": close * 0.999, "High": close * 1.01,
        "Low": close * 0.99, "Close": close,
        "Volume": np.full(n, 1_000_000),
    }, index=idx)


def _trade(**over):
    t = {
        "entry_id": 1, "ticker": "AAPL", "strategy_name": "Long Call",
        "type": "call", "strike": 110.0, "date": "2026-06-30",
        "expiration": "2026-08-14", "entry_price": 3.20,
        "entry_vega": 0.12, "pnl_pct": 0.10,
    }
    t.update(over)
    return t


class ReconstructionTest(unittest.TestCase):
    def test_a_single_leg_trade_reconstructs(self):
        r = reconstruct_one(_trade(), _hist())
        self.assertIsNotNone(r)
        self.assertGreater(r.ev_noise, 0.0)
        self.assertGreater(r.vol_basis, 0.0)

    def test_the_noise_band_matches_the_production_formula(self):
        """The one implementation, not a second copy that agrees by luck."""
        from src.trade_analysis import VOL_FORECAST_RELATIVE_ERROR
        r = reconstruct_one(_trade(entry_vega=0.12), _hist())
        expected = abs(0.12 * 100.0) * VOL_FORECAST_RELATIVE_ERROR * r.vol_basis * 100.0
        self.assertAlmostEqual(r.ev_noise, expected, places=6)

    def test_bars_after_the_entry_date_change_nothing(self):
        """No lookahead: the same trade priced against a history that runs on
        past its entry must be identical."""
        base = _hist(n=400, end="2026-06-30")
        extended = _hist(n=460, end="2026-09-22")
        # Same generator, so the overlapping prefix differs; align instead by
        # truncating the extended frame at the entry date and comparing to a
        # reconstruction that was handed the untruncated frame.
        cut = extended[extended.index <= pd.Timestamp("2026-06-30")]
        a = reconstruct_one(_trade(), cut)
        b = reconstruct_one(_trade(), extended)
        self.assertIsNotNone(a)
        self.assertAlmostEqual(a.vol_basis, b.vol_basis, places=12)
        self.assertAlmostEqual(a.ev_gross, b.ev_gross, places=12)
        self.assertEqual(a.n_bars, b.n_bars)
        del base

    def test_a_multi_leg_structure_is_refused_not_mispriced(self):
        for name in ("Bull Put", "Bear Call", "Iron Condor"):
            self.assertIsNone(reconstruct_one(_trade(strategy_name=name), _hist()),
                              f"{name} was priced as a single leg")

    def test_too_little_history_returns_none_not_zero(self):
        self.assertIsNone(reconstruct_one(_trade(), _hist(n=MIN_BARS - 5)))

    def test_a_missing_vega_returns_none(self):
        self.assertIsNone(reconstruct_one(_trade(entry_vega=None), _hist()))

    def test_an_expired_or_zero_tenor_trade_is_refused(self):
        self.assertIsNone(reconstruct_one(
            _trade(date="2026-06-30", expiration="2026-06-30"), _hist()))

    def test_a_short_position_edge_has_the_sellers_sign(self):
        """The buyer's edge on a seller's board is a defect already on record.

        Same contract, same history: the seller's gross edge is the buyer's
        negated, because one pays the premium and the other receives it.
        """
        h = _hist()
        long_r = reconstruct_one(_trade(strategy_name="Long Put", type="put"), h)
        short_r = reconstruct_one(_trade(strategy_name="Short Put", type="put"), h)
        self.assertIsNotNone(long_r)
        self.assertIsNotNone(short_r)
        self.assertAlmostEqual(long_r.ev_gross, -short_r.ev_gross, places=9)

    def test_sigma_is_the_edge_in_error_bars(self):
        r = reconstruct_one(_trade(), _hist())
        self.assertAlmostEqual(r.sigma_gross, r.ev_gross / r.ev_noise, places=9)

    def test_it_records_which_entry_id_it_describes(self):
        r = reconstruct_one(_trade(entry_id=4242), _hist())
        self.assertEqual(r.entry_id, 4242)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
